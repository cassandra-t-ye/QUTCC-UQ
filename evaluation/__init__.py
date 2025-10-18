import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from kornia.contrib import connected_components
from lpips import LPIPS
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from models.im2im import Im2Im
from models.im2im.add_uncertainty_im2im import ModelWithUncertainty
from models.quantile_uqnet import UNetModel


def get_binary_missed(lower_bound: torch.Tensor, upper_bound: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    missed = (gt < lower_bound) | (gt > upper_bound)
    return missed.to(lower_bound.dtype)

def get_lower_upper_risk(lower: torch.Tensor, upper: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    """Calculate the fraction of missed predictions based on quantile bounds."""
    B, C, H, W = lower.shape
    total_pixels = B * H * W
    missed_low_cnt = (lower > gt).float().sum().item()
    missed_high_cnt = (upper < gt).float().sum().item()

    return missed_low_cnt / total_pixels, missed_high_cnt / total_pixels

def size_stratified_risk(df_intervals_missed: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the average miss‑rate (“risk”) in four quartiles of interval length.

    Parameters
    ----------
    df_intervals_missed : pd.DataFrame
        Must contain
        * ``interval`` – numeric interval sizes
        * ``missed``   – one‑hot 0/1 (or bool) flag indicating a miss

    Returns
    -------
    Dict[str, float]
        Keys are the human‑friendly quartile labels
        ["Short", "Short-Medium", "Medium-Long", "Long"].
        Values are the mean miss‑rate in each length bin.
    """
    labels = ["Short", "Short-Medium", "Medium-Long", "Long"]
    df = df_intervals_missed.copy()
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")
    df["missed"]   = df["missed"].astype(float)
    df = df.dropna(subset=["interval", "missed"])
    if len(df["interval"].unique()) < len(labels):
        return {label: 0.0 for label in labels}
    df["bin"] = pd.qcut(df["interval"], len(labels), labels=labels, duplicates='drop')
    risk = df.groupby("bin", observed=True)["missed"].mean()
    return {label: risk.get(label, 0) for label in labels}

def return_calibrated_bounds(denoised_image: torch.Tensor, lam1: float) -> Tuple[torch.Tensor, torch.Tensor]:
    lower = denoised_image[:, 1, :, :] - lam1 * (denoised_image[:, 1, :, :] - denoised_image[:, 0, :, :])
    upper = lam1 * (denoised_image[:, 2, :, :] - denoised_image[:, 1, :, :]) + denoised_image[:, 1, :, :]
    return lower, upper

def check_quantile_crossings(dataloader: DataLoader, model: nn.Module, 
                             quantiles: torch.Tensor, device: str) -> Dict[str, int]:
    Q = len(quantiles)
    violations, total = 0, 0

    with torch.no_grad():
        model.eval()
        for noisy, _ in tqdm(dataloader, desc="Checking Quantile Crossings"):
            noisy: torch.Tensor = noisy.to(device)
            B, C, H, W = noisy.shape

            noisy = noisy.unsqueeze(0).expand(Q, -1, -1, -1, -1).reshape(-1, C, H, W)   # [Q * B, C, H, W]
            q = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                        # [Q * B]
            pred: torch.Tensor = model(noisy, q)                                        # [Q * B, 1, H, W]
            pred = pred.view(Q, B, 1, H, W).squeeze(2)                                  # [Q, B, H, W]

            violations += (pred[:-1] > pred[1:]).sum().item()
            total += (Q - 1) * B * H * W

    print(f"Total quantile crossing violations: {violations}")
    print(f"Percentage of violations: {violations / total * 100:.2f}%")

    del noisy, pred, q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return {"violations": violations, "total": total}

def get_risk(dataloader: DataLoader, im2im_model: ModelWithUncertainty, im2im_lambda: float,
             im2im_deep_model: ModelWithUncertainty, im2im_deep_lambda: float,
             quantile_model: UNetModel, lower_q: float, upper_q: float, device: str) -> Dict[str, Dict[str, float]]:
    totals = defaultdict(lambda: {"missed": 0, "n": 0, "mean_risk": 0, "std_risk": 0})
    per_image_risks = defaultdict(list)
    
    with torch.no_grad():
        im2im_model.eval(); im2im_deep_model.eval(); quantile_model.eval()
        for (noisy, clean) in tqdm(dataloader, desc="Calculating calibration risk"):
            noisy, clean = noisy.to(device), clean.to(device)
            B, C, H, W = noisy.shape
            num_pixels = B * H * W        

            # --------- im2im model ---------
            assert isinstance(im2im_model.baseModel, Im2Im)
            pred          = im2im_model(noisy)
            lower, upper  = return_calibrated_bounds(pred, im2im_lambda)
            lower, upper  = lower.view(B, 1, H, W), upper.view(B, 1, H, W)             # [B, 1, H, W]
            missed_bin = get_binary_missed(lower, upper, clean).view(B, -1)          # [B, H * W]
            per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            per_image_risks["im2im"].extend((per_image_missed_counts / (H * W)).cpu().numpy())

            totals["im2im"]["missed"] += per_image_missed_counts.sum().item()
            totals["im2im"]["n"]      += num_pixels

            # --------- im2im deep model ---------
            assert isinstance(im2im_deep_model.baseModel, UNetModel)
            timevect = torch.full((B,), 0.5, device=device, dtype=torch.float32)
            pred = im2im_deep_model(noisy, timevect)
            lower, upper = return_calibrated_bounds(pred, im2im_deep_lambda)
            lower, upper = lower.view(B, 1, H, W), upper.view(B, 1, H, W)             # [B, 1, H, W]
            missed_bin = get_binary_missed(lower, upper, clean).view(B, -1)          # [B, H * W]
            per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            per_image_risks["im2im_deep"].extend((per_image_missed_counts / (H * W)).cpu().numpy())

            totals["im2im_deep"]["missed"] += per_image_missed_counts.sum().item()
            totals["im2im_deep"]["n"]      += num_pixels

            # --------- quantile model ---------
            quantiles = torch.tensor([lower_q, upper_q], device=device, dtype=torch.float32)
            quantiles = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                             # [2 * B]
            noisy     = noisy.unsqueeze(0).expand(2, -1, -1, -1, -1).reshape(-1, C, H, W)            # [2 * B, C, H, W]
            pred      = quantile_model(noisy, quantiles).view(2, B, 1, H, W).permute(1, 0, 2, 3, 4)  # [B, 2, 1, H, W]
            lower_pred, upper_pred = pred[:, 0], pred[:, 1]                                          # [B, 1, H, W]
            missed_bin_q = get_binary_missed(lower_pred, upper_pred, clean).view(B, -1)    # [B, H * W]
            per_image_missed_counts_q = missed_bin_q.sum(dim=1)
            per_image_risks["quantile"].extend((per_image_missed_counts_q / (H * W)).cpu().numpy())
            
            totals["quantile"]["missed"] += per_image_missed_counts_q.sum().item()
            totals["quantile"]["n"]      += num_pixels

    for method in totals.keys():
        totals[method]["mean_risk"] = np.mean(per_image_risks[method])
        totals[method]["std_risk"]  = np.std(per_image_risks[method])
    
    del dataloader, noisy, clean, pred, lower, upper, lower_pred, upper_pred
    del missed_bin, missed_bin_q, per_image_missed_counts, per_image_missed_counts_q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return totals                             
        
def get_test_intervals_missed(net_type: Literal["im2im", "unet_im2im", "unet_quantile"],
                              test_dataloader: DataLoader, model: nn.Module,
                              lambda_or_lower_q: float, upper_q: Optional[float] = None,
                              device: torch.device = torch.device("cuda")) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[pd.DataFrame] = []
    total = {"missed": 0, "n": 0, "mean_risk": 0, "std_risk": 0}
    per_image_risks: List[float] = []
    
    model.eval()
    
    with torch.no_grad():
        for noisy, clean in tqdm(test_dataloader, desc=f"Evaluating {net_type}"):
            noisy, clean = noisy.to(device), clean.to(device)
            B, C, H, W = noisy.shape
            num_pixels = B * H * W

            if net_type == "im2im":
                assert isinstance(model.baseModel, Im2Im)
                pred = model(noisy)
                lower, upper = return_calibrated_bounds(pred, lambda_or_lower_q)
            elif net_type == "unet_im2im":
                assert isinstance(model.baseModel, UNetModel)
                timevect = torch.full((B,), 0.5, device=device, dtype=torch.float32)
                pred = model(noisy, timevect)
                lower, upper = return_calibrated_bounds(pred, lambda_or_lower_q)
            elif net_type == "unet_quantile":
                lower_q = lambda_or_lower_q
                assert upper_q is not None, "Upper quantile must be provided for quantile model"
                quantiles = torch.tensor([lower_q, upper_q], device=device, dtype=torch.float32)
                quantiles = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                # [2 * B]
                noisy = noisy.unsqueeze(0).expand(2, -1, -1, -1, -1).reshape(-1, C, H, W)   # [2 * B, C, H, W]
                pred = model(noisy, quantiles).view(2, B, 1, H, W).permute(1, 0, 2, 3, 4)   # [B, 2, 1, H, W]
                lower, upper = pred[:, 0], pred[:, 1]                                       # [B, 1, H, W]

            lower, upper = lower.view(B, 1, H, W), upper.view(B, 1, H, W)                   # [B, 1, H, W]
            interval_size = (upper - lower).view(-1)
            missed_bin = get_binary_missed(lower, upper, clean).view(B, -1)

            per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            per_image_risks.extend((per_image_missed_counts / (H * W)).cpu().numpy())

            total["missed"] += per_image_missed_counts.sum().item()
            total["n"]      += num_pixels

            rows.append(pd.DataFrame({
                "method":   "im2im_deep" if net_type == "unet_im2im" else net_type,
                "interval": interval_size.cpu().numpy().astype('float32'),
                "missed":   missed_bin.view(-1).cpu().numpy().astype('bool'),
            }))
    
    total["mean_risk"] = float(np.mean(per_image_risks))
    total["std_risk"]  = float(np.std(per_image_risks))

    df_intervals_missed = pd.concat(rows, ignore_index=True, copy=False)
    df_intervals_missed["method"] = df_intervals_missed["method"].astype("category")

    del noisy, clean, pred, lower, upper, interval_size, missed_bin
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    return df_intervals_missed, total

def connected_error_region_score(pred: torch.Tensor, clean: torch.Tensor):
    B = pred.shape[0]
    error_map = torch.abs(pred - clean)
    binary_mask = (error_map > error_map.mean()).to(error_map.dtype)
    labeled_map = connected_components(binary_mask)

    batch_scores = []
    for i in range(B):
        unique_labels, counts = torch.unique(labeled_map[i], return_counts=True)
        if len(counts) > 1:
            score = torch.max(counts[1:])
        else:
            score = torch.tensor(0.0, device=error_map.device)
        batch_scores.append(score)
    return torch.stack(batch_scores).float()

def max_localized_average_error(pred: torch.Tensor, clean: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    B = pred.shape[0]
    error_map = torch.abs(pred - clean)
    padding = kernel_size // 2
    pooled_map = F.avg_pool2d(error_map, kernel_size=kernel_size, stride=1, padding=padding)

    return pooled_map.view(B, -1).max(dim=1).values.float()


def get_largest_residual_indices(model: UNetModel, dataloader: DataLoader, 
                                 device: torch.device, top_k: int = 10, 
                                 metric: Literal["l2", "l1", "ssim", 'lpips', 'connected_error', 'localized_average']="l2") -> Tuple[torch.Tensor, List[float]]:
    """
    Evaluate `model` on `dataloader` once and return the dataset indices of the `top_k`
    samples with the largest residual (per–image error).
    """
    model.eval()
    errs = []
    if metric == "l2":
        loss = nn.MSELoss(reduction="none").to(device)
    elif metric == "l1":
        loss = nn.L1Loss(reduction="none").to(device)
    elif metric == "ssim":
        loss = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    elif metric == "lpips":
        loss = LPIPS(net="vgg").to(device)
    elif metric == "connected_error":
        loss = connected_error_region_score
    elif metric == "localized_average":
        loss = max_localized_average_error
    else:
        raise ValueError(f"Unknown metric: {metric}")

    with torch.no_grad():
        for noisy, clean in tqdm(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            B = noisy.shape[0]

            quantiles = torch.full((B,), 0.5, device=device, dtype=torch.float32)
            pred      = model(noisy, quantiles)

            error = loss(pred, clean).view(B, -1).mean(dim=1)
            errs.append(error.cpu())
    
    errs_tensor = torch.cat(errs, dim=0)
    if metric == "ssim":
        worst_vals, worst_pos = torch.topk(errs_tensor, top_k, dim=0, largest=False, sorted=True)
    else:
        worst_vals, worst_pos = torch.topk(errs_tensor, top_k, dim=0, largest=True, sorted=True)

    del noisy, clean, quantiles, pred, error
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return worst_pos, worst_vals.tolist()