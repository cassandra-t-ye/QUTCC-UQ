import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
from torch import nn
from tqdm import tqdm

from models.checkpointing import load_checkpoint_for_inference
from evaluation import (check_quantile_crossings, get_risk,
                        size_stratified_risk)
from evaluation.calibration import (compute_optimal_lambdas,
                                    compute_optimal_lambdas_quantile)
from evaluation.calibration_utils import (CalibrationManager,
                                          get_calib_dataloader,
                                          AnalysisManager)
from utils import _prepare_for_json, json_converter, print_memory_stats

from datasets.fastmri import FastMRIDataset
import datasets.fastmri.utils as fastmri_utils
from torch.utils.data import TensorDataset, DataLoader
from datasets.fmd.data_loader import (load_noisy_single_loader,
                                                load_real_noise_loader)
from typing import Any, Callable, Literal
from datasets.fastmri import FastMRIDataset
from datasets.LidcIdri.LIDCLDRI import load_ct_dataloader
from datasets.bsccm import get_bsccm_calib, get_bsccm_test

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import lpips


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ROOT = "/path/to/data/here"
ROOT = "/home/bl788/QUTCC-UQ"
DATAROOT = "/share/monakhova/Cassandra_data/UQNet_proj" # make the same as ROOT if data is in the same directory

with open(Path(__file__).parent / "best_runs_mri_test.json", "r") as f:
    BEST_RUNS = json.load(f)

# Replace placeholder "ROOT/" with the actual ROOT
def inject_root(d, replace="ROOT", with_path=ROOT):
    for k, v in d.items():
        if isinstance(v, str) and v.startswith(replace):
            d[k] = v.replace(replace, with_path, 1)
        elif isinstance(v, dict):
            inject_root(v, replace=replace, with_path=with_path)

inject_root(BEST_RUNS, replace="ROOT", with_path=ROOT)
inject_root(BEST_RUNS, replace="DATAROOT", with_path=DATAROOT)

print(BEST_RUNS)


def run(tasks: List[str] = None,
        calib_subset: int = 2000,
        test_subset: int = 2000,
        intervals_subset: int = 500_000,
        output_dir: str = None,
        device: torch.device = None,
        use_submitit: bool = True,
        partition: str = "gpu") -> Tuple[Dict, Dict, pd.DataFrame]:
    print(f"Device {device}")

    save_dir = Path(output_dir) if output_dir else Path(f"uqnet_results_{time.strftime('%Y-%m-%d-%H.%M')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Final results will be saved in: {save_dir}")

    if tasks is None:
        tasks = list(BEST_RUNS.keys())

    summary, all_stratified, all_df = {}, {}, []
    for task in tasks:
        print(f"\n--- Starting Experiment: {task} ---")
        if task not in BEST_RUNS:
            raise ValueError(f"Task '{task}' not found in BEST_RUNS. Available: {list(BEST_RUNS.keys())}")
        
        info = BEST_RUNS[task]
        experiment_type = info['experiment_type']
        calib_results: List[Dict[str, Any]]
        try:
            calib_results = [
                {'net_type': 'im2im', 'lambda': info['im2im_lambda'], 'epoch': info['im2im_epoch']},
                {'net_type': 'unet_im2im', 'lambda': info['unet_im2im_lambda'], 'epoch': info['unet_im2im_epoch']},
                {'net_type': 'unet_quantile', 'lower_q': info['quantile_lower_q'], 'upper_q': info['quantile_upper_q'], 'epoch': info['quantile_epoch']}
            ]

            print(
                f"For task {task}, found pre-selected lambda/lower_q/upper_q. Using precomputed calibration parameters."
            )
        except KeyError:
            print(
                f"Parameter loading failed because one of lambda/lower_q/upper_q was not included in BEST_RUNS for task {task}. Calibrating {task} from scratch."
            )

            if use_submitit:
                calibration_manager = CalibrationManager(
                    experiment_type, info,
                    calib_subset=calib_subset,
                    partition=partition,
                    ignore_checkpoints=False
                )
                requests = [
                    ("im2im", info['im2im_epoch']),
                    ("unet_im2im", info['unet_im2im_epoch']),
                    ("unet_quantile", info['quantile_epoch'])
                ]
                calib_results = calibration_manager.calibrate_multiple(requests)
            else:
                _, im2im_lambda = calibrate_im2im(experiment_type, info, device, calib_subset)
                _, im2im_deep_lambda = calibrate_im2im_deep(experiment_type, info, device, calib_subset)
                _, lower_q, upper_q = calibrate_quantile(experiment_type, info, device, calib_subset)
                calib_results = [
                    {'net_type': 'im2im', 'lambda': im2im_lambda, 'epoch': info['im2im_epoch']},
                    {'net_type': 'unet_im2im', 'lambda': im2im_deep_lambda, 'epoch': info['unet_im2im_epoch']},
                    {'net_type': 'unet_quantile', 'lower_q': lower_q, 'upper_q': upper_q, 'epoch': info['quantile_epoch']},
                ]

        # calib_risk = get_calibration_risk(experiment_type, info, device, calib_subset)
        # violations, total = check_crossings(experiment_type, info, device, calib_subset)

        analysis_manager = AnalysisManager(experiment_type, info, test_subset, intervals_subset, 
                                           partition, ignore_checkpoints=False)
        analysis_results = analysis_manager.analyze_multiple(calib_results)

        dfs_list = []
        risk_dict = {}
        for result in analysis_results:
            net_type = result['net_type']
            model_key = "im2im_deep" if net_type == "unet_im2im" else net_type

            df = result['df']
            df['method'] = model_key
            dfs_list.append(df)
            risk_dict[model_key] = result['risk']
        
        df_intervals_missed = pd.concat(dfs_list, ignore_index=True, copy=False)
        df_intervals_missed["experiment"] = task
        df_intervals_missed["experiment"] = df_intervals_missed["experiment"].astype("category")
        all_df.append(df_intervals_missed)
        
        stratified_results = compute_stratified_risk(df_intervals_missed)
        # for method in ['im2im', 'im2im_deep', 'unet_quantile']:
        for method in ['im2im_deep', 'unet_quantile']:
            print(f"{method} Stratified Risk")
            print(json.dumps(_prepare_for_json(stratified_results[method]), indent=4))
            print(f"{method} Average Risk: {risk_dict[method]['mean_risk']:.4f}")
        all_stratified[task] = stratified_results
        
        summary[task] = {
            # "calib_risk": calib_risk,
            # "crossings_violations": violations,
            # "crossings_total": total,
            "risk": risk_dict,
            "mean_interval_length": {
                "im2im": df_intervals_missed[df_intervals_missed["method"] == "im2im"]["interval"].mean(),
                "im2im_deep": df_intervals_missed[df_intervals_missed["method"] == "im2im_deep"]["interval"].mean(),
                "unet_quantile": df_intervals_missed[df_intervals_missed["method"] == "unet_quantile"]["interval"].mean(),
            },
            "std_interval_length": {
                "im2im": df_intervals_missed[df_intervals_missed["method"] == "im2im"]["interval"].std(),
                "im2im_deep": df_intervals_missed[df_intervals_missed["method"] == "im2im_deep"]["interval"].std(),
                "unet_quantile": df_intervals_missed[df_intervals_missed["method"] == "unet_quantile"]["interval"].std(),
            },
        }

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print_memory_stats()
    
    if all_df:
        all_df = pd.concat(all_df, ignore_index=True, copy=False)
        all_df.to_feather(save_dir / "all_df.feather", compression="zstd")

        with open(save_dir / "summary.json", "w") as f:
            json.dump(_prepare_for_json(summary), f, indent=4, default=json_converter)
        with open(save_dir / "all_stratified.json", "w") as f:
            json.dump(_prepare_for_json(all_stratified), f, indent=4, default=json_converter)
        
        if len(summary) > 1:
            create_stratified_risk_plot(all_stratified, save_dir)
            create_violin_plot(all_df, summary, save_dir)
        
        print(f"\nAll experiments processed. Final aggregated results saved in: {save_dir}")
    else:
        print("No data to save. All DataFrames were empty.")
    
    return summary, all_stratified, all_df

def manage_checkpoint(checkpoint_file: Path, compute_func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Manages loading from or computing and saving to a checkpoint file.
    Supported extensions: .pt, .json, .feather.
    """
    file_type = checkpoint_file.suffix
    load_device = kwargs.pop('map_location', None) # For torch.load

    if checkpoint_file.exists():
        print(f"Loading from checkpoint: {checkpoint_file}")
        if file_type == ".pt":
            return torch.load(checkpoint_file, map_location=load_device)
        elif file_type == ".json":
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        elif file_type == ".feather":
            return pd.read_feather(checkpoint_file)
        else:
            raise ValueError(f"Unsupported file type for checkpoint: {file_type}")
    else:
        # Remove map_location if it was passed for torch.load, as compute_func won't expect it
        # (unless it's a specific kwarg for compute_func, handled by its signature)
        if 'map_location' in compute_func.__code__.co_varnames and load_device:
             kwargs['map_location'] = load_device

        result = compute_func(*args, **kwargs)
        
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to: {checkpoint_file}")
        if file_type == ".pt":
            torch.save(result, checkpoint_file)
        elif file_type == ".json":
            serializable_result = _prepare_for_json(result)
            with open(checkpoint_file, "w") as f:
                json.dump(serializable_result, f, indent=4)
        elif file_type == ".feather":
            if not isinstance(result, pd.DataFrame):
                raise ValueError(f"Result for .feather checkpoint must be a pandas DataFrame, got {type(result)}")
            result.to_feather(checkpoint_file, compression="zstd")
        else:
            raise ValueError(f"Unsupported file type for checkpoint: {file_type}")
        return result
    
def calibrate_im2im(experiment_type: str, info: Dict, device: torch.device, 
                    calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float]:
    epoch = epoch if epoch is not None else info["im2im_epoch"]
    im2im_model, run_folder = load_checkpoint_for_inference(net="im2im", in_channels=info["in_channels"], 
                                     experiment_type=experiment_type, run_folder_root=info["im2im_root"], 
                                     epoch=epoch, device=device, experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net='im2im', 
                                            batch_size_multiplier=1.5, calib_subset=calib_subset)
    
    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    alpha, min_l, max_l, num_l = 0.1, 0, 2, 300
    im2im_lambda_file = analysis_dir / f"im2im_epoch{epoch}_lambda.pt"
    im2im_lambda = manage_checkpoint(im2im_lambda_file, compute_optimal_lambdas,
                                     dataloader=calib_dataloader, model=im2im_model, alpha=alpha,
                                     min_lam=min_l, max_lam=max_l, num_lam=num_l, device=device)
    del calib_dataloader
    return im2im_model, im2im_lambda

def calibrate_im2im_deep(experiment_type: str, info: Dict, device: torch.device, 
                         calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float]:
    epoch = epoch if epoch is not None else info["unet_im2im_epoch"]
    im2im_deep_model, run_folder = load_checkpoint_for_inference(net="unet_im2im", in_channels=info["in_channels"],
                                           experiment_type=experiment_type, run_folder_root=info["unet_im2im_root"],
                                           epoch=epoch, device=device, experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net="unet_im2im", 
                                            batch_size_multiplier=1.5, calib_subset=calib_subset)

    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    alpha, min_l, max_l, num_l = 0.1, 0, 2, 300
    im2im_deep_lambda_file = analysis_dir / f"unet_im2im_epoch{epoch}_lambda.pt"
    im2im_deep_lambda = manage_checkpoint(im2im_deep_lambda_file, compute_optimal_lambdas,
                                          dataloader=calib_dataloader, model=im2im_deep_model, alpha=alpha,
                                          min_lam=min_l, max_lam=max_l, num_lam=num_l, device=device)
    del calib_dataloader
    return im2im_deep_model, im2im_deep_lambda

def calibrate_quantile(experiment_type: str, info: Dict, device: torch.device, 
                       calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float, float]:
    epoch = epoch if epoch is not None else info["quantile_epoch"]
    quantile_model, run_folder = load_checkpoint_for_inference(net="unet_quantile", in_channels=info["in_channels"],
                                        experiment_type=experiment_type, run_folder_root=info["quantile_root"],
                                        epoch=epoch, device=device, 
                                        experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net="unet_quantile", calib_subset=calib_subset)

    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    target_alpha = 0.1
    lower_min, lower_max = 0, 0.1
    upper_min, upper_max = 0.9, 1
    quantile_qs_file = analysis_dir / f"quantile_qs_epoch{epoch}.json"
    quantile_qs = manage_checkpoint(quantile_qs_file, compute_optimal_lambdas_quantile,
                                     dataloader=calib_dataloader, model=quantile_model, alpha=target_alpha,
                                     lower_min=lower_min, lower_max=lower_max, 
                                     upper_min=upper_min, upper_max=upper_max,
                                     max_iterations=20, device=device)
    
    del calib_dataloader
    return quantile_model, quantile_qs["lower_q"], quantile_qs["upper_q"]

def compute_stratified_risk(df_intervals_missed: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stratified_results = {}

    for method in ["im2im", "im2im_deep", "unet_quantile"]:
        method_df = df_intervals_missed[df_intervals_missed["method"] == method][["interval", "missed"]]
        stratified_results[method] = size_stratified_risk(method_df)

    return stratified_results

def create_stratified_risk_plot(all_stratified: Dict[str, Dict[str, Dict[str, float]]], save_dir: Path):
    categories  = ["Short", "Short-Medium", "Medium-Long", "Long"]
    experiments = list(all_stratified.keys())
    methods     = ["im2im_deep", "unet_quantile"]

    xtick_labels = []
    for exp in experiments:
        xtick_labels.extend([
            f"{exp}\nIm2Im-Deep",
            f"{exp}\nQUTCC"
        ])

    data = {cat: [] for cat in categories}
    for exp in experiments:
        for m in methods:
            for cat in categories:
                data[cat].append(all_stratified[exp][m][cat])

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.18
    x = np.arange(len(xtick_labels))

    colors_im2im_deep = ['#e6e6ff', '#bfbfff', '#9999ff', '#7373ff']
    colors_quantile = ["#ffe6e6", '#ffbfbf', '#ff9999', '#ff7373']
    palette_map = {"im2im_deep": colors_im2im_deep, "unet_quantile": colors_quantile}
    for i, cat in enumerate(categories):
        bar_colors = []
        for lbl in xtick_labels:
            method = "im2im_deep" if "Deep" in lbl else "unet_quantile"
            bar_colors.append(palette_map[method][i])
        ax.bar(x + (i - 1.5) * bar_width,   
            data[cat],
            width=bar_width,
            label=cat,
            color=bar_colors,
            edgecolor='white',
            linewidth=0.5)

    ax.set_xticks(x, labels=xtick_labels, fontsize=10, ha="center")
    ax.tick_params(axis="y", labelsize=16)

    ax.set_ylim(0, 0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0.10, ls="--", color="gray", lw=1)
    ax.text(len(x)-0.3, 0.101, r"$\alpha$", va="bottom", ha="left", color="gray", fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    legend_handles = []
    for i, cat in enumerate(categories):
        g_rgb = to_rgb(colors_im2im_deep[i])
        r_rgb = to_rgb(colors_quantile[i])
        blended = to_hex([(g + r) / 2 for g, r in zip(g_rgb, r_rgb)])
        legend_handles.append(Patch(facecolor=blended, edgecolor='white',
                                    linewidth=0.5, label=cat))

    ax.legend(handles     = legend_handles,
              fontsize    = 16,
              frameon     = True,
              handlelength= 2.5,
              bbox_to_anchor=(1, 1),
              loc='upper right')
    
    fig.tight_layout()
    fig.savefig(save_dir / "stratified_risk_all_tasks.pdf", dpi=300, bbox_inches="tight")
    print("Saved stratified risk plot to:", save_dir / "stratified_risk_all_tasks.pdf")

def create_violin_plot(all_df: pd.DataFrame, summary: Dict[str, Dict], save_dir: Path):
    df_violin = all_df[(all_df["method"] == "im2im_deep") | 
                    (all_df["method"] == "unet_quantile")][["experiment", "method", "interval"]]
    method_order = ["im2im_deep", "unet_quantile"]
    method_labels = {"im2im_deep": "Im2Im-Deep", "unet_quantile": "QUTCC"}
    palette = {'im2im_deep': '#7373ff', 'unet_quantile': '#ff7373'}

    multiplier = 2.5
    fig, ax = plt.subplots(figsize=(5.52 * multiplier, 2.5 * multiplier))
    sns.violinplot(
        data         = df_violin,
        x            = "experiment",
        y            = "interval",
        hue          = "method",
        order        = list(summary.keys()),
        hue_order    = method_order,
        split        = True,
        density_norm = "count",
        cut          = 0,
        inner        = None,
        palette      = palette,
        ax           = ax
    )

    ax.set_xlabel("") # Method
    ax.set_ylabel("") # Interval Size
    ax.set_ylim(0, 0.35)

    legend_elements = [
        Patch(facecolor=palette['im2im_deep'], label='Im2Im-Deep'), 
        Patch(facecolor=palette['unet_quantile'], label='QUTCC')
    ]
    ax.legend(handles=legend_elements, fontsize=8 * multiplier, loc='upper right', bbox_to_anchor=(1.2, 1.2))

    ax.tick_params(axis="x", which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8 * multiplier)
    ax.spines[['top', 'right']].set_visible(False)    
    fig.tight_layout()

    violin_plot_path = save_dir / "interval_length_split_violin_all_tasks.png"
    fig.savefig(violin_plot_path, dpi=300, bbox_inches="tight")
    print("Saved split violin plot to:", violin_plot_path)

def get_test_dataloader(experiment_type, info, net=Literal['im2im', 'unet_quantile'], batch_size_multiplier=0.5, test_subset=200):
    if experiment_type == "mri":
        mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
        test_dataset = FastMRIDataset(info["test_data_path"], normalize_input='standard', normalize_output='min-max', 
                                      mask_info=mask_info, num_volumes=20)
        fastmri_utils.normalize_dataset(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=int(info["batch_size"] * batch_size_multiplier), 
                                     shuffle=False, num_workers=1, pin_memory=False)
    
    elif experiment_type in ["gaussian", "poisson"]:
        noise_type, min_noise, max_noise, num_noise = experiment_type, info["min_noise"], info["max_noise"], info["num_noise"]
        test_dataloader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, 
                                                   image_folder=info["test_data_path"], 
                                                   batch_size=int(info["batch_size"] * batch_size_multiplier),
                                                   types=None, transform=None)
    
    elif experiment_type == "qpi":
        run_folder = info["experiments_folder"] + f"/{experiment_type}/{info[f'{net}_root']}"
        test_indices_path = run_folder + "/test_indices.json"
        with open(test_indices_path, "r") as f:
            test_indices = json.load(f)
        test_dataloader = get_bsccm_test(test_indices[:test_subset], info["data_root"], 
                                         batch_size=int(info["batch_size"] * batch_size_multiplier), transform=None, 
                                         normalization='min-max', num_workers=0, pin_memory=False)

    elif experiment_type == "real_noise_mice":
        gt_pathway = info["test_data_path"] + "/gt"
        noisy_pathway = info["test_data_path"] + "/noisy"
        test_dataloader = load_real_noise_loader(noisy_path=noisy_pathway, gt_path=gt_pathway, batch_size=int(info["batch_size"] * batch_size_multiplier))

    elif experiment_type == "CT":
        test_dataloader = load_ct_dataloader(root_path=info["test_data_path"], dataset_type="test", batch_size = info["batch_size"], steps=info["steps"])


    print(f"Test dataset size: {len(test_dataloader.dataset)}")
    return test_dataloader

# rewritten by ChatGPT for clarity
def calculate_statistics(
    task: str,
    seed: int = 1,
    device: torch.device = None,
    test_subset: int = 2000,
    batch_size_multiplier: float = 0.7,
    show_progress: bool = True,
    return_raw: bool = False,
):
    """
    Run evaluation for a given task (matches keys in BEST_RUNS).

    Returns:
        results (dict): Nested dict with mean/std for MSE, SSIM, PSNR, LPIPS
                        for im2im, QUTCC (unet_quantile), and im2im_deep.
        (optionally) raw (dict of tensors): Per-sample metric tensors if return_raw=True.
    """
    # ── Setup ──────────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # ── Load models ───────────────────────────────────────────────────────────────
    im2im_model_desc = "im2im"
    im2im_model, _ = load_checkpoint_for_inference(
        net="im2im",
        in_channels=BEST_RUNS[task]["in_channels"],
        experiment_type=BEST_RUNS[task]["experiment_type"],
        run_folder_root=BEST_RUNS[task][f"{im2im_model_desc}_root"],
        epoch=BEST_RUNS[task][f"{im2im_model_desc}_epoch"],
        device=device,
    )

    quant_model, _ = load_checkpoint_for_inference(
        net="unet_quantile",
        in_channels=BEST_RUNS[task]["in_channels"],
        experiment_type=BEST_RUNS[task]["experiment_type"],
        run_folder_root=BEST_RUNS[task]["quantile_root"],
        epoch=BEST_RUNS[task]["quantile_epoch"],
        device=device,
    )

    unet_im2im_model, _ = load_checkpoint_for_inference(
        net="unet_im2im",
        in_channels=BEST_RUNS[task]["in_channels"],
        experiment_type=BEST_RUNS[task]["experiment_type"],
        run_folder_root=BEST_RUNS[task]["unet_im2im_root"],
        epoch=BEST_RUNS[task]["unet_im2im_epoch"],
        device=device,
    )

    # ── Dataloader & metrics ──────────────────────────────────────────────────────
    dataloader = get_test_dataloader(
        task, BEST_RUNS[task], net="unet_quantile",
        batch_size_multiplier=batch_size_multiplier,
        test_subset=test_subset,
    )

    batch_mse = torch.nn.MSELoss(reduction="none")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=(1, 2, 3)).to(device)
    lpips_metric = lpips.LPIPS(net="vgg").to(device)

    im2im_mse_acc, im2im_ssim_acc, im2im_psnr_acc, im2im_lpips_acc = [], [], [], []
    quantile_mse_acc, quantile_ssim_acc, quantile_psnr_acc, quantile_lpips_acc = [], [], [], []
    unet_im2im_mse_acc, unet_im2im_ssim_acc, unet_im2im_psnr_acc, unet_im2im_lpips_acc = [], [], [], []

    iterator = tqdm(dataloader) if show_progress else dataloader

    # ── Loop ──────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in iterator:
            noisy, clean = batch
            clean_im2im = clean[:, 0, :, :].to(device)  # (kept from your code)
            clean = clean.to(device)

            # Im2Im
            curr_denoised_lam = im2im_model(noisy.to(device))
            im2im_output = torch.unsqueeze(curr_denoised_lam[:, 1, :, :], 1)
            im2im_clean = clean

            out = batch_mse(im2im_output, im2im_clean).view(im2im_output.size(0), -1).mean(dim=1)
            im2im_mse_acc.append(out)
            im2im_ssim_acc.append(ssim_metric(im2im_output, im2im_clean).view(im2im_output.size(0), -1).mean(dim=1))
            im2im_psnr_acc.append(psnr_metric(im2im_output, im2im_clean).view(im2im_output.size(0), -1).mean(dim=1))

            # LPIPS expects [-1,1] RGB
            im2im_in  = (im2im_output.clamp(0,1) * 2.0 - 1.0).expand(-1, 3, -1, -1)
            clean_in  = (im2im_clean.clamp(0,1) * 2.0 - 1.0).expand(-1, 3, -1, -1)
            im2im_lpips_acc.append(lpips_metric(im2im_in, clean_in))

            # unet_quantile (QUTCC @ q=0.5)
            q_prediction = torch.tensor([0.5], device=device, dtype=torch.float32)
            quantile_output = quant_model(noisy.to(device), q_prediction)
            quantile_clean = clean

            out = batch_mse(quantile_clean, quantile_output).view(quantile_clean.size(0), -1).mean(dim=1)
            quantile_mse_acc.append(out)
            quantile_ssim_acc.append(ssim_metric(quantile_output, quantile_clean).view(quantile_output.size(0), -1).mean(dim=1))
            quantile_psnr_acc.append(psnr_metric(quantile_output, quantile_clean).view(quantile_output.size(0), -1).mean(dim=1))

            q_out  = (quantile_output.clamp(0,1) * 2.0 - 1.0).expand(-1, 3, -1, -1)
            q_clean = (quantile_clean.clamp(0,1) * 2.0 - 1.0).expand(-1, 3, -1, -1)
            quantile_lpips_acc.append(lpips_metric(q_out, q_clean))

            # U-Net Im2Im (im2im_deep)
            timevect = torch.full((noisy.shape[0],), 0.5, device=device, dtype=torch.float32)
            unet_im2im_output = unet_im2im_model(noisy.to(device), timevect)
            unet_im2im_output = torch.unsqueeze(unet_im2im_output[:, 1, :, :], 1)
            unet_im2im_clean = clean

            out = batch_mse(unet_im2im_clean, unet_im2im_output).view(unet_im2im_output.size(0), -1).mean(dim=1)
            unet_im2im_mse_acc.append(out)
            unet_im2im_ssim_acc.append(ssim_metric(unet_im2im_output, unet_im2im_clean).view(unet_im2im_output.size(0), -1).mean(dim=1))
            unet_im2im_psnr_acc.append(psnr_metric(unet_im2im_output, unet_im2im_clean).view(unet_im2im_output.size(0), -1).mean(dim=1))

            ui_out  = (unet_im2im_output.clamp(0,1) * 2.0 - 1.0).expand(-1, 3, -1, -1)
            ui_clean = (unet_im2im_clean.clamp(0,1) * 2.0 - 1.0).expand(-1, 3, -1, -1)
            unet_im2im_lpips_acc.append(lpips_metric(ui_out, ui_clean))

    # ── Concatenate and summarize ─────────────────────────────────────────────────
    def _cat(x): return torch.cat(x, dim=0)
    im2im_mse_acc, im2im_ssim_acc, im2im_psnr_acc, im2im_lpips_acc = map(_cat, [im2im_mse_acc, im2im_ssim_acc, im2im_psnr_acc, im2im_lpips_acc])
    quantile_mse_acc, quantile_ssim_acc, quantile_psnr_acc, quantile_lpips_acc = map(_cat, [quantile_mse_acc, quantile_ssim_acc, quantile_psnr_acc, quantile_lpips_acc])
    unet_im2im_mse_acc, unet_im2im_ssim_acc, unet_im2im_psnr_acc, unet_im2im_lpips_acc = map(_cat, [unet_im2im_mse_acc, unet_im2im_ssim_acc, unet_im2im_psnr_acc, unet_im2im_lpips_acc])

    print(f"im2im_uq MSE: {im2im_mse_acc.mean().item():.7f} +/- {im2im_mse_acc.std().item():.7f}")
    print(f"im2im_uq SSIM: {im2im_ssim_acc.mean().item():.3f} +/- {im2im_ssim_acc.std().item():.3f}")
    print(f"im2im_uq PSNR: {im2im_psnr_acc.mean().item():.3f} +/- {im2im_psnr_acc.std().item():.3f}")
    print(f"im2im_uq LPIPS: {im2im_lpips_acc.mean().item():.3f} +/- {im2im_lpips_acc.std().item():.3f}")

    print(f"QUTCC MSE: {quantile_mse_acc.mean().item():.7f} +/- {quantile_mse_acc.std().item():.7f}")
    print(f"QUTCC SSIM: {quantile_ssim_acc.mean().item():.3f} +/- {quantile_ssim_acc.std().item():.3f}")
    print(f"QUTCC PSNR: {quantile_psnr_acc.mean().item():.3f} +/- {quantile_psnr_acc.std().item():.3f}")
    print(f"QUTCC LPIPS: {quantile_lpips_acc.mean().item():.3f} +/- {quantile_lpips_acc.std().item():.3f}")

    print(f"im2im_deep MSE: {unet_im2im_mse_acc.mean().item():.7f} +/- {unet_im2im_mse_acc.std().item():.7f}")
    print(f"im2im_deep SSIM: {unet_im2im_ssim_acc.mean().item():.3f} +/- {unet_im2im_ssim_acc.std().item():.3f}")
    print(f"im2im_deep PSNR: {unet_im2im_psnr_acc.mean().item():.3f} +/- {unet_im2im_psnr_acc.std().item():.3f}")
    print(f"im2im_deep LPIPS: {unet_im2im_lpips_acc.mean().item():.3f} +/- {unet_im2im_lpips_acc.std().item():.3f}")

    results = {
        "im2im": {
            "MSE":   {"mean": im2im_mse_acc.mean().item(),   "std": im2im_mse_acc.std().item()},
            "SSIM":  {"mean": im2im_ssim_acc.mean().item(),  "std": im2im_ssim_acc.std().item()},
            "PSNR":  {"mean": im2im_psnr_acc.mean().item(),  "std": im2im_psnr_acc.std().item()},
            "LPIPS": {"mean": im2im_lpips_acc.mean().item(), "std": im2im_lpips_acc.std().item()},
        },
        "QUTCC": {
            "MSE":   {"mean": quantile_mse_acc.mean().item(),   "std": quantile_mse_acc.std().item()},
            "SSIM":  {"mean": quantile_ssim_acc.mean().item(),  "std": quantile_ssim_acc.std().item()},
            "PSNR":  {"mean": quantile_psnr_acc.mean().item(),  "std": quantile_psnr_acc.std().item()},
            "LPIPS": {"mean": quantile_lpips_acc.mean().item(), "std": quantile_lpips_acc.std().item()},
        },
        "im2im_deep": {
            "MSE":   {"mean": unet_im2im_mse_acc.mean().item(),   "std": unet_im2im_mse_acc.std().item()},
            "SSIM":  {"mean": unet_im2im_ssim_acc.mean().item(),  "std": unet_im2im_ssim_acc.std().item()},
            "PSNR":  {"mean": unet_im2im_psnr_acc.mean().item(),  "std": unet_im2im_psnr_acc.std().item()},
            "LPIPS": {"mean": unet_im2im_lpips_acc.mean().item(), "std": unet_im2im_lpips_acc.std().item()},
        },
    }

    if return_raw:
        raw = {
            "im2im":        {"MSE": im2im_mse_acc, "SSIM": im2im_ssim_acc, "PSNR": im2im_psnr_acc, "LPIPS": im2im_lpips_acc},
            "QUTCC":        {"MSE": quantile_mse_acc, "SSIM": quantile_ssim_acc, "PSNR": quantile_psnr_acc, "LPIPS": quantile_lpips_acc},
            "im2im_deep":   {"MSE": unet_im2im_mse_acc, "SSIM": unet_im2im_ssim_acc, "PSNR": unet_im2im_psnr_acc, "LPIPS": unet_im2im_lpips_acc},
        }
        return results, raw

    return results

def check_crossings(experiment_type: str, info: Dict, device: torch.device, 
                    calib_subset: int = 2000) -> Tuple[int, int]:
    epoch = info["quantile_epoch"]
    quantile_model, run_folder = load_checkpoint_for_inference(
        net="unet_quantile", in_channels=info["in_channels"],
        experiment_type=experiment_type, run_folder_root=info["quantile_root"],
        epoch=epoch, device=device, experiments_folder=info["experiments_folder"])
    
    analysis_dir = run_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    crossings_dataloader = get_calib_dataloader(experiment_type, info, net="unet_quantile",
                                                 batch_size_multiplier=0.1, calib_subset=calib_subset)
    quantiles = torch.arange(0.1, 1.0, 0.1).to(device)
    crossings_file = analysis_dir / "quantile_crossings.json"
    crossings_data = manage_checkpoint(crossings_file, check_quantile_crossings,
                                        dataloader=crossings_dataloader, model=quantile_model, 
                                        quantiles=quantiles, device=device)
    violations, total = crossings_data["violations"], crossings_data["total"]
    del crossings_dataloader
    return violations, total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary, all_stratified, all_df = run(
        # can replace with experiment types like ["gaussian", "mri", "qpi"]
        tasks=None, 
        calib_subset=2000,
        test_subset=2000,
        intervals_subset=500_000,
        device=device,
        use_submitit=True,
        partition="gpu",
    )