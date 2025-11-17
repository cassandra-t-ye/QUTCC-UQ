import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from tqdm import tqdm

from models.checkpointing import load_checkpoint_for_inference
from evaluation.calibration_utils import get_calib_dataloader
from evaluation.calibration import compute_optimal_lambdas
from utils import _prepare_for_json, json_converter

ROOT = "/share/monakhova/Cassandra_data/UQNet_proj"


def inject_root(d: Dict[str, Any]) -> None:
    """
    Replace placeholder 'ROOT/' in paths with the actual ROOT.
    """
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("ROOT/"):
            d[k] = v.replace("ROOT", ROOT, 1)
        elif isinstance(v, dict):
            inject_root(v)


# -------------------------------------------------------------------------
# Core: compute crossings for an im2im model
# -------------------------------------------------------------------------


def compute_im2im_crossings(
    dataloader,
    model: nn.Module,
    im2im_lam: float,
    device: torch.device,
) -> Dict[str, int]:
    """
    Compute coverage violations using the im2im model.

    We assume model output channels are:
        0 = lower-like prediction
        1 = mean/central prediction
        2 = upper-like prediction

    Intervals are constructed consistently with compute_optimal_lambdas:
        lower = mean - lam * (mean - lower_channel)
        upper = mean + lam * (upper_channel - mean)
    """
    model.eval()
    violations = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Im2Im crossings", leave=False):
            noisy, target = batch
            noisy = noisy.to(device)
            target = target.to(device)

            preds = model(noisy)  # [B, 3, H, W]
            preds = preds.cpu()
            target = target.cpu()

            # Split channels
            ch0 = preds[:, 0, ...]  # lower-like
            ch1 = preds[:, 1, ...]  # mean
            ch2 = preds[:, 2, ...]  # upper-like

            lower = ch1 - im2im_lam * (ch1 - ch0)
            upper = ch1 + im2im_lam * (ch2 - ch1)

            # Ground truth handling
            if target.ndim == 4:
                # [B, C, H, W]; if single-channel, squeeze it; else mean over channels
                if target.shape[1] == 1:
                    gt = target[:, 0, ...]
                else:
                    gt = target.mean(dim=1)
            else:
                gt = target

            batch_violations = (lower > upper).sum().item()
            batch_total = gt.numel()

            violations += batch_violations
            total += batch_total

    return {"violations": int(violations), "total": int(total)}


def check_im2im_crossings(
    experiment_type: str,
    info: Dict[str, Any],
    device: torch.device,
    calib_subset: int = 2000,
) -> Tuple[int, int, float]:
    """
    For a given task/experiment:
        - Load im2im model at info['im2im_epoch'].
        - Calibrate im2im_lam via compute_optimal_lambdas (NO caching).
        - Run on calibration dataloader and compute violation stats (NO caching).
    """
    if "im2im_root" not in info or "im2im_epoch" not in info:
        raise KeyError("im2im_root or im2im_epoch missing in BEST_RUNS entry.")

    epoch = info["im2im_epoch"]

    im2im_model, run_folder = load_checkpoint_for_inference(
        net="im2im",
        in_channels=info["in_channels"],
        experiment_type=experiment_type,
        run_folder_root=info["im2im_root"],
        epoch=epoch,
        device=device,
        experiments_folder=info["experiments_folder"],
    )

    # Calibration dataloader; adjust batch_size_multiplier if desired
    calib_dataloader = get_calib_dataloader(
        experiment_type,
        info,
        net="im2im",
        batch_size_multiplier=1,
        calib_subset=calib_subset,
    )

    # Just to have a consistent place to dump final results if you ever want
    run_folder = Path(run_folder)
    analysis_dir = run_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Calibrate im2im_lam (NO checkpointing; always recompute)
    # ------------------------------------------------------------------
    alpha, min_l, max_l, num_l = 0.1, 1, 1.5, 300

    im2im_lam = compute_optimal_lambdas(
        dataloader=calib_dataloader,
        model=im2im_model,
        alpha=alpha,
        min_lam=min_l,
        max_lam=max_l,
        num_lam=num_l,
        device=device,
    )

    print(f"Using calibrated im2im lambda = {im2im_lam}")

    # ------------------------------------------------------------------
    # 2) Compute crossings for that lambda (NO checkpointing; always recompute)
    # ------------------------------------------------------------------
    crossings_data = compute_im2im_crossings(
        dataloader=calib_dataloader,
        model=im2im_model,
        im2im_lam=float(im2im_lam),
        device=device,
    )

    violations = int(crossings_data["violations"])
    total = int(crossings_data["total"])
    del calib_dataloader

    return violations, total, float(im2im_lam)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BEST_RUNS
    with open(Path(__file__).parent / "best_runs.json", "r") as f:
        BEST_RUNS = json.load(f)

    inject_root(BEST_RUNS)

    im2im_crossings_summary: Dict[str, Dict[str, Any]] = {}

    for task_name, info in BEST_RUNS.items():
        experiment_type = info["experiment_type"]
        print(
            f"\n=== Checking im2im coverage crossings for task '{task_name}' "
            f"(experiment_type='{experiment_type}') ==="
        )

        # Skip tasks without im2im config
        if "im2im_root" not in info or "im2im_epoch" not in info:
            print(
                f"Skipping task '{task_name}' because 'im2im_root' or "
                f"'im2im_epoch' is missing in best_runs.json."
            )
            continue

        try:
            violations, total, im2im_lam = check_im2im_crossings(
                experiment_type=experiment_type,
                info=info,
                device=device,
                calib_subset=2000,
            )
        except Exception as e:
            print(f"Error while processing task '{task_name}': {e}")
            continue

        rate = float(violations) / float(total) if total > 0 else float("nan")
        print(
            f"Im2Im interval (λ = {im2im_lam:.4f}) for '{task_name}': "
            f"{violations} / {total} violations (rate = {rate:.6f})"
        )

        im2im_crossings_summary[task_name] = {
            "violations": violations,
            "total": total,
            "rate": rate,
            "im2im_lam": im2im_lam,
        }

    # Save summary across all tasks (this is *final* output, not a cache)
    out_path = Path("im2im_crossings_all_models.json")
    with open(out_path, "w") as f:
        json.dump(
            _prepare_for_json(im2im_crossings_summary),
            f,
            indent=4,
            default=json_converter,
        )

    print(f"\nSaved im2im crossings summary to: {out_path}")
