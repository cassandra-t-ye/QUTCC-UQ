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
ROOT = "/share/monakhova/Cassandra_data/UQNet_proj"

with open(Path(__file__).parent / "best_runs.json", "r") as f:
    BEST_RUNS = json.load(f)

# Replace placeholder "ROOT/" with the actual ROOT
def inject_root(d):
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("ROOT/"):
            d[k] = v.replace("ROOT", ROOT, 1)
        elif isinstance(v, dict):
            inject_root(v)

inject_root(BEST_RUNS)

print(BEST_RUNS)

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

def calibrate_quantile(experiment_type: str, info: Dict, device: torch.device, 
                       calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float, float]:
    epoch = epoch if epoch is not None else info["quantile_epoch"]
    quantile_model, run_folder = load_checkpoint_for_inference(net="unet_quantile", in_channels=info["in_channels"],
                                        experiment_type=experiment_type, run_folder_root=info["quantile_root"],
                                        epoch=epoch, device=device, 
                                        experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net="quantile", calib_subset=calib_subset)

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

from pathlib import Path
import json

def check_crossings(
    experiment_type: str,
    info: Dict,
    device: torch.device,
    calib_subset: int = 2000,
) -> Tuple[int, int, float, float]:
    epoch = info["quantile_epoch"]

    # Where calibrated quantiles are stored/loaded
    qs_path = Path(ROOT) / "analysis_checkpoints" / experiment_type / "quantile_qs.json"

    if qs_path.exists():
        print(f"Loading calibrated quantiles from {qs_path}")
        with open(qs_path, "r") as f:
            qs = json.load(f)
        lower_q = float(qs["lower_q"])
        upper_q = float(qs["upper_q"])

        quantile_model, _ = load_checkpoint_for_inference(
            net="unet_quantile",
            in_channels=info["in_channels"],
            experiment_type=experiment_type,
            run_folder_root=info["quantile_root"],
            epoch=epoch,
            device=device,
            experiments_folder=info["experiments_folder"],
        )

    else:
        print(f"quantile_qs.json not found at {qs_path}. Calibrating quantiles on the fly.")
        quantile_model, lower_q, upper_q = calibrate_quantile(
            experiment_type=experiment_type,
            info=info,
            device=device,
            calib_subset=calib_subset,
            epoch=epoch,
        )

        qs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qs_path, "w") as f:
            json.dump(
                {
                    "lower_q": lower_q,
                    "upper_q": upper_q,
                },
                f,
                indent=4,
            )
        print(f"Saved calibrated quantiles to {qs_path}")

    # This is where other analysis artifacts for this experiment_type live
    analysis_dir = qs_path.parent
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Dataloader used for crossings
    crossings_dataloader = get_calib_dataloader(
        experiment_type,
        info,
        net="quantile",
        batch_size_multiplier=1,   # your updated setting
        calib_subset=calib_subset,
    )

    quantiles = torch.tensor([lower_q, upper_q], device=device, dtype=torch.float32)
    print("Using quantiles:", quantiles)

    # -----------------------
    # 1) Global stats (shared file, unchanged schema)
    # -----------------------
    crossings_file = analysis_dir / "quantile_crossings.json"

    crossings_data = manage_checkpoint(
        crossings_file,
        check_quantile_crossings,
        dataloader=crossings_dataloader,
        model=quantile_model,
        quantiles=quantiles,
        device=device,
    )

    violations = int(crossings_data["violations"])
    total = int(crossings_data["total"])

    # -----------------------
    # 2) Per-batch stats (in-memory run, not using shared checkpoint)
    # -----------------------
    batch_stats = check_quantile_crossings(
        dataloader=crossings_dataloader,
        model=quantile_model,
        quantiles=quantiles,
        device=device,
        return_per_batch=True,
    )

    per_batch = batch_stats.get("per_batch", [])

    # Dict: {batch_idx: {"violations": int, "total": int}}
    batch_crossings: Dict[int, Dict[str, int]] = {
        int(entry["batch_idx"]): {
            "violations": int(entry["violations"]),
            "total": int(entry["total"]),
        }
        for entry in per_batch
    }

    # Sort by batch index for readability
    batch_crossings = dict(sorted(batch_crossings.items(), key=lambda kv: kv[0]))

    # Aggregate from per-batch dict for sanity-check
    per_batch_violations = sum(v["violations"] for v in batch_crossings.values())
    per_batch_total = sum(v["total"] for v in batch_crossings.values())
    per_batch_rate = (
        per_batch_violations / per_batch_total
        if per_batch_total > 0
        else float("nan")
    )
    print(
        f"Per-batch aggregated crossings for '{experiment_type}': "
        f"{per_batch_violations} / {per_batch_total} "
        f"(rate = {per_batch_rate:.6f})"
    )

    # -----------------------
    # 3) Save per-batch stats to a separate file per experiment_type
    # -----------------------
    batch_out_dir = Path("batch_crossings")
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    batch_out_file = batch_out_dir / f"{experiment_type}_batch_crossings.json"

    with open(batch_out_file, "w") as f:
        json.dump(batch_crossings, f, indent=4)

    print(f"Saved batch crossings for '{experiment_type}' to: {batch_out_file}")

    del crossings_dataloader

    return int(violations), int(total), float(lower_q), float(upper_q)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(Path(__file__).parent / "best_runs.json", "r") as f:
        BEST_RUNS = json.load(f)
    inject_root(BEST_RUNS)

    crossings_summary = {}  # collect only successful tasks

    for task_name, info in BEST_RUNS.items():
        experiment_type = info["experiment_type"]

        print(f"\n=== Checking calibrated crossings for task '{task_name}' "
              f"(experiment_type='{experiment_type}') ===")

        result = check_crossings(
            experiment_type=experiment_type,
            info=info,
            device=device,
            calib_subset=2000,
        )

        # If quantile_qs.json was missing, result is None → skip
        if result is None:
            print(f"Skipping task '{task_name}' because quantile_qs.json was not found.")
            continue

        violations, total, lower_q, upper_q = result
        rate = float(violations) / float(total) if total > 0 else float("nan")

        print(
            f"Calibrated interval [{lower_q:.4f}, {upper_q:.4f}] for '{task_name}': "
            f"{violations} / {total} crossings (rate = {rate:.6f})"
        )

        crossings_summary[task_name] = {
            "violations": violations,
            "total": total,
            "rate": rate,
            "lower_q": lower_q,
            "upper_q": upper_q,
        }

    # Optional: save out (only for tasks that had quantile_qs.json)
    out_path = Path("quantile_crossings_calibrated_interval_all_models.json")
    with open(out_path, "w") as f:
        json.dump(_prepare_for_json(crossings_summary), f, indent=4, default=json_converter)
    print(f"Saved calibrated-interval crossings summary to: {out_path}")
