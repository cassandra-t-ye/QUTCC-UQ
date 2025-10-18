import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis import BEST_RUNS
from evaluation.calibration_utils import (CalibrationManager, AnalysisManager)
from models.checkpointing import get_checkpoint_epochs
from utils import _prepare_for_json


def run_epoch_sweep(task: str, model_type: str, calib_subset: int, test_subset: int, 
                    intervals_subset: int, partition: str, ignore_checkpoints=False) -> None:
    print(f"--- Starting sweep for {task} - {model_type} ---")
    if task not in BEST_RUNS:
        raise ValueError(f"Task '{task}' not found. Available: {list(BEST_RUNS.keys())}")

    info = BEST_RUNS[task]
    experiment_type = info['experiment_type']

    run_folder_root = info.get(f"{model_type}_root")
    if not run_folder_root:
        raise ValueError(f"Root directory for '{model_type}' not found in '{experiment_type}'")
    checkpoints_dir = Path(info["experiments_folder"]) / experiment_type / run_folder_root / "checkpoints"
    epochs = get_checkpoint_epochs(checkpoints_dir)
    if not epochs:
        print(f"No valid checkpoints found in {checkpoints_dir}. Skipping sweep.")
        return

    print(f"--- Calibrating {len(epochs)} epochs ---")
    # can put ignore_checkpoints=True to recalibrate
    calib_manager = CalibrationManager(experiment_type=experiment_type, info=info, 
                                       calib_subset=calib_subset, partition=partition,
                                       ignore_checkpoints=ignore_checkpoints)
    calib_requests = [(model_type, epoch) for epoch in epochs]
    calib_results = calib_manager.calibrate_multiple(calib_requests)
    print(f"--- Calibration complete for {len(calib_results)} epochs ---")

    if not calib_results:
        print("No results returned from calibration jobs. Cannot proceed with analysis.")
        return
    
    print(f"--- Analyzing {len(calib_results)} calibrated epochs ---")
    analysis_manager = AnalysisManager(experiment_type, info, test_subset, intervals_subset, partition, 
                                       ignore_checkpoints=ignore_checkpoints)
    analysis_results = analysis_manager.analyze_multiple(calib_results)
    if not analysis_results:
        print("No results returned from analysis jobs.")
        return
    
    processed_results = []
    for result in analysis_results:
        processed_results.append({
            "epoch": result["epoch"],
            "mean_interval_length": result["mean_interval_length"],
            "risk": result["risk"]["mean_risk"],
        })
    
    df = pd.DataFrame(processed_results)

    valid_epochs_df = df[df["risk"] < 0.10]
    best_epoch, min_interval_length = None, None
    if not valid_epochs_df.empty:
        best_epoch_row = valid_epochs_df.loc[valid_epochs_df["mean_interval_length"].idxmin()]
        best_epoch = int(best_epoch_row["epoch"])
        min_interval_length = best_epoch_row["mean_interval_length"]

    print(f"\n--- Results for {task} - {model_type} ---")
    print(df.to_string())
    if best_epoch is not None:
        print(f"\nBest epoch: {best_epoch} (Mean Interval Length: {min_interval_length:.4f})")
    else:
        print("No valid epochs found with risk < 0.10")

    run_folder = Path(info["experiments_folder"]) / experiment_type / run_folder_root
    analysis_dir: Path = run_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sweep_summary_path = analysis_dir / f"sweep_summary_{model_type}.json"
    summary = {
        "best_epoch": best_epoch,
        "min_mean_interval_length": min_interval_length,
        "all_epochs_results": df.to_dict("records")
    }
    with open(sweep_summary_path, 'w') as f:
        json.dump(_prepare_for_json(summary), f, indent=4)

    print(f"Sweep summary saved to {sweep_summary_path}")
    return best_epoch, min_interval_length

def _create_and_save_plot(epoch_list: List[int], mean_diff_list: List[float], 
                          risk_diff_list: List[float], experiment_type: str, save_dir: Path):
    plt.figure(figsize=(8,6))
    plt.scatter(mean_diff_list, risk_diff_list)

    for epoch, m, r in zip(epoch_list, mean_diff_list, risk_diff_list):
        plt.annotate(
            str(epoch),
            xy=(m, r),
            xytext=(5,5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.xlabel("ΔMean Interval Length (QUTCC - im2im_deep)")
    plt.ylabel("ΔRisk (QUTCC - im2im_deep)")
    plt.title(f"ΔInterval length vs. ΔRisk across Epochs: {experiment_type}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = save_dir / f"mean_vs_risk_qutcc_{experiment_type}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved risk vs. mean interval length ({experiment_type}) plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    #Choose which experiments you want to run: "mri", "qpi", "CT", "gaussian", "poisson", "real_noise_mice"
    experiments_to_run = ['mri']

    #Choose which models you want to calibrate: 'unet_quantile', 'unet_im2im', 'im2im'
    model_types_to_run = ['unet_quantile']

    all_run_results = {}
    for experiment in experiments_to_run:
        all_run_results[experiment] = {}
        for model_type in model_types_to_run:
            all_run_results[experiment][model_type] = run_epoch_sweep(
                task=experiment, 
                model_type=model_type,
                calib_subset=500,
                test_subset=1000,
                intervals_subset=200_000,
                partition="gpu",           #Choose which gpu partition you want to calibrate on
                ignore_checkpoints=True
            )

    print(all_run_results)

    print("\n\n" + "="*25 + " Run Summary " + "="*25)
    for task, models in all_run_results.items():
        print(f"\n--- Task: {task} ---")
        for model, (epoch, length) in models.items():
            # Use left-justification for clean alignment
            model_name_padded = f"{model:<12}" 

            if epoch is not None and length is not None:
                print(f"  - {model_name_padded}: Best Epoch: {epoch}, Min Interval Length: {length:.4f}")
            else:
                print(f"  - {model_name_padded}: No valid epoch found.")
    print("\n" + "="*63 + "\n")