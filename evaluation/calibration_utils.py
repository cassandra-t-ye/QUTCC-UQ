import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import submitit
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets.fastmri.utils as fastmri_utils
from datasets.fastmri import FastMRIDataset
from datasets.bsccm import get_bsccm_calib, get_bsccm_test
from datasets.fmd.data_loader import (load_noisy_single_loader,
                                                load_real_noise_loader)
from datasets.LidcIdri import load_ct_dataloader

from evaluation import get_test_intervals_missed
from evaluation.calibration import (compute_optimal_lambdas,
                                    compute_optimal_lambdas_quantile)
from models.checkpointing import load_checkpoint_for_inference
from utils import _prepare_for_json


def get_calib_dataloader(experiment_type: str, info: Dict[str, Any], net: Literal['im2im', 'unet_quantile', 'unet_im2im'], 
                         batch_size_multiplier: float = 1.0, calib_subset: int = 200) -> DataLoader:
    batch_size = int(info["batch_size"] * batch_size_multiplier)

    if experiment_type == "mri":
        mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
        calib_dataset = FastMRIDataset(info["calib_data_path"], normalize_input='standard', 
                                       normalize_output='min-max', mask_info=mask_info, num_volumes=100)
        fastmri_utils.normalize_dataset(calib_dataset)
    elif experiment_type in ["gaussian", "poisson"]:
        noise_type, min_noise, max_noise, num_noise = experiment_type, info["min_noise"], info["max_noise"], info["num_noise"]
        loader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise,
                                                 image_folder=info["calib_data_path"],
                                                 batch_size=batch_size,
                                                 types=None, transform=None)
        calib_dataset = loader.dataset
    elif experiment_type == "qpi":
        run_folder = Path(info["experiments_folder"]) / experiment_type / info[f'{net}_root']
        calib_indices_path = run_folder / "calib_indices.json"
        with open(calib_indices_path, "r") as f:
            calib_indices = json.load(f)
        return get_bsccm_calib(calib_indices[:calib_subset], info["data_root"], 
                               batch_size=batch_size, transform=None, 
                               normalization='min-max', num_workers=1, pin_memory=True)
    elif experiment_type == "real_noise":        
        loader = load_real_noise_loader(noisy_path=info["calib_data_path"] + "/noisy",
                                        gt_path=info["calib_data_path"] + "/gt",
                                        batch_size=batch_size, shuffle=False)
        calib_dataset = loader.dataset

    elif experiment_type == "CT":
        loader = load_ct_dataloader(root_path=info["test_data_path"],
                            dataset_type = "calibrate",
                            batch_size = 16,
                            steps = 800)
        calib_dataset = loader.dataset

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    if len(calib_dataset) > calib_subset:
        indices = torch.randperm(len(calib_dataset))[:calib_subset]
        calib_dataset = Subset(calib_dataset, indices)

    print(f"Calibration dataset size: {len(calib_dataset)}")
    return DataLoader(calib_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

def get_test_dataloader(experiment_type: str, info: Dict[str, Any], net: Literal['im2im', 'unet_quantile', 'unet_im2im'], 
                        batch_size_multiplier: float = 0.5, test_subset: int = 200) -> DataLoader:
    batch_size = int(info["batch_size"] * batch_size_multiplier)
    if experiment_type == "mri":
        mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
        test_dataset = FastMRIDataset(info["test_data_path"], normalize_input='standard', normalize_output='min-max', 
                                      mask_info=mask_info, num_volumes=20)
        fastmri_utils.normalize_dataset(test_dataset)
    elif experiment_type in ["gaussian", "poisson"]:
        noise_type, min_noise, max_noise, num_noise = experiment_type, info["min_noise"], info["max_noise"], info["num_noise"]
        loader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, 
                                          image_folder=info["test_data_path"], 
                                          batch_size=batch_size,
                                          types=None, transform=None)
        test_dataset = loader.dataset
    elif experiment_type == "qpi":
        run_folder = Path(info["experiments_folder"]) / experiment_type / info[f'{net}_root']
        test_indices_path = run_folder / "test_indices.json"
        with open(test_indices_path, "r") as f:
            test_indices = json.load(f)
        return get_bsccm_test(test_indices[:test_subset], info["data_root"], 
                              batch_size=batch_size, transform=None, 
                              normalization='min-max', num_workers=1, pin_memory=True)
    elif experiment_type == "real_noise":
        loader = load_real_noise_loader(noisy_path=info["test_data_path"] + "/noisy",
                                        gt_path=info["test_data_path"] + "/gt",
                                        batch_size=batch_size, shuffle=False)
        test_dataset = loader.dataset     
    elif experiment_type == "CT":
        loader = load_ct_dataloader(info["test_data_path"],
                            dataset_type = "validate",
                            batch_size = 16,
                            steps = 800)
        test_dataset = loader.dataset
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    if len(test_dataset) > test_subset:
        indices = torch.randperm(len(test_dataset))[:test_subset]
        test_dataset = Subset(test_dataset, indices)
    print(f"Test dataset size: {len(test_dataset)}")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

def get_checkpoint_info(experiment_type: str, info: Dict, epoch: int, 
                       net_type: Literal['im2im', 'unet_im2im', "unet_quantile"]) -> Tuple[Path, str]:
    run_folder = Path(info["experiments_folder"]) / experiment_type / info[f"{net_type}_root"]
    analysis_dir = run_folder / "analysis"
    filename = f"quantile_qs_epoch_{epoch}.json" if net_type == "unet_quantile" else f"{net_type}_epoch{epoch}_lambda.pt"
    return analysis_dir, filename

def load_existing_checkpoint(experiment_type: str, info: Dict, epoch: int, 
                           net_type: Literal['im2im', 'unet_im2im', "unet_quantile"]) -> Optional[Dict[str, Any]]:
    try:
        analysis_dir, filename = get_checkpoint_info(experiment_type, info, epoch, net_type)
        checkpoint_file = analysis_dir / filename
        
        if not checkpoint_file.exists():
            return None
            
        # print(f"Found checkpoint: {checkpoint_file}")
        if net_type == "unet_quantile":
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            result = {"lower_q": data['lower_q'], "upper_q": data['upper_q']}
        else:
            result = {"lambda": torch.load(checkpoint_file)}
        
        result.update({
            "epoch": epoch, "net_type": net_type, "analysis_dir": str(analysis_dir),
            "checkpoint_file": str(checkpoint_file), "loaded_from_checkpoint": True
        })
        return result
        
    except Exception as e:
        print(f"Error checking checkpoint for {experiment_type} - {net_type} - {epoch}: {e}")
        return None

class CalibrateModelTask(submitit.helpers.Checkpointable):
    def __init__(self, experiment_type: str, info: Dict, epoch: int,
                 net_type: Literal['im2im', 'unet_im2im', "unet_quantile"], calib_subset: int = 2000):
        os.environ['TMPDIR'] = "/tmp"
        self.experiment_type = experiment_type
        self.info = info
        self.epoch = epoch
        self.net_type = net_type
        self.calib_subset = calib_subset

    def __call__(self) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Calibrating {self.experiment_type} - {self.net_type} - {self.epoch} - {device}")

        model, run_folder = load_checkpoint_for_inference(
            net=self.net_type,
            in_channels=self.info['in_channels'], experiment_type=self.experiment_type,
            run_folder_root=self.info[f"{self.net_type}_root"], epoch=self.epoch,
            device=device, experiments_folder=self.info['experiments_folder']
        )

        analysis_dir = Path(run_folder) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        calib_dataloader = get_calib_dataloader(
            self.experiment_type, self.info,
            net=self.net_type,
            batch_size_multiplier=1 if self.net_type != "unet_quantile" else 0.9,
            calib_subset=self.calib_subset
        )
        
        if self.net_type == "unet_quantile":
            checkpoint_file = analysis_dir / f"quantile_qs_epoch_{self.epoch}.json"
            data = compute_optimal_lambdas_quantile(
                dataloader=calib_dataloader, model=model, alpha=0.1,
                lower_min=1e-20, lower_max=0.5, upper_min=0.5, upper_max=1 - 1e-20,
                max_iterations=30, device=device
            )
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            result = {"lower_q": data['lower_q'], "upper_q": data['upper_q']}
        else:
            checkpoint_file = analysis_dir / f"{self.net_type}_epoch{self.epoch}_lambda.pt"
            lambda_val = compute_optimal_lambdas(
                dataloader=calib_dataloader, model=model, alpha=0.1,
                min_lam=0, max_lam=10, num_lam=1000, device=device
            )
            torch.save(lambda_val, checkpoint_file)
            
            result = {"lambda": lambda_val}

        result.update({
            "epoch": self.epoch, 
            "net_type": self.net_type, 
            "analysis_dir": str(analysis_dir),
            "checkpoint_file": str(checkpoint_file), 
            "loaded_from_checkpoint": False
        })
        
        del model, calib_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        return result

class CalibrationManager:
    def __init__(self, experiment_type: str, info: Dict, calib_subset: int = 2000,
                 partition: str = "gpu", timeout_min: int = 120, logs_dir: str = "slurm_archive",
                 ignore_checkpoints: bool = False):
        self.experiment_type = experiment_type
        self.info = info
        self.calib_subset = calib_subset
        self.ignore_checkpoints = ignore_checkpoints
        self.executor = create_local_executor(partition, logs_dir, timeout_min, job_name=f"{experiment_type}-calibration")
        
    def calibrate(self, net_type: Literal['im2im', 'unet_im2im', "unet_quantile"], 
                  epoch: Optional[int] = None) -> Dict[str, Any]:
        if epoch is None:
            epoch = self.info[f"{net_type}_epoch"]
        return self.calibrate_multiple([(net_type, epoch)])

    def calibrate_multiple(self, calibration_requests: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
        """calibration_requests: List of (net_type, epoch) tuples"""
        results = []
        jobs = []
        
        with self.executor.batch():
            for net_type, epoch in calibration_requests:
                existing_result = load_existing_checkpoint(self.experiment_type, self.info, epoch, net_type)
                if existing_result and not self.ignore_checkpoints:
                    print(f"Using existing calibration for {self.experiment_type} - {net_type} - {epoch}")
                    results.append(existing_result)
                else:
                    task = CalibrateModelTask(self.experiment_type, self.info, epoch, net_type, self.calib_subset)
                    job = self.executor.submit(task)
                    jobs.append(job)
                    print(f"Submitted calibration for {self.experiment_type} - {net_type} - {epoch}")

        if jobs:
            print(f"Waiting for {len(jobs)} calibration jobs to complete...")
            for job in tqdm(jobs, desc="Calibrating"):
                try:
                    results.append(job.result())
                except Exception as e:
                    print(f"Error in calibration job: {e}")
        
        return results
    
def get_intervals_and_risk(net_type: Literal["im2im", "unet_im2im", "unet_quantile"],
                           experiment_type: str, info: Dict, epoch: int,
                           lambda_or_lower_q: float, upper_q: Optional[float] = None,
                           test_subset: int = 2000, intervals_subset: int = 500_000,
                           device: torch.device = torch.device('cpu'),
                           df_file: Path = None, risk_file: Path = None) -> Tuple[pd.DataFrame, Dict]:
    model, run_folder = load_checkpoint_for_inference(net=net_type, in_channels=info["in_channels"],
                                        experiment_type=experiment_type,
                                        run_folder_root=info[f"{net_type}_root"],
                                        epoch=epoch, device=device,
                                        experiments_folder=info['experiments_folder'])
    
    if df_file and risk_file:
        df_intervals_missed_file = df_file
        risk_file = risk_file
    else:
        analysis_dir = Path(run_folder) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        df_intervals_missed_file = analysis_dir / f"intervals_missed_epoch{epoch}.feather"
        risk_file = analysis_dir / f"risk_epoch{epoch}.json"

    test_dataloader = get_test_dataloader(experiment_type, info, net=net_type,
                                          batch_size_multiplier=0.7, test_subset=test_subset)
    df, risk = get_test_intervals_missed(net_type, test_dataloader, model, 
                                         lambda_or_lower_q, upper_q, device=device)
    df = df.sample(n=min(intervals_subset, len(df)), random_state=42)
    
    df.to_feather(df_intervals_missed_file, compression="zstd")
    print("Saved intervals_missed to: ", df_intervals_missed_file)
    with open(risk_file, "w") as f: json.dump(_prepare_for_json(risk), f, indent=4)
    print("Saved risk to: ", risk_file)

    del test_dataloader, model
    return df, risk    

class IntervalRiskTask(submitit.helpers.Checkpointable):
    def __init__(self, net_type: str, experiment_type: str, info: Dict,
                 epoch: int, lambda_or_lower_q: float, upper_q: Optional[float],
                 test_subset: int, intervals_subset: int, df_file: Path, risk_file: Path,
                 pop_df: bool = False):
        os.environ['TMPDIR'] = "/tmp"
        self.net_type = net_type
        self.experiment_type = experiment_type
        self.info = info
        self.epoch = epoch
        self.lambda_or_lower_q = lambda_or_lower_q
        self.upper_q = upper_q
        self.test_subset = test_subset
        self.intervals_subset = intervals_subset
        self.df_file = df_file
        self.risk_file = risk_file
        self.pop_df = pop_df  # If True, will not return df in results
    
    def __call__(self) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df, risk = get_intervals_and_risk(
            net_type=self.net_type,
            experiment_type=self.experiment_type,
            info=self.info,
            epoch=self.epoch,
            lambda_or_lower_q=self.lambda_or_lower_q,
            upper_q=self.upper_q,
            test_subset=self.test_subset,
            intervals_subset=self.intervals_subset,
            device=device,
            df_file=self.df_file,
            risk_file=self.risk_file
        )
        return {
            'epoch': self.epoch,
            'net_type': self.net_type,
            'df': df if not self.pop_df else None,
            'risk': risk,
            'mean_interval_length': df["interval"].mean()
        }
    
class AnalysisManager:
    def __init__(self, experiment_type: str, info: Dict, test_subset: int = 2000,
                 intervals_subset: int = 500_000, partition: str = "gpu", 
                 timeout_min: int = 60, logs_dir: str = "slurm_archive",
                 ignore_checkpoints: bool = False,
                 pop_df = False):
        self.experiment_type = experiment_type
        self.info = info
        self.test_subset = test_subset
        self.intervals_subset = intervals_subset
        self.ignore_checkpoints = ignore_checkpoints
        self.pop_df = pop_df  # If True, will not return df in results
        self.executor = create_local_executor(partition, logs_dir, timeout_min, job_name=f'{experiment_type}-analysis')
    
    def analyze_multiple(self, analysis_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        jobs: List[submitit.Job] = []

        with self.executor.batch():
            for request in analysis_requests:
                net_type = request['net_type']
                epoch = request['epoch']

                root = self.info[f"{net_type}_root"]
                run_folder = Path(self.info["experiments_folder"]) / self.experiment_type / root
                analysis_dir = run_folder / "analysis"
                df_file = analysis_dir / f"intervals_missed_epoch{epoch}.feather"
                risk_file = analysis_dir / f"risk_epoch{epoch}.json"

                if df_file.exists() and risk_file.exists() and not self.ignore_checkpoints:
                    print(f"Found existing analysis for {self.experiment_type} - {net_type} - {epoch}.")
                    df = pd.read_feather(df_file)
                    with open(risk_file, 'r') as f:
                        risk = json.load(f)
                    results.append({
                        'epoch': epoch,
                        'net_type': net_type,
                        'mean_interval_length': df["interval"].mean(),
                        'df': None if self.pop_df else df, 
                        'risk': risk,})
                else:
                    lambda_or_lower_q = request.get('lambda') or request.get('lower_q')
                    upper_q = request.get('upper_q')
                    task = IntervalRiskTask(net_type, self.experiment_type, self.info, epoch, 
                                            lambda_or_lower_q, upper_q,
                                            self.test_subset, self.intervals_subset,
                                            df_file=df_file, risk_file=risk_file,
                                            pop_df=self.pop_df)
                    job = self.executor.submit(task)
                    jobs.append(job)
                    print(f"Submitted analysis: {self.experiment_type} - {net_type} - {epoch}")
        
        if jobs:
            print(f"Waiting for {len(jobs)} analysis jobs to complete...")
            for job in tqdm(jobs, desc="Analyzing"):
                try:
                    result = job.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in analysis job: {e}")
        
        return results
    
def create_local_executor(partition: str, logs_dir: str, timeout_min: int = 120,
                          gpus_per_node: int = 1, 
                          mem_gb: int = 64, job_name="uqnet_analysis") -> submitit.AutoExecutor:
    exclude_nodes = ""
    os.environ.pop('SLURM_CPU_BIND', None) # unset SLURM_CPU_BIND
    executor = submitit.AutoExecutor(folder=logs_dir, slurm_max_num_timeout=3)
    executor.update_parameters(
        nodes=1,
        job_name=job_name,
        gpus_per_node=gpus_per_node,
        tasks_per_node=1,
        cpus_per_task=4,
        mem_gb=mem_gb,
        slurm_constraint="gpu-high",
        slurm_partition=partition,
        timeout_min=timeout_min,
        slurm_wckey="",
        slurm_exclude=exclude_nodes,
        slurm_additional_parameters={"requeue": True}
    )
    return executor