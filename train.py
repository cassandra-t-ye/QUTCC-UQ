import argparse
import json
import random
import shutil
import time
import wandb
import glob
import os
import re
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import transforms

import datasets.fastmri.utils as fastmri_utils
import models
from datasets.bsccm import get_bsccm_dataloaders
from datasets.fmd.data_loader import (fluore_to_tensor,
                                                load_noisy_all_loaders)
from datasets.fastmri import FastMRIDataset

from datasets.LidcIdri import load_ct_dataloader

from models.pinball import BatchedPinballLoss, PinballLoss
from utils import load_yaml, mkdirs, module_size
from utils.metrics import cal_psnr, cal_ssim2, cal_lpips
from utils.plot import save_samples, save_stats
from utils.practices import OneCycleScheduler, adjust_learning_rate
from models.checkpointing import create_model, find_latest_checkpoint, load_checkpoint_for_training

plt.switch_backend('agg')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Quantile UQNet')
    parser.add_argument('--exp-name', type=str, help='experiment name', required=True)
    parser.add_argument('--exp-dir', type=str, default="experiments", help='directory to save output of experiments')
    parser.add_argument('--data-root', type=str, help='directory to dataset root', required=True)
    parser.add_argument('--experiment-type', type=str, choices=["MRI", 
                                                                "Denoising", 
                                                                "QPI", 
                                                                "CT"], help='experiment type', required=True)

    parser.add_argument('--net', type=str, default='unet_quantile', choices=['unet_quantile',
                                                                             'unet_im2im', 
                                                                             'im2im'], )
    parser.add_argument('--imsize', type=int, default=256, help='image size')
    parser.add_argument('--in-channels', type=int, default=1, help='input channels')
    parser.add_argument('--out-channels', type=int, default=1, help='output channels')
    parser.add_argument('--transform', type=str, default='center_crop', choices=['four_crop', 'multi_image', 'center_crop'], help='data transform')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=0., help="weight decay")
    parser.add_argument('--ckpt-freq', type=int, default=5, help='how many epochs to wait before saving model')
    parser.add_argument('--print-freq', type=int, default=100, help='how many minibatches to wait before printing training status')
    parser.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
    parser.add_argument('--plot-freq', type=int, default=5, help='how many epochs to wait before plotting test output')

    # Resume training arguments
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint folder or specific checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true', default=False, help='automatically resume from latest checkpoint in exp-dir if available')
    
    #FMD-args
    parser.add_argument('--noise-type', type=str, default="none", choices=['gaussian', 'poisson', 'real', 'none'], help='noise type')
    parser.add_argument('--noise-levels-train', type=list, default=[1], help='noise levels for training')
    parser.add_argument('--noise-levels-test', type=list, default=[1], help='noise levels for testing')
    parser.add_argument('--sigma', type=float, default = 0, help='maximum noise level for training')

    #CT-args
    parser.add_argument('--ct-step-size', type=int, default=800, help='step size for CT radon transform')
    # WandB-specific args 
    parser.add_argument('--wandb-project', type=str, help='UQ-NET project name')
    parser.add_argument('--wandb-entity', type=str, help='wandb entity (team/user)')
    #Misc-args
    parser.add_argument('--debug', action='store_true', default=False, help='enable verbose stdout')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device number')
    parser.add_argument('--cmap', type=str, default='inferno', help='colormap for plotting')
    return parser.parse_args()

def setup_experiment(args):
    resume_checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            # Direct path to checkpoint file
            resume_checkpoint = args.resume
            args.run_dir = os.path.dirname(os.path.dirname(args.resume))  # Go up from checkpoints/epoch*.pth
        elif os.path.isdir(args.resume):
            # Path to experiment directory
            checkpoint_dir = os.path.join(args.resume, 'checkpoints')
            resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
            args.run_dir = args.resume
        else:
            print(f"Resume path {args.resume} does not exist. Starting fresh training.")
    elif args.auto_resume:
        # Look for existing experiments with same name
        potential_dirs = glob.glob(f"{args.exp_dir}/{args.exp_name}/*")
        if potential_dirs:
            # Find the most recent experiment directory
            latest_dir = max(potential_dirs, key=os.path.getctime)
            checkpoint_dir = os.path.join(latest_dir, 'checkpoints')
            resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if resume_checkpoint:
                args.run_dir = latest_dir
                print(f"Auto-resuming from {latest_dir}")
    
    if not resume_checkpoint:
        args.run_dir = f"{args.exp_dir}/{args.exp_name}/" \
                       f"{args.net}_epochs{args.epochs}_bs{args.batch_size}" \
                       f"_{time.strftime('%Y-%m-%d-%H.%M.%S')}"
        shutil.rmtree(args.run_dir, ignore_errors=True)
    
    args.ckpt_dir = args.run_dir + '/checkpoints'
    args.train_dir = args.run_dir + "/training"
    args.pred_dir = args.train_dir + "/predictions"
    mkdirs([args.run_dir, args.ckpt_dir, args.train_dir, args.pred_dir])
    args.resume_checkpoint = resume_checkpoint

    print(f"Seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    print('Arguments:')
    pprint(vars(args))
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)

def get_transform(args):
    if args.transform == 'four_crop':
        return transforms.Compose([
            transforms.FiveCrop(args.imsize),
            transforms.Lambda(lambda crops: torch.stack([fluore_to_tensor(crop) for crop in crops[:4]])),
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
    elif args.transform == 'multi_image':
        return transforms.Compose([
            transforms.FiveCrop(args.imsize),
            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops[:4]])),
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
    elif args.transform == 'center_crop':
        return None
    else:
        raise ValueError(f"Unknown transform: {args.transform}")
    
def get_dataloaders(args, transform):
    if args.experiment_type == "MRI": # TODO: Handle this in a more generic way
        input_path = args.data_root + "/training"
        mask_info = {'type': 'equispaced', 'center_fraction': [0.08], 'acceleration': [4]}
        
        train_dataset = FastMRIDataset(input_path, normalize_input='standard', normalize_output='min-max', mask_info=mask_info, num_volumes=0)
        fastmri_utils.normalize_dataset(train_dataset)
        train_size = int(0.9 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        print(f"Experiment type: {args.experiment_type} | Train dataset size: {len(train_loader.dataset)} | Test dataset size: {len(test_loader.dataset)}")
    
    elif args.experiment_type == "Denoising":
        if args.noise_type == "real":
            gt_pathway = args.data_root + "/real_noise_new/train/gt"
            noisy_pathway = args.data_root + "/real_noise_new/train/noisy"

            train_loader, test_loader = load_noisy_all_loaders(
                noise_type=args.noise_type, 
                sigma_max=None, 
                image_folder=None, 
                batch_size=args.batch_size, 
                gt_path = gt_pathway,
                noisy_path = noisy_pathway,
                transform=transform)
            
        else: 
            input_path = args.data_root + "/training"
            train_loader, test_loader = load_noisy_all_loaders(
                noise_type=args.noise_type, 
                sigma_max=args.sigma, 
                image_folder=input_path, 
                batch_size=args.batch_size, 
                gt_path = None, 
                noisy_path = None,
                transform=transform,
            )
        print(f"Experiment type: {args.experiment_type} | Noise Type: {args.noise_type} | Train dataset size: {len(train_loader.dataset)} | Test dataset size: {len(test_loader.dataset)}")
    elif args.experiment_type == "QPI":
        train_loader, test_loader, calib_indices, test_indices = get_bsccm_dataloaders(
            dataset_path=args.data_root,
            batch_size=args.batch_size,
            input_channels=['DPC_Left', 'DPC_Right'],
            validation_split=0.1,
            calib_split=0.2,
            test_split=0.1,
            transform=transform,
            normalization='min-max',
            num_workers=4,
        )
        json.dump(calib_indices.tolist(), open(args.run_dir + "/calib_indices.json", 'w'), indent=4)
        json.dump(test_indices.tolist(), open(args.run_dir + "/test_indices.json", 'w'), indent=4)
        print(f"Experiment type: {args.experiment_type} | Train dataset size: {len(train_loader.dataset)} | Test dataset size: {len(test_loader.dataset)} | Calib dataset size: {len(calib_indices)} | Test dataset size: {len(test_indices)}")

    elif args.experiment_type == "CT":
        train_loader = load_ct_dataloader(
            root_path=args.data_root,
            dataset_type = "train",
            batch_size = args.batch_size,
            steps = args.ct_step_size
        )

        test_loader = load_ct_dataloader(
            root_path=args.data_root,
            dataset_type = "test",
            batch_size = args.batch_size,
            steps = args.ct_step_size
        )
        [print(f"Experiment type: {args.experiment_type} | Train dataset size: {len(train_loader.dataset)} | Test dataset size:{len(test_loader.dataset)}")]
     
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment_type}")
    
    return train_loader, test_loader


###################### Core training logic ############################
def calculate_output_and_loss(model, net_type, noisy, clean, pinball_05, pinball_95, pinball, device):
    # TODO: Check the shapes of the tensors to make sure they are correct
    assert noisy.ndim >= 4 and clean.ndim >= 4, f"Expected BCHW format, got noisy: {noisy.shape}, clean: {clean.shape}"
    batch_size = noisy.shape[0]
    denoised, loss = None, None
    # print(f"noisy shape: {noisy.shape}, clean shape: {clean.shape}")

    if net_type in ['unet_quantile']:
        curr_quantiles = torch.rand(batch_size, device=device, dtype=torch.float32)
        curr_quantiles[curr_quantiles == 0] = 1e-7
        pred = denoised = model(noisy, curr_quantiles)
        # print(f"noisy shape: {noisy.shape}, clean shape: {clean.shape}, curr_quantiles shape: {curr_quantiles.shape}, pred shape: {pred.shape}")
        loss = pinball(pred, clean, curr_quantiles)

    elif net_type in ['unet_im2im', 'im2im']:
        if net_type == 'unet_im2im':
            timevect = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
            denoised = model(noisy, timevect)
        else:
            denoised = model(noisy)
        pred = denoised[:, 1:2]
        loss = F.mse_loss(pred, clean, reduction='mean') + \
            pinball_05(denoised[:,0:1], clean) + pinball_95(denoised[:,2:], clean)

    else:
        raise ValueError(f"Unknown network type for loss calculation: {net_type}")

    return denoised, loss, pred

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, args, pinball_05, pinball_95, pinball, device, total_steps):
    model.train()
    epoch_psnr, epoch_mse, epoch_lpips, epoch_ssim, iters = 0., 0., 0., 0., 0
    start_epoch_time = time.time()

    for batch_idx, (noisy, clean) in enumerate(train_loader):
        iters += 1
        noisy, clean = noisy.to(device), clean.to(device)

        if args.transform in ['four_crop', 'multi_image']:
            noisy = noisy.view(-1, *noisy.shape[2:])
            clean = clean.view(-1, *clean.shape[2:])

        model.zero_grad(set_to_none=True)
        _, loss, pred = calculate_output_and_loss(
            model, args.net, noisy, clean, pinball_05, pinball_95, pinball, device)

        step = epoch * len(train_loader) + batch_idx
        lr = scheduler.step(step / total_steps)
        adjust_learning_rate(optimizer, lr)

        loss.backward()
        optimizer.step()

        epoch_mse += loss.item()
        with torch.no_grad():
            epoch_psnr += cal_psnr(clean, pred).item()
            epoch_ssim += cal_ssim2(clean, pred).item()
            epoch_lpips += 0 # cal_lpips(clean, pred).item()
            if iters % args.print_freq == 0:
                print(f'[{batch_idx+1}/{len(train_loader)}][{epoch}/{args.epochs}] '
                    f'PSNR: {epoch_psnr/iters:.4f} | SSIM: {epoch_ssim/iters:.4f} | '
                    f'LPIPS: {epoch_lpips/iters:.4f} | Time: {time.time()-start_epoch_time:.2f}s')

    epoch_duration = time.time() - start_epoch_time
    avg_psnr  = epoch_psnr  / len(train_loader)
    avg_ssim  = epoch_ssim  / len(train_loader)
    avg_lpips = epoch_lpips / len(train_loader)
    avg_rmse  = np.sqrt(epoch_mse / len(train_loader))
    print(f"Epoch {epoch} Training Summary | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | "
          f"LPIPS: {avg_lpips:.4f} | RMSE: {avg_rmse:.6f} | Time: {time.time()-start_epoch_time:.2f}s")
    return avg_psnr, avg_ssim, avg_lpips, avg_rmse

def evaluate(model, test_loader, epoch, args, pinball_05, pinball_95, pinball, device):
    model.eval()
    epoch_mse, epoch_psnr, epoch_ssim, epoch_lpips = 0., 0., 0., 0.
    saved_samples = []

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            batch_size = noisy.shape[0]
        
            pred = None
            denoised = None

            if args.net in ['unet_quantile']:
                curr_quantile = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
                pred = denoised = model(noisy, curr_quantile)
                loss = F.mse_loss(pred, clean, reduction='mean')
            elif args.net == "unet_im2im":
                timevect = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
                denoised = model(noisy, timevect)
                pred = denoised[:,1:2, :, :]
                loss = F.mse_loss(pred, clean, reduction='mean') \
                    + pinball_05(denoised[:,0:1, :, :], clean) + pinball_95(denoised[:,2:, :, :], clean)
            elif args.net == 'im2im':
                denoised = model(noisy)
                pred = denoised[:,1:2, :, :]
                loss = F.mse_loss(pred, clean, reduction='mean') \
                    + pinball_05(denoised[:,0:1, :, :], clean) + pinball_95(denoised[:,2:, :, :], clean)
 
            epoch_mse += loss.item()
            epoch_psnr += cal_psnr(clean, pred).item()
            epoch_ssim += cal_ssim2(clean, pred).item()
            epoch_lpips += 0 # cal_lpips(clean, pred).item()

            if epoch % args.plot_freq == 0 and len(saved_samples) < 8:
                sampled_idx = 0
                sample = {'noisy': noisy[sampled_idx:sampled_idx+1].cpu(),
                        'clean': clean[sampled_idx:sampled_idx+1].cpu(),
                        'pred': pred[sampled_idx:sampled_idx+1].cpu()}
                if denoised.shape[1] == 3:
                    sample['denoised_05'] = denoised[sampled_idx:sampled_idx+1, 0:1].cpu()
                    sample['denoised_95'] = denoised[sampled_idx:sampled_idx+1, 2:].cpu()
                saved_samples.append(sample)

    avg_psnr  = epoch_psnr  / len(test_loader)
    avg_ssim  = epoch_ssim  / len(test_loader)
    avg_lpips = epoch_lpips / len(test_loader)
    avg_rmse  = np.sqrt(epoch_mse / len(test_loader))
    print(f"Epoch {epoch} Test Summary | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | "
          f"LPIPS: {avg_lpips:.4f} | RMSE: {avg_rmse:.6f}")

    if epoch % args.plot_freq == 0 and saved_samples:
        print(f"Epoch {epoch} | Plotting {len(saved_samples)} test predictions...")
        noisy_batch = torch.cat([s['noisy'] for s in saved_samples], dim=0)
        clean_batch = torch.cat([s['clean'] for s in saved_samples], dim=0)
        pred_batch = torch.cat([s['pred'] for s in saved_samples], dim=0)
        samples_to_save = torch.cat([noisy_batch, pred_batch, clean_batch], dim=0)
        save_samples(args.pred_dir, samples_to_save, epoch, f'test_pred_epoch{epoch}', epoch=False, cmap=args.cmap)
    
    return avg_psnr, avg_ssim, avg_lpips, avg_rmse

def post_training(model, logger, args, start_time, epoch=None):
    training_time = time.time() - start_time
    status = "Finished" if epoch is None else f"Interrupted at epoch {epoch}"
    print(f"{status} training using {training_time:.2f} seconds")

    final_epoch = args.epochs if epoch is None else epoch

    if logger['rmse_train']:
        min_len = min(len(logger['rmse_train']), len(logger['rmse_test']), 
                      len(logger['psnr_train']), len(logger['psnr_test']))
        x_axis = np.arange(args.log_freq, min_len * args.log_freq + 1, args.log_freq)[:min_len]
        logger_trimmed = {k: v[:min_len] for k, v in logger.items()}

        save_stats(args.train_dir, logger_trimmed, x_axis, 
                   'psnr_train', 'psnr_test', 'rmse_train', 'rmse_test')
        print(f"Training statistics saved to {args.train_dir}")

    save_path = f"{args.ckpt_dir}/epoch{final_epoch}_{'final' if epoch is None else 'interrupted'}.pth"
    try:
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'logger': logger,
            'args': vars(args),
        }, save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    args.training_time = training_time
    args.n_params, args.n_layers = module_size(model)
    try:
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)
            print(f"Arguments saved to {args.run_dir}/args.txt")
    except Exception as e:
        print(f"Error saving arguments: {e}")

###################### Main loop ############################
def main(args):
    setup_experiment(args)

    wandb_name = f"{args.exp_name}-{args.net}-bs{args.batch_size}"
    wandb_name += f"-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        config=vars(args)
    )
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = create_model(args.net, device, args.in_channels, args.out_channels, args.debug)
    wandb.watch(model, log='all', log_freq=args.print_freq)
    transform = get_transform(args)
    train_loader, test_loader = get_dataloaders(args, transform)
    print(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=[0.9, 0.99])
    scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=10, pct_start=0.3)
    total_steps = len(train_loader) * args.epochs

    start_epoch = 1
    logger = {'psnr_train': [], 'rmse_train': [], 'psnr_test': [], 'rmse_test': []}
    
    # Load checkpoint if resuming
    if args.resume_checkpoint:
        start_epoch, logger = load_checkpoint_for_training(args.resume_checkpoint, model, optimizer, device)
    pinball_05 = PinballLoss(quantile=0.05)
    pinball_95 = PinballLoss(quantile=0.95)
    pinball = BatchedPinballLoss(reduction='mean').to(device)

    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    start_time = time.time()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            psnr_train, ssim_train, lpips_train, rmse_train = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, args, 
                pinball_05, pinball_95, pinball, device, total_steps)

            wandb.log({
                'epoch': epoch,
                'train/psnr':  psnr_train,
                'train/ssim':  ssim_train,
                # 'train/lpips': lpips_train,
                'train/rmse':  rmse_train,
            })
            
            psnr_test, ssim_test, lpips_test, rmse_test = evaluate(
                model, test_loader, epoch, args, pinball_05, pinball_95, pinball, device)

            wandb.log({
                'epoch': epoch,
                'val/psnr':  psnr_test,
                'val/ssim':  ssim_test,
                # 'val/lpips': lpips_test,
                'val/rmse':  rmse_test,
            })
                        
            if epoch % args.log_freq == 0:
                logger['psnr_train'].append(psnr_train)
                logger['rmse_train'].append(rmse_train)
                logger['psnr_test'].append(psnr_test)
                logger['rmse_test'].append(rmse_test)

            if epoch % args.ckpt_freq == 0:
                ckpt_path = f"{args.ckpt_dir}/epoch{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'logger': logger,
                    'args': vars(args),
                }, ckpt_path)
                print(f"Checkpoint saved at {ckpt_path}")
            
        post_training(model, logger, args, start_time)
    except (KeyboardInterrupt, Exception) as e:
        print(f'\n{"Keyboard Interrupt" if isinstance(e, KeyboardInterrupt) else f"An error occurred: {e}"}')
        print('Saving models & training logs...')
        post_training(model, logger, args, start_time, epoch=epoch)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
