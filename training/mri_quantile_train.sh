#!/bin/bash
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 # Request 1 GPU (adjust type as needed)
#SBATCH -N 1                                  # Number of nodes
#SBATCH -n 16                                 # Number of CPU cores
#SBATCH -t 48:00:00                           # Time limit (hh:mm:ss)
#SBATCH --mem 32gb                            # Memory requirement

source training/setup.sh

python -u train.py \
    --net unet_quantile \
    --transform "center_crop" \
    --epochs 50 \
    --experiment-type "MRI" \
    --data-root /share/monakhova/Cassandra_data/UQNet_proj/Fast_MRI/RAW_singlecoil_train \
    --in-channels 1 \
    --noise-type "poisson" \
    --sigma 0.75 \
    --batch-size 12 \
    --exp-name "mri" \
    --ckpt-freq 2 \
