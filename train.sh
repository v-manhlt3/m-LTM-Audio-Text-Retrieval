#!/bin/bash
#SBATCH --job-name=Orcliz-ot
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=15000
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tien.luong@monash.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A100

conda init bash
source ~/.bashrc
conda activate audio-text
export HYDRA_FULL_ERROR=1 
export WANDB_API_KEY=6592275041b7ab5ee604a4c53c31241d83ff1512

# python train.py -n Zeroshot-NTXent
# python train.py -n test-Gaussian-dotprod-kernel
# python train.py -n 0reg-reproduce-second
# python train.py -n CL-train-wavcaps -c rebuttal-wavcaps

# python train.py -n AudioCaps-m-LTM -c rebuttal-settings
# python train.py -n test2 -c m-ltm-settings
python train.py -n test_orcliz -c m-ltm-settings