#!/bin/bash
#SBATCH --job-name=aliu40_train_model
#SBATCH --partition=free-gpu        # use 'gpu' if you have allocated hours
#SBATCH --account=aliu40      # only needed for 'gpu' partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                # request 1 GPU
#SBATCH --time=04:00:00             # max walltime
#SBATCH --output=slurm-%j.out

module load anaconda/2024.06
conda activate mymodel

python train.py
