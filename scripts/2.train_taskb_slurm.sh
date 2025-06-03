#!/bin/bash

#SBATCH --gres=gpu:a40
#SBATCH --job-name=train-talentcelf
#SBATCH --output=logs/train_talentclef_%j.out
#SBATCH --error=logs/train_talentclef_%j.err
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --time=2-00:00:00 
#SBATCH --mem=128G

export HF_TOKEN=""

source $HOME/.bashrc 
singularity exec --nv ~/conda_container.sif python3 scripts/train_contrastive.py \
  --data_path "" \
  --model_name "" \
  --output_dir "" \
  --num_train_epochs 1 \
  --batch_size 16 \
  --learning_rate 2e-6 \
  # --e5_prefix \ # if you are working with E5 models, uncomment this line
