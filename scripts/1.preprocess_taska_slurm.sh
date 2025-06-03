#!/bin/bash

#SBATCH --gres=gpu:a10
#SBATCH --job-name=esco-data-process
#SBATCH --output=logs/preprocess_data_%j.out
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --time=12:00:00 
#SBATCH --mem=128G


source $HOME/.bashrc 
singularity exec --nv ~/conda_container.sif python3 scripts/preprocess_job_title_nce_loss_description.py