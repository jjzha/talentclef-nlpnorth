#!/bin/bash

#SBATCH --gres=gpu:a40
#SBATCH --job-name=eval_talentclef
#SBATCH --output=logs/eval_talentclef_%j.out
#SBATCH --error=logs/eval_talentclef_%j.err
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --time=2:00:00 
#SBATCH --mem=128G

export HF_TOKEN="" # for Hugging Face access token, if needed
source $HOME/.bashrc 

singularity exec --nv ~/conda_container.sif python3 scripts/evaluate_taskb.py \
  --validation_root "" \
  --model_path "" \
  --output_dir "" \
  --min_top_k 1 \
  --max_top_k 5000 \
  --threshold_start 0.05 \
  --threshold_end 0.10 \
  --threshold_step 0.05 \
  --custom_tag "" \
  # --instruct_prompt "Given a job title, retrieve the most similar skills" \ # if you work with instruct models, uncomment this line
  # --e5_prefix \ # if you are working with E5 models, uncomment this line
  