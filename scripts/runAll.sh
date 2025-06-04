# preprocessing
singularity exec --nv ~/conda_container.sif python3 scripts/0.preprocess_job_title_nce_loss_description.py
singularity exec --nv ~/conda_container.sif python3 scripts/0.preprocess_skill_nce_loss_description.py

# Training

singularity exec --nv ~/conda_container.sif python3 scripts/2.contrastive.train.py \
  --data_path "" \
  --model_name "" \
  --output_dir "" \
  --num_train_epochs 1 \
  --batch_size 16 \
  --learning_rate 2e-6 \
  # --e5_prefix \ # if you are working with E5 models, uncomment this line
singularity exec --nv ~/conda_container.sif python3 scripts/2.contrastive.train.py \
  --data_path "" \
  --model_name "" \
  --output_dir "" \
  --num_train_epochs 1 \
  --batch_size 16 \
  --learning_rate 2e-6 \
  # --e5_prefix \ # if you are working with E5 models, uncomment this line


singularity exec --nv ~/conda_container.sif python3 scripts/4.evaluate_taska.py \
  --validation_root "" \
  --model_path "" \
  --output_dir "" \
  --languages "" \
  --min_top_k 1 \
  --max_top_k 5000 \
  --threshold_start 0.05 \
  --threshold_end 0.10 \
  --threshold_step 0.05 \
  --custom_tag "" \
  # --instruct_prompt "Given a job title, retrieve the most similar job titles" \ # if you work with instruct models, uncomment this line
  # --e5_prefix \ # if you are working with E5 models, uncomment this line
singularity exec --nv ~/conda_container.sif python3 scripts/4.evaluate_taskb.py \
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
  
