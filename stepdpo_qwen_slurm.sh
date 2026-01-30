#!/bin/bash
#SBATCH --account=all
#SBATCH --partition=learnai
#SBATCH --job-name=unir_stepdpo_qwen
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --output=logs/unir_stepdpo_qwen-%j.out
#SBATCH --error=logs/unir_stepdpo_qwen-%j.err
#SBATCH --time=72:00:00

cd "/storage/home/tamboli/UniR"
source "/storage/home/tamboli/miniconda3/etc/profile.d/conda.sh"
source ~/.bashrc
conda activate "unir_fresh"

export HF_HOME="/storage/home/tamboli/.cache/huggingface"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --main_process_port 6667 \
  --num_processes=2 \
  src/unir/train.py \
  --config recipes/unir.yaml \
  --report_to wandb \
  --logging_steps 16  \
  --dataset_name xinlai/Math-Step-DPO-10K \
  --dataset_config default \
  --output_dir run/unir_qwen_math_step_dpo_10k \
  --run_name unir_qwen_math_step_dpo_10k \
  --ref_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --num_generations 8 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 8 \
  --max_completion_length 1024 \
  --max_steps 1000 \
  --save_steps 100 \
  --beta 0.0 \
  --system_prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Your response should be in the following format: <think>
Your reasoning here
</think>
<answer>
 answer here 
</answer>. The reasoning process Note that respond by English, NOT use other languages." \
  --reward_funcs rule_based_accuracy \
  --reward_weights 1.0

