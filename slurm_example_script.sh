#!/bin/bash
#SBATCH --account=mrs_2
#SBATCH --qos=h200_mrs_shared
#SBATCH --job-name=unir_gsm8k
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --output=logs/unir_gsm8k-%j.out
#SBATCH --error=logs/unir_gsm8k-%j.err
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
  --dataset_name openai/gsm8k \
  --dataset_config main \
  --output_dir run/GSM8k-llama-backbone3b_reasoning1b \
  --run_name GSM8k-llama-backbone3b_reasoning1b \
  --ref_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
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

