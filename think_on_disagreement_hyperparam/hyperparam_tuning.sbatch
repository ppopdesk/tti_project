#!/bin/bash
#SBATCH --job-name=hyperparam_tune
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

mkdir -p logs
mkdir -p outputs/grid_search

source ~/.bashrc
conda activate vllm-py310

set -u pipefail

python scripts/hyperparam_tuning.py \
  --model_name "ShivaniiKum/qwen-medreason-finetuned" \
  --data_path "validation.jsonl" \
  --long_reason_json "outputs/long_cot_answers_validation.json" \
  --output_dir "outputs/grid_search"