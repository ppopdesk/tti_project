#!/bin/bash
#SBATCH --job-name=medqa_hyperparam_tuning
#SBATCH --account=eecs545w26_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G             # Increased to 32G for safer overhead with 7B models
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/error%x-%j.out

set -euo pipefail

mkdir -p logs

# Load modules
module purge
module load python

# Activate your venv
source venv/bin/activate

pip install -r requirements.txt

# 4. Handle Model Caching (Crucial for Slurm)
# By default, Transformers saves to ~/.cache. This can fill up your home quota.
# It is better to point this to a scratch directory.
export HF_HOME=/scratch/$USER/huggingface_cache
mkdir -p "$HF_HOME"

# 5. Sanity check GPU
nvidia-smi || true

# Run your script
python -u confidence_based_reasoning/tune_log_diff_threshold.py > logs/out-test.txt 2>&1
