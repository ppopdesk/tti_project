#!/bin/bash
#SBATCH --job-name=medqa_hyperparam_tuning
#SBATCH --account=eecs545w26_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/error%x-%j.out

set -euo pipefail

mkdir -p logs

# Load modules
module purge
module load python
module load ollama

# Activate your venv
source venv/bin/activate

pip install -r requirements.txt

# IMPORTANT: run ollama on the same port your code uses (11434)
export OLLAMA_HOST=127.0.0.1:11434
# Optional: store large models somewhere with space (recommended)
# export OLLAMA_MODELS=/scratch/$USER/ollama_models
# mkdir -p "$OLLAMA_MODELS"

# Start Ollama server in background
ollama serve > logs/ollama-${SLURM_JOB_ID}.log 2>&1 &
OLPID=$!

# Give server a moment
sleep 3

# Pull model (first run only; can take time + disk)
ollama pull qwen2.5:7b-instruct

# Sanity check GPU + server
nvidia-smi || true
curl -s http://localhost:11434/api/tags || true

# Run your script
python -u qwen_2_answer_test.py 2>&1 | tee out.txt

# Cleanup
kill $OLPID || true
