#!/bin/bash
#SBATCH --job-name=te_infer
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/te_infer_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shivanii@umich.edu

module load python3.10-anaconda/2023.03 cuda/12.3.0
export PATH=/home/shivanii/.conda/envs/arm_verl/bin:$PATH
export LD_LIBRARY_PATH=/home/shivanii/.conda/envs/arm_verl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/shivanii/.conda/envs/arm_verl/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

PROJECT_DIR=$HOME/token_entropy
TEST_DATA=/home/shivanii/ARM/data/medqa/test.parquet

# !! SET THIS after running the tuning job !!
THRESHOLD=0.1


mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Threshold: $THRESHOLD"
nvidia-smi --query-gpu=name --format=csv,noheader

python3 token_entropy.py \
    --model_name    Qwen/Qwen2.5-7B-Instruct \
    --data_path     $TEST_DATA \
    --threshold     $THRESHOLD \
    --num_questions -1 \
    --output_dir    $PROJECT_DIR/results

echo "Job finished: $(date)"
