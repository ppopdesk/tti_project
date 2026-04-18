#!/bin/bash
#SBATCH --job-name=te_tune
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/te_tune_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shivanii@umich.edu

module load python3.10-anaconda/2023.03 cuda/12.3.0
export PATH=/home/shivanii/.conda/envs/arm_verl/bin:$PATH
export LD_LIBRARY_PATH=/home/shivanii/.conda/envs/arm_verl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/shivanii/.conda/envs/arm_verl/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

PROJECT_DIR=$HOME/token_entropy
VAL_DATA=/home/shivanii/ARM/data/medqa/test.parquet   # <-- use test parquet as val set for now

mkdir -p $PROJECT_DIR/logs
cd $PROJECT_DIR

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

python3 tune_threshold.py \
    --model_name    /home/shivanii/finetuned_model/qwen_medreason_finetuned \
    --data_path     $VAL_DATA \
    --output_dir    $PROJECT_DIR/tuning_results

echo "Job finished: $(date)"
