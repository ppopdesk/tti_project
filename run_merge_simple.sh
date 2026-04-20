#!/bin/bash
#SBATCH --job-name=merge_lora
#SBATCH --account=eecs545w26_class
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/home/shivanii/logs/merge_%j.log

module load python3.10-anaconda/2023.03 cuda/12.3.0
export PATH=/home/shivanii/.conda/envs/arm_verl/bin:$PATH
export LD_LIBRARY_PATH=/home/shivanii/.conda/envs/arm_verl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/shivanii/.conda/envs/arm_verl/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

mkdir -p /home/shivanii/finetuned_model/checkpoint_merged
echo "Job started: $(date)"
python3 ~/merge_lora.py
echo "Job finished: $(date)"
