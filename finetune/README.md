<!-- finetune -->
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
myenv/bin/accelerate launch \
 --config_file ./configs/accelerate_single_gpu.yaml \
 ./src/model_training/SFT.py \
 --model_path Qwen/Qwen2.5-7B-Instruct \
 --data_path /home/nakarmi/project/545-project/medreason_ours_train.jsonl \
 --n_epochs 1 \
 --experiment_name qwen25_7b_medreason_sft_gpu1 \
 --base_model Qwen \
 --train_bsz_per_gpu 1 \
 --gradient_accumulation_steps 16 \
 --learning_rate 5e-6 \
 --max_seq_len 1024 \
 --optim paged_adamw8bit


<!-- test accuracy -->
python3 src/evaluation/eval.py \
  --model_name ckpts/qwen25_7b_medreason_sft_gpu1/checkpoint-0-32682/tfmr \
  --eval_file eval_data/medqa_test.jsonl \
  --batch_size 4 \
  --max_new_tokens 256 \
  --task_floder medqa_ft_eval
