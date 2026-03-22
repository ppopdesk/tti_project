# `SFT.py` (Supervised Fine-Tuning) — Detailed Guide, Roadmap, and Examples

This document explains how `./src/model_training/SFT.py` works, what data format it expects, how it builds training prompts, how to run it (single GPU / multi GPU), and how to debug common errors (OOM, DeepSpeed issues, etc.). It also includes a roadmap for improving the script.

---

## What `SFT.py` does

`SFT.py` is a **supervised fine-tuning** training script built on:

- **🤗 Transformers** for model + tokenizer loading (`AutoModelForCausalLM`, `AutoTokenizer`)
- **🤗 Accelerate** for device placement / mixed precision / distributed launching (`Accelerator`)
- **PyTorch** for dataloading and the training loop
- **Weights & Biases** (offline) for logging (`wandb`)
- **(Optional) bitsandbytes** for memory-saving **8-bit optimizers**

At a high level:

- It reads a local dataset (`.json` or `.jsonl`)
- Converts each example into a **chat-formatted** conversation (user question → assistant answer-with-reasoning)
- Trains a causal LM with standard next-token prediction
- Masks the **prompt tokens** so loss is computed only on the assistant response
- Saves checkpoints each epoch

---

## File layout (important paths)

- `./src/model_training/SFT.py`: the training script
- `./configs/accelerate_single_gpu.yaml`: single-GPU Accelerate config (no DeepSpeed)
- `./configs/deepspeed_zero3.yaml`: DeepSpeed ZeRO-3 config (multi-GPU), may require `CUDA_HOME`

Outputs:

- `./ckpts/<experiment_name>/checkpoint-<epoch>-<global_step>/...`
- `./train_logs/<experiment_name>/...` (wandb offline logs)

---

## Expected dataset format

`SFT.py` loads data from `--data_path` which must end with:

- `.jsonl` (one JSON object per line), or
- `.json` (a JSON list)

The **schema** used by the current script is decided by whether the `--data_path` string contains `"ours"`:

### MedReason (“ours”) schema

Use this if you exported MedReason and named the file with `ours`, e.g. `medreason_ours_train.jsonl`.

Each record must contain:

```json
{"question":"...","reasoning":"...","answer":"..."}
```

### Huatuo-style schema

Used if `"ours"` is **not** in the path:

```json
{"Question":"...","Complex_CoT":"...","Response":"..."}
```

---

## Prompt + label construction (how training examples are built)

The dataset class is `Train_dataset`.

### Chat templates

`SFT.py` uses a Jinja chat template to format conversations. It supports:

- `--base_model Llama`: uses Llama-3 style headers (`<|start_header_id|>...`)
- `--base_model Qwen`: uses Qwen chat tags (`<|im_start|>...`)

If the loaded tokenizer does not already have `tokenizer.chat_template`, the script assigns its own template.

### What is the “assistant response” for training?

For `"ours"` datasets, the assistant response string is:

- **Thinking**: the dataset’s `reasoning`
- **Final Answer**: the dataset’s `answer`

Rendered as:

```text
## Thinking

{reasoning}

## Final Answer

{answer}
```

For non-`"ours"` paths, it uses `Complex_CoT` and `Response` and formats `## Final Response`.

### How loss masking works

For each example:

- `query` = chat template with only the user message, and an assistant generation prompt
- `input` = chat template with user + assistant content fully filled

Then:

- `labels = [-100] * len(query_ids) + input_ids[len(query_ids):]`

So:

- Prompt tokens get label `-100` (ignored)
- Only assistant tokens contribute to the training loss

### Truncation / padding

- The script truncates from the **left**: `input_ids[-max_seq_len:]`
- Pads with `tokenizer.eos_token_id` in `input_ids`
- Pads with `-100` in `labels`

---

## Training loop (what happens each step)

Main steps inside `train(args)`:

- Create `Accelerator(mixed_precision='bf16', gradient_accumulation_steps=...)`
- Load tokenizer + model
  - The script loads the model with `torch_dtype=bfloat16` when CUDA is available
  - Enables gradient checkpointing
  - Sets `model.config.use_cache = False`
- Build AdamW parameter groups (weight decay vs no decay)
- Create optimizer (see next section)
- Build dataloader
- Compute `num_training_steps` and cosine LR schedule with warmup
- `accelerator.prepare(model, optimizer, dataloader)`
- For each batch:
  - `loss = model(input_ids, labels).loss`
  - `accelerator.backward(loss)`
  - Step optimizer every `gradient_accumulation_steps`
  - Log to wandb (offline) on main process
- Save checkpoint at end of each epoch

---

## Optimizers and memory (why you saw OOM)

Full fine-tuning a 7B model is often limited by **optimizer state memory**.

`SFT.py` supports:

- `--optim adamw` (PyTorch AdamW, highest memory)
- `--optim adamw8bit` (bitsandbytes 8-bit AdamW, lower memory)
- `--optim paged_adamw8bit` (bitsandbytes paged 8-bit AdamW, often best for VRAM fragmentation)

Recommended for single-GPU 7B finetune:

- **`--optim paged_adamw8bit`**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Smaller `--max_seq_len` (start with `1024`, then increase)

---

## How to run (copy/paste examples)

### Example A — Export MedReason to a local file (`ours` schema)

Run with the repo virtualenv:

```bash
/home/nakarmi/project/545-project/myenv/bin/python - <<'PY'
from datasets import load_dataset
import json

out_path = "/home/nakarmi/project/545-project/medreason_ours_train.jsonl"
ds = load_dataset("UCSC-VLAA/MedReason", split="train")

with open(out_path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(json.dumps(
            {"question": ex["question"], "reasoning": ex["reasoning"], "answer": ex["answer"]},
            ensure_ascii=False
        ) + "\n")

print("Wrote:", out_path, "examples:", len(ds))
PY
```

### Example B — Single GPU, memory-safe full finetune (Qwen2.5-7B-Instruct)

```bash
cd /home/nakarmi/project/545-project/MedReason

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/nakarmi/project/545-project/myenv/bin/accelerate launch \
  --config_file ./configs/accelerate_single_gpu.yaml \
  ./src/model_training/SFT.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --data_path /home/nakarmi/project/545-project/medreason_ours_train.jsonl \
  --n_epochs 1 \
  --experiment_name qwen25_7b_medreason_sft \
  --base_model Qwen \
  --train_bsz_per_gpu 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --max_seq_len 1024 \
  --optim paged_adamw8bit
```

### Example C — Multi-GPU with DeepSpeed ZeRO-3 (if your system supports it)

```bash
cd /home/nakarmi/project/545-project/MedReason

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/nakarmi/project/545-project/myenv/bin/accelerate launch \
  --config_file ./configs/deepspeed_zero3.yaml \
  --num_processes 8 \
  ./src/model_training/SFT.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --data_path /home/nakarmi/project/545-project/medreason_ours_train.jsonl \
  --n_epochs 1 \
  --experiment_name qwen25_7b_medreason_sft_z3 \
  --base_model Qwen \
  --train_bsz_per_gpu 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --max_seq_len 2048 \
  --optim adamw
```

> Note: DeepSpeed may fail with `MissingCUDAException: CUDA_HOME does not exist` if CUDA toolkit paths aren’t set on your machine.

---

## Common errors and fixes

### 1) `AttributeError: 'NoneType' object has no attribute 'deepspeed_config'`

Cause: script assumed DeepSpeed plugin exists even when launched without DS.

Fix: run with the included `accelerate_single_gpu.yaml` (no DS), or launch with a DeepSpeed config.

### 2) DeepSpeed fails: `CUDA_HOME does not exist`

Cause: DeepSpeed tries to compile/load CUDA ops and can’t find CUDA toolkit.

Fix options:

- Use `./configs/accelerate_single_gpu.yaml` (no DeepSpeed)
- Or set `CUDA_HOME` to a valid CUDA toolkit path and reinstall/compile DS ops as needed

### 3) CUDA OOM at `optimizer.step()`

Cause: full fine-tuning + AdamW optimizer state is huge.

Fix:

- Use `--optim paged_adamw8bit`
- Reduce `--max_seq_len`
- Free VRAM (other processes)
- Consider LoRA/QLoRA (roadmap below)

### 4) CUDA OOM during `accelerator.prepare()` / `.to(device)`

Cause: GPU is already full (another job running).

Fix:

- Check `nvidia-smi` and kill/stop other processes, or select another GPU:
  - `CUDA_VISIBLE_DEVICES=1 ...`

---

## Roadmap (recommended improvements)

### Short-term (low effort, high value)

- **Schema auto-detection**: don’t rely on `"ours"` substring in `data_path`
- **Disable verbose debug prints**: the script prints decoded prompts early; add a `--debug` flag
- **Add validation split**: compute eval loss periodically
- **Add save frequency**: save every N steps, not only at epoch end

### Medium-term

- **LoRA / QLoRA** support (PEFT):
  - Much lower VRAM than full finetune
  - Faster iteration
- **Packing / sequence concatenation**:
  - Improves throughput and token utilization
- **Flash Attention / SDPA settings**:
  - Speed up long-context training

### Long-term

- **Multi-node training** support (robust cluster configs)
- **Checkpoint resume**: resume from `training_state.pt`
- **Evaluation harness integration**: run `./src/evaluation/*` automatically after training

---

## Tips for practical runs

- Start with `--max_seq_len 1024` and confirm stability before moving to `2048/4096`
- Use `--optim paged_adamw8bit` for single-GPU full finetuning
- Use `WANDB_MODE=offline` (already used) or disable wandb entirely if you want pure stdout

