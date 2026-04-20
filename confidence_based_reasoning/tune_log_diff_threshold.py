import sys
import os
import re
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

# --- Path Handling ---
# Repo root on path: needed when running this file directly (e.g. Slurm)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import your custom prompts from prompts.py
try:
    from prompts import NO_REASONING_PROMPT, LONG_REASONING_PROMPT
except ImportError:
    print("Warning: prompts.py not found. Please ensure it is in the same directory or repo root.")
    # Fallback placeholders if needed
    NO_REASONING_PROMPT = "Question: {question}\nOptions: {options_text}\nAnswer:"
    LONG_REASONING_PROMPT = "Question: {question}\nOptions: {options}\nReasoning:"

# --- Configuration ---
MODEL_NAME = "ShivaniiKum/qwen-medreason-finetuned"
VAL_SIZE = 300
SAVE_EVERY = 10

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("MEDQA_CALIBRATION_DIR", _SCRIPT_DIR)).resolve()
OUTPUT_FILE = OUTPUT_DIR / "medqa_calibration_data.json"
GRID_SEARCH_CSV = OUTPUT_DIR / "medqa_threshold_grid_search.csv"

# --- Utilities ---

def _atomic_write_json(path: Path, obj) -> None:
    """Prevents file corruption by writing to a temp file then swapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        # We use float() conversion because numpy/torch types aren't JSON serializable
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)

def load_model_and_tokenizer(model_name: str):
    """Loads the fine-tuned model and ensures pad_token is set."""
    print(f"Loading tokenizer and model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        return model.eval(), tokenizer
    except Exception as e:
        print(f"PEFT load failed ({e}), attempting standard load...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        return model.eval(), tokenizer

def build_hf_prompt(tokenizer, user_text: str, assistant_prefix: str = ""):
    """Formats prompt using Qwen template and injects target prefix for steering."""
    messages = [{"role": "user", "content": user_text}]
    # add_generation_prompt adds the <|im_start|>assistant header
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + assistant_prefix

# --- Core Inference Logic ---

def get_no_reasoning_answer(model, tokenizer, question, options_text):
    """
    Calculates log_diff: log(P_top1) - log(P_top2).
    Injects <ANSWER> tag to force the model to pick a letter immediately.
    """
    raw_content = NO_REASONING_PROMPT.format(question=question, options_text=options_text)
    full_prompt = build_hf_prompt(tokenizer, raw_content, assistant_prefix="<ANSWER>\n")
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    
    # Map letters to their specific token IDs in Qwen's vocab
    choice_ids = {letter: tokenizer.encode(letter, add_special_tokens=False)[-1] for letter in ['A', 'B', 'C', 'D']}
    choice_probs = {letter: probs[idx].item() for letter, idx in choice_ids.items()}
    
    sorted_choices = sorted(choice_probs.items(), key=lambda x: x[1], reverse=True)
    best_letter, best_prob = sorted_choices[0]
    second_letter, second_prob = sorted_choices[1]
    
    # Calculate Log-Diff
    log_diff = np.log(best_prob + 1e-9) - np.log(second_prob + 1e-9)
    
    return best_letter, second_letter, float(log_diff), 1

def get_reasoned_answer(model, tokenizer, question, options_text):
    """Generates a full chain-of-thought response starting with <LONG_COT>."""
    raw_content = LONG_REASONING_PROMPT.format(question=question, options=options_text)
    full_prompt = build_hf_prompt(tokenizer, raw_content, assistant_prefix="<LONG_COT>\n")
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Extract A/B/C/D from the <ANSWER> tag in the CoT output
    match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", full_text, re.IGNORECASE)
    answer = match.group(1).upper() if match else None
    
    return answer, len(generated_ids)

# --- Data Collection and Analysis ---

def collect_data(model, tokenizer):
    """Main loop to collect MedQA validation results with resume support."""
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="validation", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    results = []
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                results = json.load(f)
            print(f"Resuming from index {len(results)}...")
        except (json.JSONDecodeError, OSError):
            results = []

    next_i = len(results)

    for i, entry in enumerate(ds):
        if i >= 3:
            break
        if i < next_i:
            continue

        q_text = entry['sent1']
        opts_list = [f"{idx_to_letter[j]}) {entry[f'ending{j}']}" for j in range(4)]
        opts_text = "\n".join(opts_list)
        correct = idx_to_letter[entry['label']]

        # 1. Quick Probe
        probe_pred, _, log_diff, probe_tokens = get_no_reasoning_answer(model, tokenizer, q_text, opts_text)
        
        # 2. Full Reasoning
        reason_pred, reason_tokens = get_reasoned_answer(model, tokenizer, q_text, opts_text)

        results.append({
            "id": i,
            "log_diff": log_diff,
            "probe_correct": bool(probe_pred == correct),
            "reason_correct": bool(reason_pred == correct),
            "probe_tokens": probe_tokens,
            "reason_tokens": int(reason_tokens)
        })

        if len(results) % SAVE_EVERY == 0:
            _atomic_write_json(OUTPUT_FILE, results)
            print(f"Checkpoint: saved {len(results)}/{VAL_SIZE} rows to {OUTPUT_FILE}", flush=True)

    _atomic_write_json(OUTPUT_FILE, results)
    return results

def run_grid_search(data):
    """Calculates accuracy and token usage across a range of log_diff thresholds."""
    thresholds = np.linspace(0, 25, 101)
    analysis = []

    for T in thresholds:
        total_correct = 0
        total_tokens = 0
        escalated_count = 0

        for row in data:
            # If the model is uncertain (low log_diff), use reasoning
            if row['log_diff'] < T:
                total_correct += 1 if row['reason_correct'] else 0
                total_tokens += (row['probe_tokens'] + row['reason_tokens'])
                escalated_count += 1
            else:
                # If certain enough, trust the quick probe
                total_correct += 1 if row['probe_correct'] else 0
                total_tokens += row['probe_tokens']

        acc = total_correct / len(data)
        avg_tokens = total_tokens / len(data)
        
        analysis.append({
            "threshold": round(float(T), 2),
            "accuracy": round(acc, 4),
            "avg_tokens": round(avg_tokens, 2),
            "escalation_rate": round(escalated_count / len(data), 2)
        })

    df = pd.DataFrame(analysis)
    print("\n--- Grid Search Results (Summary) ---")
    print(df[::10].to_string(index=False))
    
    best_acc = df['accuracy'].max()
    best_row = df[df['accuracy'] == best_acc].iloc[0]
    print(f"\nMax Accuracy: {best_acc} at Threshold {best_row['threshold']}")
    return df

# --- Main Entry Point ---

def main():
    # 1. Setup Model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # 2. Collect Data (or load existing)
    if not OUTPUT_FILE.exists() or len(json.load(open(OUTPUT_FILE))) < VAL_SIZE:
        data = collect_data(model, tokenizer)
    else:
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)

    # 3. Analyze Thresholds
    df = run_grid_search(data)
    
    # 4. Export Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GRID_SEARCH_CSV, index=False)
    print(f"\nGrid search results exported to {GRID_SEARCH_CSV}")

if __name__ == "__main__":
    main()