import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "ShivaniiKum/qwen-medreason-finetuned"
VAL_SIZE = 300

# Path handling consistent with your original script
_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("MEDQA_CALIBRATION_DIR", _SCRIPT_DIR)).resolve()
OUTPUT_FILE = OUTPUT_DIR / "medqa_calibration_data_vllm.json"
GRID_SEARCH_CSV = OUTPUT_DIR / "medqa_threshold_grid_search.csv"

# Token IDs for A-D in Qwen2.5 (Verified)
LETTER_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}

# --- Utilities ---

def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    tmp_path.replace(path)

def format_prompt(tokenizer, question, options_text, target_tag):
    """Matches the teammate's chat template formatting logic."""
    user_content = f"Question: {question}\nOptions: {options_text}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<{target_tag}>\n"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

def extract_answer_letter(text: str) -> str:
    """Extracts A/B/C/D from the generated reasoning text."""
    match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text.upper())
    return matches[-1] if matches else "A"

# --- Main Logic ---

def main():
    print(f"Loading tokenizer and vLLM model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Initialize vLLM the same way her code does
    # Note: On V100, float16 is often faster than bfloat16
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="float16", 
    )

    print("Loading MedQA dataset...")
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="validation")
    examples = ds.select(range(min(VAL_SIZE, len(ds))))
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    # 1. BATCH FAST PROBE (to get log_diff)
    print(f"Running fast probes on {len(examples)} questions...")
    probe_prompts = []
    for ex in examples:
        opts = "\n".join([f"{idx_to_letter[j]}) {ex[f'ending{j}']}" for j in range(4)])
        probe_prompts.append(format_prompt(tokenizer, ex['sent1'], opts, "ANSWER"))

    # Logprobs=20 ensures we get the scores for all 4 letters
    probe_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    probe_outputs = llm.generate(probe_prompts, probe_params)

    # 2. BATCH REASONING PASS
    print(f"Running full reasoning on {len(examples)} questions...")
    reason_prompts = []
    for ex in examples:
        opts = "\n".join([f"{idx_to_letter[j]}) {ex[f'ending{j}']}" for j in range(4)])
        reason_prompts.append(format_prompt(tokenizer, ex['sent1'], opts, "COT"))

    reason_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=1024)
    reason_outputs = llm.generate(reason_prompts, reason_params)

    # 3. PROCESS RESULTS
    results = []
    for i in range(len(examples)):
        ex = examples[i]
        correct = idx_to_letter[ex['label']]
        
        # Calculate Log-Diff from vLLM logprobs
        lp_dict = probe_outputs[i].outputs[0].logprobs[0]
        choice_lps = {
            letter: lp_dict[tid].logprob if tid in lp_dict else -20.0 
            for letter, tid in LETTER_TOKEN_IDS.items()
        }
        sorted_lps = sorted(choice_lps.items(), key=lambda x: x[1], reverse=True)
        best_letter, best_lp = sorted_lps[0]
        second_letter, second_lp = sorted_lps[1]
        log_diff = float(best_lp - second_lp)

        # Extraction for reasoning
        reason_text = reason_outputs[i].outputs[0].text
        reason_pred = extract_answer_letter(reason_text)
        reason_tokens = len(reason_outputs[i].outputs[0].token_ids)

        results.append({
            "id": i,
            "log_diff": log_diff,
            "probe_correct": bool(best_letter == correct),
            "reason_correct": bool(reason_pred == correct),
            "probe_tokens": 1,
            "reason_tokens": int(reason_tokens)
        })

    _atomic_write_json(OUTPUT_FILE, results)
    print(f"Collected data for {len(results)} samples.")

    # 4. RUN GRID SEARCH (Same logic as original script)
    thresholds = np.linspace(0, 10, 101) # Log-diff usually ranges 0-10
    analysis = []
    for T in thresholds:
        total_correct = 0
        total_tokens = 0
        esc_count = 0
        for row in results:
            if row['log_diff'] < T: # Escalate
                total_correct += 1 if row['reason_correct'] else 0
                total_tokens += (row['probe_tokens'] + row['reason_tokens'])
                esc_count += 1
            else: # Trust probe
                total_correct += 1 if row['probe_correct'] else 0
                total_tokens += row['probe_tokens']
        
        analysis.append({
            "threshold": round(float(T), 2),
            "accuracy": round(total_correct / len(results), 4),
            "avg_tokens": round(total_tokens / len(results), 2),
            "escalation_rate": round(esc_count / len(results), 2)
        })

    df = pd.DataFrame(analysis)
    df.to_csv(GRID_SEARCH_CSV, index=False)
    print(f"Max Accuracy: {df['accuracy'].max()} at threshold {df.loc[df['accuracy'].idxmax(), 'threshold']}")

if __name__ == "__main__":
    main()