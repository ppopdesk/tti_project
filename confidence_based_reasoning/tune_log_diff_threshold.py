import sys
from pathlib import Path

# Repo root on path: needed when running this file directly (e.g. Slurm) without pip install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
from datasets import load_dataset
import re
import json
import os
import numpy as np
import pandas as pd

from prompts import NO_REASONING_PROMPT, SHORT_REASONING_PROMPT, LONG_REASONING_PROMPT

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7B"
VAL_SIZE = 300
SAVE_EVERY = 10

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("MEDQA_CALIBRATION_DIR", _SCRIPT_DIR)).resolve()
OUTPUT_FILE = OUTPUT_DIR / "medqa_calibration_data.json"
GRID_SEARCH_CSV = OUTPUT_DIR / "medqa_threshold_grid_search.csv"


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)

def extract_tag_answer(text):
    match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", text, re.IGNORECASE)
    return match.group(1).upper() if match else None

def get_no_reasoning_answer(question, options_text):
    prompt = NO_REASONING_PROMPT.format(question=question, options_text=options_text)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "logprobs": True,
        "top_logprobs": 5
    }
    response = requests.post(OLLAMA_URL, json=payload).json()
    token_info = response['logprobs'][0]
    candidates = token_info['top_logprobs']
    output_tokens = response.get('eval_count', 0)
    
    # Filter for A, B, C, D
    valid = [c for c in candidates if c['token'].strip().upper().replace(")", "") in ['A', 'B', 'C', 'D']]
    
    if len(valid) >= 2:
        return valid[0]['token'].strip().upper(), valid[1]['token'].strip().upper(), (valid[0]['logprob'] - valid[1]['logprob']), candidates, output_tokens
    elif len(valid) == 1:
        return valid[0]['token'].strip().upper(), None, 99.0, candidates, output_tokens
    return None, None, 0, candidates, output_tokens

def get_reasoned_answer(question, options_text):
    reasoning_prompt = LONG_REASONING_PROMPT.format(question=question, options=options_text)
    
    payload = {
        "model": MODEL_NAME,
        "prompt": reasoning_prompt,
        "stream": False,
        "options": {"temperature": 0.1} 
    }
    
    response = requests.post(OLLAMA_URL, json=payload).json()

    full_text = response['response']
    output_tokens = response.get('eval_count', 0)
    print(full_text)
    
    answer = extract_tag_answer(full_text)

    if answer:
        return answer, output_tokens
    else:    
        return None, output_tokens

def collect_data():
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="validation", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    results: list = []
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                results = loaded
        except (json.JSONDecodeError, OSError):
            results = []

    next_i = len(results)

    for i, entry in enumerate(ds):
        if i < next_i:
            continue

        q_text = entry['sent1']
        opts_list = [f"{idx_to_letter[j]}) {entry[f'ending{j}']}" for j in range(4)]
        opts_text = "\n".join(opts_list)
        correct = idx_to_letter[entry['label']]

        probe_pred, probe_second, log_diff, all_candidates, probe_tokens = get_no_reasoning_answer(q_text, opts_text)
        reason_pred, reason_tokens = get_reasoned_answer(q_text, opts_text)

        results.append({
            "id": i,
            "log_diff": log_diff,
            "probe_correct": probe_pred == correct,
            "reason_correct": reason_pred == correct,
            "probe_tokens": probe_tokens,
            "reason_tokens": reason_tokens
        })
        n = len(results)
        if n % SAVE_EVERY == 0:
            _atomic_write_json(OUTPUT_FILE, results)
            print(f"Checkpoint: saved {n}/{VAL_SIZE} rows to {OUTPUT_FILE}", flush=True)

    if results:
        _atomic_write_json(OUTPUT_FILE, results)
    print(f"Wrote calibration data to {OUTPUT_FILE}", flush=True)
    return results

def run_grid_search(data):
    thresholds = np.linspace(0, 25, 101)
    analysis = []

    for T in thresholds:
        total_correct = 0
        total_tokens = 0
        escalated_count = 0

        for row in data:
            if row['log_diff'] < T:
                total_correct += 1 if row['reason_correct'] else 0
                total_tokens += (row['probe_tokens'] + row['reason_tokens'])
                escalated_count += 1
            else:
                total_correct += 1 if row['probe_correct'] else 0
                total_tokens += row['probe_tokens']

        acc = total_correct / len(data)
        avg_tokens = total_tokens / len(data)
        
        analysis.append({
            "threshold": round(T, 2),
            "accuracy": round(acc, 4),
            "avg_tokens": round(avg_tokens, 2),
            "escalation_rate": round(escalated_count / len(data), 2)
        })

 
    df = pd.DataFrame(analysis)

    print("\n--- Grid Search Results (Summary) ---")
    print(df[::10].to_string(index=False))
    
    best_row = df.iloc[(df['accuracy'] - df['accuracy'].max()).abs().argsort()[:1]]
    print(f"\nMax Accuracy reached: {df['accuracy'].max()} at Threshold {best_row['threshold'].values[0]}")
        
    return df

if __name__ == "__main__":
    data = []
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
        except (json.JSONDecodeError, OSError):
            data = []

    if len(data) < VAL_SIZE:
        data = collect_data()
    
    df = run_grid_search(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GRID_SEARCH_CSV, index=False)
    print(f"\nGrid search results exported to {GRID_SEARCH_CSV}", flush=True)