import requests
import json
import math
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# --- CONFIG ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
OUTPUT_FILE = "medqa_val_probe_results.csv"

def get_margin_metrics(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "logprobs": True,
        "top_logprobs": 5
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload).json()
        ti = res['logprobs'][0]
        cands = ti['top_logprobs']

        valid = []
        seen_letters = set()
        for c in cands:
            letter = c['token'].strip().upper().replace(")", "")
            if letter in ['A', 'B', 'C', 'D'] and letter not in seen_letters:
                valid.append({"letter": letter, "lp": c['logprob']})
                seen_letters.add(letter)

        # Extract Top 1 and Top 2
        if len(valid) >= 2:
            top_1, top_2 = valid[0], valid[1]
            return top_1['letter'], top_1['lp'], top_2['letter'], top_2['lp'], (top_1['lp'] - top_2['lp'])
        elif len(valid) == 1:
            return valid[0]['letter'], valid[0]['lp'], "N/A", -20.0, 20.0

        return "N/A", 0.0, "N/A", 0.0, 0.0
    except Exception as e:
        print(e)
        return "ERROR", 0.0, "ERROR", 0.0, 0.0

# --- DATA COLLECTION ---
ds_val = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="validation")
results = []

print(f"🚀 Starting Probe on 1.27k Validation Questions...")

for i, entry in enumerate(tqdm(ds_val)):
    if i > 0:
        break
    correct_answer = {0:'A', 1:'B', 2:'C', 3:'D'}[entry['label']]
    opts = "\:n".join([f"{k}) {entry[f'ending{j}']}" for j, k in enumerate('ABCD')])
    prompt = f"You are answering a multiple-choice question.\n Return ONLY the letter of the correct option (A, B, C, D, ...).\n\n Question:\n{entry['sent1']}\n\nOptions:\n{opts}\n\nAnswer:"

    pred, lp1, pred2, lp2, margin = get_margin_metrics(prompt)

    res = {
        "index": i,
        "margin": margin,
        "top_1_letter": pred,
        "top_1_lp": lp1,
        "top_2_letter": pred2,
        "top_2_lp": lp2,
        "ground_truth": correct_answer,
        "is_correct": int(pred == correct_answer)
    }
    results.append(res)

    # Save checkpoint every 100 iterations
    if i % 100 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Data Collection Complete! File: {OUTPUT_FILE}")