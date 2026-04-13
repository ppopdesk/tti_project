import requests
from datasets import load_dataset
import re
import json
import numpy as np
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "vijayavp/medreason-qwen25-shortcot-exp2:latest"
VAL_SIZE = 300
OUTPUT_FILE = "medqa_calibration_data.json"

def get_no_reasoning_answer(question, options_text):
    prompt = (
        f"Question: {question}\nOptions:\n{options_text}\n\n"
        "Provide the letter corresponding to the correct final answer to this question. "
        "Your output should only be the letter of your chosen output choice, nothing else."
    )

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
    reasoning_prompt = (
        f"Question: {question}\nOptions:\n{options_text}\n\n"
        "Think step-by-step about the clinical presentation, the differential diagnosis, "
        "and then provide the final answer letter at the end as 'Final Answer: [Letter]'"
    )
    
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
    
    match = re.search(r"Final Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), output_tokens
    else:    
        final_answer = None
        return final_answer, output_tokens

def collect_data():
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="test", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    results = []

    for i, entry in enumerate(ds):
        if i >= VAL_SIZE: break
        
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
        if i % 10 == 0: print(f"Processed {i}/{VAL_SIZE}...")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f)
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
    try:
        with open(OUTPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = collect_data()
    
    df = run_grid_search(data)
    # Save the full search results to a CSV file
    df.to_csv("medqa_threshold_grid_search.csv", index=False)
    print(f"\nGrid search results exported to 'medqa_threshold_grid_search.csv'")