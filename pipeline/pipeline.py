import sys
from pathlib import Path

# Repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests
import re
import json
import os
import numpy as np
import random
from datasets import load_dataset
from prompts import (
    NO_REASONING_PROMPT, 
    SHORT_REASONING_PROMPT, 
    LONG_REASONING_PROMPT,
    NO_REASONING_TOKEN_LIMIT,
    SHORT_REASONING_TOKEN_LIMIT,
    LOG_DIFF_THRESHOLD,
    ENTROPY_THRESHOLD,
    SHORT_COT_TEMPERATURE,
    SHORT_COT_K,
    LONG_COT_TEMPERATURE,
)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
VAL_SIZE = 300

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = _SCRIPT_DIR / "medqa_pipeline_results.json"

def calculate_entropy(top_logprobs):
    probs = np.exp([c['logprob'] for c in top_logprobs])
    probs = probs / np.sum(probs)  # Normalize
    return -float(np.sum(probs * np.log(probs + 1e-12)))

def extract_tag_answer(text):
    match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", text, re.IGNORECASE)
    return match.group(1).upper() if match else None

def call_ollama(prompt, max_tokens, temperature=0.0, logprobs=False, seed=None):
    num_logprobs = 0
    if logprobs:
        num_logprobs = 5

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
        },
        "logprobs": logprobs,
        "top_logprobs": num_logprobs
    }
    response = requests.post(OLLAMA_URL, json=payload).json()
    return response

def pass_1(question, options_text):
    """
    Returns a tuple of (pred, entropy, logdiff, total_tokens)
    """

    # makes the ollama call
    p1_prompt = NO_REASONING_PROMPT.format(question=question, options_text=options_text)
    res1 = call_ollama(p1_prompt, NO_REASONING_TOKEN_LIMIT, temperature=0, logprobs=True)
    total_tokens = res1.get('eval_count', 0)
    
    # extracts necessary fields from the response
    token_info = res1.get('logprobs', [{}])[0]
    candidates = token_info.get('top_logprobs', [])

    # filters for valid A, B, C, D tokens in the candidates
    valid = [c for c in candidates if c['token'].strip().upper().replace(")", "") in ['A', 'B', 'C', 'D']]
    
    # calculates entropy
    log_diff = 0.0
    entropy = 0.0
    if len(valid) > 1:
        entropy = calculate_entropy(valid)
    
    # calculates the log difference and prediction
    pred = None
    if len(valid) >= 2:
        log_diff = valid[0]['logprob'] - valid[1]['logprob']
        pred = valid[0]['token'].strip().upper()
    elif len(valid) == 1:
        log_diff = 99.0
        pred = valid[0]['token'].strip().upper()

    return pred, entropy, log_diff, total_tokens

def pass_2(question, options_text):
    """
    Returns tuple of (list of preds, total_tokens)
    """
    prompt = SHORT_REASONING_PROMPT.format(question=question, options=options_text)
    
    samples = []
    total_tokens = 0
    
    # seed to ensure that the model reruns inference for each pass
    current_seed = random.randint(1, 1000000)
 
    # generate k sample answers with a short reasoning pass
    for i in range(SHORT_COT_K):
        response = call_ollama(prompt, SHORT_REASONING_TOKEN_LIMIT, temperature=SHORT_COT_TEMPERATURE, logprobs=False, seed=current_seed)
        total_tokens += response.get('eval_count', 0)
        ans = extract_tag_answer(response['response'])
        if ans:
            samples.append(ans)
    
    return samples, total_tokens

def pass_3(question, options_text):
    """
    Returns tuple of (pred, total_tokens)
    """
    prompt = LONG_REASONING_PROMPT.format(question=question, options=options_text)
    total_tokens = 0

    response = call_ollama(prompt, LONG_REASONING_TOKEN_LIMIT, temperature=LONG_COT_TEMPERATURE, logprobs=False)
    total_tokens += response.get('eval_count', 0)
    ans = extract_tag_answer(response['response'])
    
    if ans:
        return ans, total_tokens
    else:
        return None, total_tokens

def run_pipeline(question, options_text):
    """
    Returns tuple of (pred, pass_type, total_tokens)
    """
    total_tokens = 0
    # Pass 1: no reasoning
    print("    Pass 1: No Reasoning")
    pred, entropy, log_diff, p1_tokens = pass_1(question, options_text)
    print(f"    Pred: {pred}")
    total_tokens += p1_tokens

    # Gate 1 Logic
    if pred and log_diff >= LOG_DIFF_THRESHOLD and entropy <= ENTROPY_THRESHOLD:
        return pred, 1, total_tokens

    # Pass 2: Short COT pass
    print("    Pass 2: Short COT")
    short_answers, p2_tokens = pass_2(question, options_text)
    print(f"    Short Answers: {short_answers}")
    total_tokens += p2_tokens

    # Gate 2 Logic: Unanimous agreement
    answers_set = set(short_answers)
    if len(answers_set) == 1:
        return next(iter(answers_set)), 2, total_tokens

    # Pass 3: Long COT pass
    print("    Pass 3: Long COT")
    long_answer, p3_tokens = pass_3(question, options_text)
    print(f"    Long Answer: {long_answer}")
    total_tokens += p3_tokens

    # Gate 3 Logic: Unanimous agreement
    if long_answer:
        return long_answer, 3, total_tokens
    else:
        return None, 3, total_tokens

def collect_pipeline_data():
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="test", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    results = []

    for i, entry in enumerate(ds):
        print(f"Question {i+1}")
        q_text = entry['sent1']
        opts_text = "\n".join([f"{idx_to_letter[j]}) {entry[f'ending{j}']}" for j in range(4)])
        correct_answer = idx_to_letter[entry['label']]
        pred, pass_type, total_tokens = run_pipeline(q_text, opts_text)
        print("-"*50)
        results.append({
            "id": i,
            "correct": correct_answer,
            "prediction": pred,
            "is_correct": pred == correct_answer,
            "gate_reached": pass_type,
            "total_tokens": total_tokens
        })
        
        # if i % 5 == 0:
        #     print(f"Processed {i}/{VAL_SIZE}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    collect_pipeline_data()