import requests
import json
import math
from typing import List, Dict
from datasets import load_dataset
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"

def get_logprobs(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "logprobs": True,
        "top_logprobs": 5
    }
    response = requests.post(OLLAMA_URL, json=payload).json()
    # print(f"First response: {response}")
    token_info = response['logprobs'][0]
    candidates = token_info['top_logprobs']
    
    # Filter for A, B, C, D
    valid = [c for c in candidates if c['token'].strip().upper().replace(")", "") in ['A', 'B', 'C', 'D']]
    
    if len(valid) >= 2:
        return valid[0]['token'].strip().upper(), (valid[0]['logprob'] - valid[1]['logprob']), candidates
    elif len(valid) == 1:
        return valid[0]['token'].strip().upper(), 99.0, candidates # Extremely high confidence
    return None, 0, candidates

def get_reasoned_answer(question, options_text):
    # Fixed the variable name from 'opts' to 'options_text'
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
    # print(f"Reasoning response: {response}")

    full_text = response['response']
    
    # Improved Extraction Logic: Look for "Final Answer: X" or just the last capitalized letter
    match = re.search(r"Final Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: find the last occurrence of A, B, C, or D in the text
    backup_match = re.findall(r"\b([A-D])\b", full_text)
    return backup_match[-1].upper() if backup_match else "A" # Default fallback

def run_gated_test():
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="test", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i, entry in enumerate(ds):
        if i > 5: break
        # Format basics
        q_text = entry['sent1']
        opts_list = [f"{idx_to_letter[j]}) {entry[f'ending{j}']}" for j in range(4)]
        opts_text = "\n".join(opts_list)
        correct_answer = idx_to_letter[entry['label']]

        # 1. PROBE: Check confidence
        probe_prompt = f"Question: {q_text}\nOptions:\n{opts_text}\nAnswer (Letter only):"
        top_choice, log_diff, all_candidates = get_logprobs(probe_prompt)

        print(f"Q{i+1} Analysis:")
        print(f"Top choice: {top_choice}")
        print(f"  Initial Log-Diff: {log_diff:.4f}")

        # 2. GATE: Decide if we need reasoning
        if log_diff < 10:
            print(f"  ⚠️ Confidence low (< 10). Invoking Reasoning Mode...")
            reasoning_output = get_reasoned_answer(q_text, opts_text)
            
            # Simple extraction: look for 'Final Answer: X'
            final_pred = reasoning_output.split("Final Answer:")[-1].strip()[0] if "Final Answer:" in reasoning_output else top_choice
            print(f"  Reasoning Result: {final_pred}")
        else:
            print(f"  ✅ High confidence. Sticking with Top-1.")
            final_pred = top_choice

        status = "CORRECT" if final_pred == correct_answer else f"WRONG (Target: {correct_answer})"
        print(f"  FINAL RESULT: {final_pred} | {status}\n" + "-"*40)

if __name__ == "__main__":
    run_gated_test()