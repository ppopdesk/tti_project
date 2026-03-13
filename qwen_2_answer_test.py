import requests
import json
import math
from typing import List, Dict
from datasets import load_dataset
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct"

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
    token_info = response['logprobs'][0]
    candidates = token_info['top_logprobs']
    output_tokens = response.get('eval_count', 0)
    
    # Filter for A, B, C, D
    valid = [c for c in candidates if c['token'].strip().upper().replace(")", "") in ['A', 'B', 'C', 'D']]
    
    if len(valid) >= 2:
        return valid[0]['token'].strip().upper(), (valid[0]['logprob'] - valid[1]['logprob']), candidates, output_tokens
    elif len(valid) == 1:
        return valid[0]['token'].strip().upper(), 99.0, candidates, output_tokens  # Extremely high confidence
    return None, 0, candidates, output_tokens

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

    full_text = response['response']
    output_tokens = response.get('eval_count', 0)
    
    # Improved Extraction Logic: Look for "Final Answer: X" or just the last capitalized letter
    match = re.search(r"Final Answer:\s*([A-D])", full_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), output_tokens
    
    # Fallback: find the last occurrence of A, B, C, or D in the text
    backup_match = re.findall(r"\b([A-D])\b", full_text)
    final_answer = backup_match[-1].upper() if backup_match else "A"
    return final_answer, output_tokens

def run_gated_test():
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="test", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i, entry in enumerate(ds):
	# Format basics
        q_text = entry['sent1']
        opts_list = [f"{idx_to_letter[j]}) {entry[f'ending{j}']}" for j in range(4)]
        opts_text = "\n".join(opts_list)
        correct_answer = idx_to_letter[entry['label']]

        # 1. PROBE: Check confidence
        probe_prompt = f"Question: {q_text}\nOptions:\n{opts_text}\nAnswer (Letter only):"
        top_choice, log_diff, all_candidates, probe_output_tokens = get_logprobs(probe_prompt)

        print(f"Q{i+1} Analysis:")
        print(f"Top choice: {top_choice}")
        print(f"  Initial Log-Diff: {log_diff:.4f}")
        print(f"  Probe output tokens: {probe_output_tokens}")

        # 2. GATE: Decide if we need reasoning
        total_output_tokens = probe_output_tokens
        if log_diff < 22.53:
            print(f"  ⚠️ Confidence low (< 10). Invoking Reasoning Mode...")
            reasoning_output, reason_output_tokens = get_reasoned_answer(q_text, opts_text)
            total_output_tokens += reason_output_tokens
            
            # reasoning_output is already the extracted letter from get_reasoned_answer
            final_pred = reasoning_output
            print(f"  Reasoning Result: {final_pred}")
            print(f"  Reasoning output tokens: {reason_output_tokens}")
        else:
            print(f"  ✅ High confidence. Sticking with Top-1.")
            final_pred = top_choice

        print(f"  Total output tokens: {total_output_tokens}")

        status = "CORRECT" if final_pred == correct_answer else f"WRONG (Target: {correct_answer})"
        print(f"  FINAL RESULT: {final_pred} | {status}\n" + "-"*40)

if __name__ == "__main__":
    run_gated_test()
