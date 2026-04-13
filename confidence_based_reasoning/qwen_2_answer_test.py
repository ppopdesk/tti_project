import requests
from datasets import load_dataset
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
LOG_DIFF_THRESHOLD = 22.53

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

def run_gated_test():
    ds = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="test", streaming=True)
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i, entry in enumerate(ds):
        if i > 5: break
        q_text = entry['sent1']
        opts_list = [f"{idx_to_letter[j]}) {entry[f'ending{j}']}" for j in range(4)]
        opts_text = "\n".join(opts_list)
        correct_answer = idx_to_letter[entry['label']]

        top_choice, second_choice, log_diff, all_candidates, probe_output_tokens = get_no_reasoning_answer(q_text, opts_text)

        print(f"Q{i+1} Analysis:")
        print(f"Top choice: {top_choice}")
        print(f"  Initial Log-Diff: {log_diff:.4f}")
        print(f"  Probe output tokens: {probe_output_tokens}")
        print(f"  All candidates: {all_candidates}")

        total_output_tokens = probe_output_tokens

        # if the top two answers are the same, don't escalate to reasoning
        if top_choice == second_choice:
            print(f"  Top 2 choices are the same. Sticking with Top-1")
            final_pred = top_choice
        elif log_diff < LOG_DIFF_THRESHOLD:
            print(f"  Confidence low (< {LOG_DIFF_THRESHOLD}). Invoking Reasoning Mode...")
            reasoning_output, reason_output_tokens = get_reasoned_answer(q_text, opts_text)
            total_output_tokens += reason_output_tokens
            
            final_pred = reasoning_output
            print(f"  Reasoning Result: {final_pred}")
            print(f"  Reasoning output tokens: {reason_output_tokens}")
        else:
            print(f"    High confidence. Sticking with Top-1.")
            final_pred = top_choice

        print(f"  Total output tokens: {total_output_tokens}")

        status = "CORRECT" if final_pred == correct_answer else f"WRONG (Target: {correct_answer})"
        print(f"  FINAL RESULT: {final_pred} | {status}\n" + "-"*40)

if __name__ == "__main__":
    run_gated_test()
