"""
Token Entropy Algorithm for MedQA
-----------------------------------
Uses vLLM (same as Shivani's ARM setup) to run inference on Qwen-2.5-7B-Instruct.
Reads data from parquet format (same as existing MedQA data on Great Lakes).

For each question:
  1. Fast pass: generate 1 token with logprobs, extract probs for A/B/C/D/E
  2. Compute entropy H = -sum(p * log(p)) over A/B/C/D/E
  3. If H > threshold --> escalate to full CoT reasoning
  4. Otherwise        --> return fast answer directly
"""

import os
import json
import argparse
import re
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_medqa_parquet(path: str):
    df = pd.read_parquet(path)
    examples = []
    for _, row in df.iterrows():
        answer = str(row['reward_model'].get('ground_truth', '')).strip().upper()
        examples.append({
            "prompt_messages": row['prompt'],
            "answer": answer,
        })
    return examples

def load_medqa_validation(path: str):
    df = pd.read_parquet(path)
    examples = []
    for _, row in df.iterrows():
        answer = str(row['answer_idx']).strip().upper()
        question = row['question']
        options = row['options']
        options_text = "\n".join([f"{opt['key']}. {opt['value']}" for opt in options])
        full_question = f"{question}\nAnswer Choices:\n{options_text}"
        examples.append({
            "prompt_messages": [{"role": "user", "content": full_question}],
            "answer": answer,
        })
    return examples

def format_fast_prompt(prompt_messages, tokenizer):
    user_content = ""
    for msg in prompt_messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "<ANSWER>\n"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

def format_reasoning_prompt(prompt_messages, tokenizer):
    user_content = ""
    for msg in prompt_messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "<COT>\n"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

def extract_answer_letter(text: str) -> str:
    match = re.search(r'<ANSWER>\s*(?:\\boxed\{)?([A-E])\}?\s*</ANSWER>', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r'(?:answer is|answer:|final answer[:\s]+)\s*([ABCDE])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r'\b([ABCDE])\b', text.upper())
    if matches:
        return matches[-1]
    return "A"


def compute_entropy_from_logprobs(logprobs_dict: dict) -> tuple:
    # Token IDs for A-E in Qwen2.5-7B-Instruct tokenizer (verified)
    letters = ["A", "B", "C", "D", "E"]
    letter_token_ids = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}

    raw_logprobs = {}
    for letter in letters:
        token_id = letter_token_ids[letter]
        if token_id in logprobs_dict:
            raw_logprobs[letter] = logprobs_dict[token_id].logprob
        else:
            raw_logprobs[letter] = -20.0

    lp = np.array([raw_logprobs[l] for l in letters])
    lp = lp - np.max(lp)
    probs = np.exp(lp) / np.sum(np.exp(lp))
    #entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    top2_indices = np.argsort(probs)[-2:]
    top2_probs = probs[top2_indices]
    top2_probs = top2_probs / top2_probs.sum()
    entropy = float(-np.sum(top2_probs * np.log(top2_probs + 1e-12)))
    predicted = letters[int(np.argmax(probs))]
    return predicted, entropy, probs.tolist()


def main(args):
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Initializing vLLM with {args.model_name}...")
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )

    print(f"Loading data from {args.data_path}...")
    examples = load_medqa_parquet(args.data_path)
    if args.num_questions > 0:
        examples = examples[: args.num_questions]
    print(f"Loaded {len(examples)} questions. Threshold = {args.threshold}")

    print("\nRunning fast passes...")
    fast_prompts = [format_fast_prompt(ex["prompt_messages"], tokenizer) for ex in examples]
    fast_params = SamplingParams(temperature=0.0, max_tokens=10, logprobs=20)
    fast_outputs = llm.generate(fast_prompts, fast_params)
    print("Fast passes done.")

    needs_reasoning = []
    fast_results = []
    for i, output in enumerate(fast_outputs):
        logprobs_dict = output.outputs[0].logprobs[0]
        pred, entropy, probs = compute_entropy_from_logprobs(logprobs_dict)
        fast_results.append({"pred": pred, "entropy": entropy, "probs": probs})
        if entropy > args.threshold:
            needs_reasoning.append(i)

    print(f"Escalating {len(needs_reasoning)}/{len(examples)} ({len(needs_reasoning)/len(examples)*100:.1f}%) to reasoning")

    reasoning_answers = {}
    if needs_reasoning:
        print("\nRunning reasoning passes...")
        reasoning_prompts = [format_reasoning_prompt(examples[i]["prompt_messages"], tokenizer) for i in needs_reasoning]
        reasoning_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=2048)
        reasoning_outputs = llm.generate(reasoning_prompts, reasoning_params)
        for idx, output in zip(needs_reasoning, reasoning_outputs):
            text = output.outputs[0].text
            reasoning_answers[idx] = {
                "answer": extract_answer_letter(text),
                "text": text,
                "n_tokens": len(output.outputs[0].token_ids),
            }
        print("Reasoning passes done.")

    results = []
    n_correct = 0
    n_wrong_no_escalate = 0
    total_output_tokens = 0

    for i, ex in enumerate(examples):
        used_reasoning = i in needs_reasoning
        final_ans = reasoning_answers[i]["answer"] if used_reasoning else fast_results[i]["pred"]
        total_output_tokens += reasoning_answers[i]["n_tokens"] if used_reasoning else 1
        is_correct = final_ans == ex["answer"]
        if is_correct:
            n_correct += 1
        if not used_reasoning and not is_correct:
            n_wrong_no_escalate += 1
        results.append({
            "idx": i,
            "correct": ex["answer"],
            "fast_answer": fast_results[i]["pred"],
            "final_answer": final_ans,
            "entropy": fast_results[i]["entropy"],
            "probs": fast_results[i]["probs"],
            "used_reasoning": used_reasoning,
            "is_correct": is_correct,
            "reasoning_text": reasoning_answers[i]["text"] if used_reasoning else "",
        })

    N = len(results)
    E = len(needs_reasoning)
    W = n_wrong_no_escalate
    penalty = (1 / 3) * (E / N) + (W / N)
    accuracy = n_correct / N * 100
    avg_tokens = total_output_tokens / N

    print("\n=== RESULTS ===")
    print(f"Total questions     : {N}")
    print(f"Accuracy            : {accuracy:.2f}%")
    print(f"Escalation rate     : {E/N*100:.2f}%  ({E}/{N})")
    print(f"Wrong w/o escalation: {W/N*100:.2f}%  ({W}/{N})")
    print(f"Penalty score       : {penalty:.4f}")
    print(f"Avg output tokens   : {avg_tokens:.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"token_entropy_thresh{args.threshold:.3f}.json")
    with open(out_path, "w") as f:
        json.dump({
            "threshold": args.threshold,
            "accuracy": accuracy,
            "escalation_rate": E / N,
            "wrong_without_escalation": W / N,
            "penalty": penalty,
            "avg_output_tokens": avg_tokens,
            "results": results,
        }, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path",     type=str,   required=True)
    parser.add_argument("--threshold",     type=float, default=1.0)
    parser.add_argument("--num_questions", type=int,   default=-1)
    parser.add_argument("--output_dir",    type=str,   default="./results")
    args = parser.parse_args()
    main(args)
