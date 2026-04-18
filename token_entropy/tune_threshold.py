"""
Threshold Tuning for Token Entropy
------------------------------------
Runs reasoning on ALL questions (not just wrong ones) for accurate threshold sweep.
This is the correct version that matches how the inference script calculates accuracy.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from token_entropy import (
    load_medqa_parquet,
    format_fast_prompt,
    format_reasoning_prompt,
    extract_answer_letter,
    compute_entropy_from_logprobs,
)


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
    print(f"Loaded {len(examples)} questions.")

    # Step 1: Fast pass on ALL questions
    print("\nRunning fast passes on all questions...")
    fast_prompts = [format_fast_prompt(ex["prompt_messages"], tokenizer) for ex in examples]
    fast_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    fast_outputs = llm.generate(fast_prompts, fast_params)

    fast_data = []
    for i, output in enumerate(fast_outputs):
        logprobs_dict = output.outputs[0].logprobs[0]
        pred, entropy, probs = compute_entropy_from_logprobs(logprobs_dict)
        fast_correct = pred == examples[i]["answer"]
        fast_data.append({"pred": pred, "entropy": entropy, "fast_correct": fast_correct})

    print(f"Fast pass done. {sum(1 for d in fast_data if not d['fast_correct'])} questions wrong on fast pass.")

    # Step 2: Reasoning pass on ALL questions
    print(f"\nRunning reasoning on ALL {len(examples)} questions...")
    reasoning_prompts = [format_reasoning_prompt(ex["prompt_messages"], tokenizer) for ex in examples]
    reasoning_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=2048)
    reasoning_outputs = llm.generate(reasoning_prompts, reasoning_params)

    reasoning_data = []
    for i, output in enumerate(reasoning_outputs):
        text = output.outputs[0].text
        ans = extract_answer_letter(text)
        reasoning_data.append({
            "answer": ans,
            "reasoning_correct": ans == examples[i]["answer"],
            "n_tokens": len(output.outputs[0].token_ids),
        })
    print("Reasoning passes done.")

    # Step 3: Sweep thresholds using TRUE reasoning answers
    thresholds = np.round(np.arange(0.0, 1.45, 0.025), 3).tolist()
    N = len(fast_data)
    print(f"\nSweeping {len(thresholds)} thresholds...")

    sweep_results = []
    for thresh in thresholds:
        n_correct = 0
        n_escalated = 0
        n_wrong_no_escalate = 0
        total_tokens = 0

        for i, d in enumerate(fast_data):
            if d["entropy"] > thresh:
                # Escalate to reasoning
                n_escalated += 1
                total_tokens += reasoning_data[i]["n_tokens"]
                if reasoning_data[i]["reasoning_correct"]:
                    n_correct += 1
            else:
                # Use fast answer
                total_tokens += 1
                if d["fast_correct"]:
                    n_correct += 1
                else:
                    n_wrong_no_escalate += 1

        E, W = n_escalated, n_wrong_no_escalate
        penalty = (1 / 3) * (E / N) + (W / N)
        sweep_results.append({
            "threshold": thresh,
            "accuracy": round(n_correct / N * 100, 2),
            "escalation_rate": round(E / N * 100, 2),
            "wrong_without_escalation_rate": round(W / N * 100, 2),
            "penalty": round(penalty, 4),
            "avg_tokens": round(total_tokens / N, 1),
        })

    sweep_results_by_penalty = sorted(sweep_results, key=lambda x: x["penalty"])

    print("\n=== TOP 10 THRESHOLDS BY PENALTY SCORE ===")
    print(f"{'Rank':<5} {'Threshold':<12} {'Accuracy':<12} {'Esc Rate':<12} {'Wrong/NoEsc':<14} {'Penalty':<10} {'Avg Tokens'}")
    print("-" * 75)
    for rank, r in enumerate(sweep_results_by_penalty[:10], 1):
        print(f"{rank:<5} {r['threshold']:<12.3f} {r['accuracy']:<12.2f} {r['escalation_rate']:<12.2f} {r['wrong_without_escalation_rate']:<14.2f} {r['penalty']:<10.4f} {r['avg_tokens']:.1f}")

    best = sweep_results_by_penalty[0]
    print(f"\n✓ Best threshold: {best['threshold']}  (penalty={best['penalty']}, accuracy={best['accuracy']}%, avg_tokens={best['avg_tokens']})")
    print(f"  --> Put this in run_token_entropy.sh as THRESHOLD={best['threshold']}")

    print("\n=== FULL SWEEP (all thresholds) ===")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Esc Rate':<12} {'Wrong/NoEsc':<14} {'Penalty':<10} {'Avg Tokens'}")
    print("-" * 75)
    for r in sweep_results:
        print(f"{r['threshold']:<12.3f} {r['accuracy']:<12.2f} {r['escalation_rate']:<12.2f} {r['wrong_without_escalation_rate']:<14.2f} {r['penalty']:<10.4f} {r['avg_tokens']:.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "threshold_sweep.json")
    with open(out_path, "w") as f:
        json.dump({"sweep": sweep_results, "best": best}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path",     type=str, required=True)
    parser.add_argument("--num_questions", type=int, default=-1)
    parser.add_argument("--output_dir",    type=str, default="./tuning_results")
    args = parser.parse_args()
    main(args)
