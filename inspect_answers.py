"""
Inspect individual answers from the token entropy pipeline.
Runs a small sample of questions and prints:
- The question
- The correct answer
- The fast pass answer + entropy
- The reasoning answer (if entropy > threshold)
- Whether each was correct

Usage:
    python3 inspect_answers.py --num_questions 20
"""

import argparse
import re
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
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )

    print(f"Loading data...")
    examples = load_medqa_parquet(args.data_path)
    examples = examples[:args.num_questions]
    print(f"Inspecting {len(examples)} questions. Threshold = {args.threshold}\n")

    # Fast pass
    fast_prompts = [format_fast_prompt(ex["prompt_messages"], tokenizer) for ex in examples]
    fast_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    fast_outputs = llm.generate(fast_prompts, fast_params)

    fast_results = []
    needs_reasoning = []
    for i, output in enumerate(fast_outputs):
        logprobs_dict = output.outputs[0].logprobs[0]
        if i == 0:
            print("DEBUG top tokens for Q1:", list(logprobs_dict.keys())[:10])
        pred, entropy, probs = compute_entropy_from_logprobs(logprobs_dict)
        fast_results.append({"pred": pred, "entropy": entropy, "probs": probs})
        if entropy > args.threshold:
            needs_reasoning.append(i)

    # Reasoning pass
    reasoning_answers = {}
    if needs_reasoning:
        reasoning_prompts = [format_reasoning_prompt(examples[i]["prompt_messages"], tokenizer) for i in needs_reasoning]
        reasoning_params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=2048)
        reasoning_outputs = llm.generate(reasoning_prompts, reasoning_params)
        for idx, output in zip(needs_reasoning, reasoning_outputs):
            text = output.outputs[0].text
            reasoning_answers[idx] = {
                "answer": extract_answer_letter(text),
                "text": text,
            }

    # Print results
    print("=" * 80)
    n_correct = 0
    for i, ex in enumerate(examples):
        correct = ex["answer"]
        fast_pred = fast_results[i]["pred"]
        entropy = fast_results[i]["entropy"]
        probs = fast_results[i]["probs"]
        used_reasoning = i in needs_reasoning

        if used_reasoning:
            final_ans = reasoning_answers[i]["answer"]
            reasoning_text = reasoning_answers[i]["text"][:200]
        else:
            final_ans = fast_pred
            reasoning_text = None

        is_correct = final_ans == correct
        if is_correct:
            n_correct += 1

        status = "✓" if is_correct else "✗"

        for msg in ex["prompt_messages"]:
            if msg["role"] == "user":
                question_preview = msg["content"][:150].replace("\n", " ")
                break

        print(f"[{i+1:3d}] {status}  Correct={correct}  FastPred={fast_pred}  FinalAns={final_ans}  Entropy={entropy:.4f}  Reasoned={used_reasoning}")
        print(f"       Q: {question_preview}...")
        print(f"       Probs: A={probs[0]:.3f} B={probs[1]:.3f} C={probs[2]:.3f} D={probs[3]:.3f} E={probs[4]:.3f}")
        if used_reasoning and reasoning_text:
            print(f"       Reasoning preview: {reasoning_text}...")
        print()

    print("=" * 80)
    print(f"Accuracy on this sample: {n_correct}/{len(examples)} = {n_correct/len(examples)*100:.1f}%")
    print(f"Escalated to reasoning:  {len(needs_reasoning)}/{len(examples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path",     type=str,   default="/home/shivanii/ARM/data/medqa/test.parquet")
    parser.add_argument("--threshold",     type=float, default=1.0)
    parser.add_argument("--num_questions", type=int,   default=20)
    args = parser.parse_args()
    main(args)
