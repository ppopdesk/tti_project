#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_NAME = "ShivaniiKum/qwen-medreason-finetuned"

DIRECT_PREFIX = "<ANSWER>\n"
SHORT_PREFIX = "<COT>\n"
LONG_PREFIX = "<LONG_COT>\n"

# Pass 1 thresholds
LOG_DIFF_THRESHOLD = 3.75
ENTROPY_THRESHOLD = 0.350

# Pass 2 settings
SHORT_COT_SEEDS = [11, 22, 33]
SHORT_COT_K = 3
SHORT_COT_TEMPERATURE = 0.4
SHORT_COT_MAX_TOKENS = 128
SHORT_COT_TOP_P = 0.95

# Pass 3 settings
LONG_COT_TEMPERATURE = 0.0
LONG_COT_MAX_TOKENS = 2048
LONG_COT_TOP_P = 1.0

# Qwen answer-token ids
LETTER_TOKEN_IDS = {
    "A": 32,
    "B": 33,
    "C": 34,
    "D": 35,
    "E": 36,
}


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e
    return records


def format_question(example: dict) -> str:
    question = example["question"].strip()
    options = example["options"]

    option_lines = []
    for key in ["A", "B", "C", "D", "E", "F"]:
        if key in options:
            option_lines.append(f"{key}. {str(options[key]).strip()}")

    return f"{question}\nAnswer Choices:\n" + "\n".join(option_lines)


def build_prompt(tokenizer, question_text: str, prefix: str) -> str:
    messages = [
        {"role": "user", "content": question_text},
        {"role": "assistant", "content": prefix},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )


def extract_answer_letter(text: str) -> Optional[str]:
    match = re.search(r"<ANSWER>\s*(?:\\boxed\{)?([A-E])\}?\s*</ANSWER>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:answer is|answer:|final answer[:\s]+)\s*([A-E])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([A-E])\b", text.upper())
    return matches[-1] if matches else None


def unanimous_vote(letters: List[Optional[str]]) -> Optional[str]:
    clean = [x for x in letters if x is not None]
    if len(clean) != len(letters):
        return None
    first = clean[0]
    if all(x == first for x in clean):
        return first
    return None


def make_llm(model_name: str, gpu_memory_utilization: float, max_model_len: int, dtype: str) -> LLM:
    return LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
    )


def get_valid_letters(example: dict) -> List[str]:
    return [k for k in ["A", "B", "C", "D", "E", "F"] if k in example["options"]]


def compute_top2_entropy_and_logdiff(
    logprobs_dict: dict,
    valid_letters: List[str],
) -> Tuple[Optional[str], float, float, Dict[str, float], List[Tuple[str, float]]]:
    raw_logprobs = {}
    for letter in valid_letters:
        token_id = LETTER_TOKEN_IDS.get(letter)
        if token_id is None:
            continue
        if token_id in logprobs_dict:
            raw_logprobs[letter] = float(logprobs_dict[token_id].logprob)
        else:
            raw_logprobs[letter] = -20.0

    if not raw_logprobs:
        return None, 0.0, 0.0, {}, []

    sorted_items = sorted(raw_logprobs.items(), key=lambda x: x[1], reverse=True)
    pred_letter = sorted_items[0][0]

    if len(sorted_items) == 1:
        return pred_letter, 0.0, 99.0, raw_logprobs, sorted_items

    top1_letter, top1_lp = sorted_items[0]
    top2_letter, top2_lp = sorted_items[1]

    pair_lps = np.array([top1_lp, top2_lp], dtype=np.float64)
    pair_lps = pair_lps - np.max(pair_lps)
    pair_probs = np.exp(pair_lps)
    pair_probs = pair_probs / np.sum(pair_probs)

    entropy = float(-np.sum(pair_probs * np.log(pair_probs + 1e-12)))
    log_diff = float(top1_lp - top2_lp)

    return pred_letter, entropy, log_diff, raw_logprobs, sorted_items


def pass_1_direct(
    llm: LLM,
    tokenizer,
    example: dict,
    max_tokens: int = 16,
) -> dict:
    question_text = format_question(example)
    prompt = build_prompt(tokenizer, question_text, DIRECT_PREFIX)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=20,
    )

    output = llm.generate([prompt], sampling_params)[0]
    generated_text = output.outputs[0].text
    full_output = DIRECT_PREFIX + generated_text

    valid_letters = get_valid_letters(example)
    first_step_logprobs = output.outputs[0].logprobs[0]

    pred_letter, entropy, log_diff, choice_logprobs, sorted_items = compute_top2_entropy_and_logdiff(
        first_step_logprobs,
        valid_letters,
    )

    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
    generated_tokens = len(output.outputs[0].token_ids)
    total_tokens = prompt_tokens + generated_tokens

    extracted_letter = extract_answer_letter(full_output)
    if extracted_letter is None:
        extracted_letter = pred_letter

    return {
        "predicted_answer_idx": extracted_letter,
        "fast_probe_letter": pred_letter,
        "entropy": entropy,
        "log_diff": log_diff,
        "choice_logprobs": choice_logprobs,
        "sorted_choice_logprobs": [{"letter": k, "logprob": v} for k, v in sorted_items],
        "raw_output": full_output,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
        },
    }


def pass_2_short_cot(
    llm: LLM,
    tokenizer,
    example: dict,
    seeds: List[int],
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> dict:
    question_text = format_question(example)
    prompt = build_prompt(tokenizer, question_text, SHORT_PREFIX)

    letters = []
    raw_outputs = []
    trial_token_usage = []

    for seed in seeds:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

        output = llm.generate([prompt], sampling_params)[0]
        generated_text = output.outputs[0].text
        full_output = SHORT_PREFIX + generated_text
        pred_letter = extract_answer_letter(full_output)

        prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
        generated_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + generated_tokens

        letters.append(pred_letter)
        raw_outputs.append(full_output)
        trial_token_usage.append({
            "seed": seed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
        })

    return {
        "letters": letters,
        "raw_outputs": raw_outputs,
        "trial_token_usage": trial_token_usage,
        "prompt_tokens": sum(x["prompt_tokens"] for x in trial_token_usage),
        "generated_tokens": sum(x["generated_tokens"] for x in trial_token_usage),
        "total_tokens": sum(x["total_tokens"] for x in trial_token_usage),
    }


def pass_3_long_cot(
    llm: LLM,
    tokenizer,
    example: dict,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> dict:
    question_text = format_question(example)
    prompt = build_prompt(tokenizer, question_text, LONG_PREFIX)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    output = llm.generate([prompt], sampling_params)[0]
    generated_text = output.outputs[0].text
    full_output = LONG_PREFIX + generated_text
    pred_letter = extract_answer_letter(full_output)

    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
    generated_tokens = len(output.outputs[0].token_ids)
    total_tokens = prompt_tokens + generated_tokens

    return {
        "predicted_answer_idx": pred_letter,
        "raw_output": full_output,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
        },
    }


def run_pipeline(
    llm: LLM,
    tokenizer,
    example: dict,
) -> dict:
    gold_answer = str(example["answer_idx"]).strip().upper()

    # Pass 1
    p1 = pass_1_direct(llm, tokenizer, example)
    total_prompt_tokens = p1["token_usage"]["prompt_tokens"]
    total_generated_tokens = p1["token_usage"]["generated_tokens"]
    total_tokens = p1["token_usage"]["total_tokens"]

    direct_gate_pass = (
        p1["predicted_answer_idx"] is not None
        and p1["log_diff"] >= LOG_DIFF_THRESHOLD
        and p1["entropy"] <= ENTROPY_THRESHOLD
    )

    if direct_gate_pass:
        final_answer = p1["predicted_answer_idx"]
        gate_reached = 1
        answer_source = "direct_fast_pass"
        short_pass = None
        long_pass = None
    else:
        # Pass 2
        p2 = pass_2_short_cot(
            llm=llm,
            tokenizer=tokenizer,
            example=example,
            seeds=SHORT_COT_SEEDS[:SHORT_COT_K],
            temperature=SHORT_COT_TEMPERATURE,
            max_tokens=SHORT_COT_MAX_TOKENS,
            top_p=SHORT_COT_TOP_P,
        )
        total_prompt_tokens += p2["prompt_tokens"]
        total_generated_tokens += p2["generated_tokens"]
        total_tokens += p2["total_tokens"]

        unanimous = unanimous_vote(p2["letters"])
        short_pass = p2

        if unanimous is not None:
            final_answer = unanimous
            gate_reached = 2
            answer_source = "short_cot_unanimous"
            long_pass = None
        else:
            # Pass 3
            p3 = pass_3_long_cot(
                llm=llm,
                tokenizer=tokenizer,
                example=example,
                temperature=LONG_COT_TEMPERATURE,
                max_tokens=LONG_COT_MAX_TOKENS,
                top_p=LONG_COT_TOP_P,
            )
            total_prompt_tokens += p3["token_usage"]["prompt_tokens"]
            total_generated_tokens += p3["token_usage"]["generated_tokens"]
            total_tokens += p3["token_usage"]["total_tokens"]

            final_answer = p3["predicted_answer_idx"]
            gate_reached = 3
            answer_source = "long_cot_escalation"
            long_pass = p3

    return {
        "question": example["question"],
        "answer": example["answer"],
        "answer_idx": gold_answer,
        "final_answer": final_answer,
        "is_correct": final_answer == gold_answer,
        "gate_reached": gate_reached,
        "answer_source": answer_source,
        "pass_1": p1,
        "pass_2": short_pass,
        "pass_3": long_pass,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_generated_tokens,
            "total_tokens": total_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="3-stage vLLM pipeline for MedQA: direct -> think-on-disagreement -> long-cot"
    )
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--data_path", type=str, default="medqa_test.jsonl")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_examples", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading dataset from {args.data_path}")
    dataset = load_jsonl(args.data_path)
    if args.num_examples > 0:
        dataset = dataset[:args.num_examples]
    print(f"Loaded {len(dataset)} examples")

    print("Initializing vLLM...")
    llm = make_llm(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    results = []
    correct = 0
    pass1_count = 0
    pass2_count = 0
    pass3_count = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_tokens = 0

    for idx, example in enumerate(dataset):
        result = run_pipeline(llm, tokenizer, example)
        result["idx"] = idx
        results.append(result)

        correct += int(result["is_correct"])
        pass1_count += int(result["gate_reached"] == 1)
        pass2_count += int(result["gate_reached"] == 2)
        pass3_count += int(result["gate_reached"] == 3)

        total_prompt_tokens += result["token_usage"]["prompt_tokens"]
        total_generated_tokens += result["token_usage"]["generated_tokens"]
        total_tokens += result["token_usage"]["total_tokens"]

        print(
            f"[{idx + 1}/{len(dataset)}] "
            f"pred={result['final_answer']} "
            f"gold={result['answer_idx']} "
            f"correct={result['is_correct']} "
            f"gate={result['gate_reached']}"
        )

    total = len(results)
    summary = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "pass_1_direct_count": pass1_count,
        "pass_2_short_cot_count": pass2_count,
        "pass_3_long_cot_count": pass3_count,
        "pass_1_direct_rate": pass1_count / total if total else 0.0,
        "pass_2_short_cot_rate": pass2_count / total if total else 0.0,
        "pass_3_long_cot_rate": pass3_count / total if total else 0.0,
        "escalation_rate_to_short_cot_or_beyond": (pass2_count + pass3_count) / total if total else 0.0,
        "escalation_rate_to_long_cot": pass3_count / total if total else 0.0,
        "avg_prompt_tokens": total_prompt_tokens / total if total else 0.0,
        "avg_generated_tokens": total_generated_tokens / total if total else 0.0,
        "avg_total_tokens": total_tokens / total if total else 0.0,
        "log_diff_threshold": LOG_DIFF_THRESHOLD,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "short_cot_k": SHORT_COT_K,
        "short_cot_seeds": SHORT_COT_SEEDS[:SHORT_COT_K],
        "short_cot_temperature": SHORT_COT_TEMPERATURE,
        "short_cot_max_tokens": SHORT_COT_MAX_TOKENS,
        "long_cot_temperature": LONG_COT_TEMPERATURE,
        "long_cot_max_tokens": LONG_COT_MAX_TOKENS,
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved results to:", args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
