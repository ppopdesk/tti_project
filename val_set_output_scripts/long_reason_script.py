#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LONG_PREFIX = "<LONG_COT>\n"


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def format_question(example: dict) -> str:
    question = example["question"].strip()
    options = example["options"]

    option_lines = []
    if isinstance(options, list):
        for opt in options:
            key = str(opt["key"]).strip().upper()
            value = str(opt["value"]).strip()
            option_lines.append(f"{key}. {value}")
    else:
        for key in ["A", "B", "C", "D", "E", "F"]:
            if key in options:
                option_lines.append(f"{key}. {str(options[key]).strip()}")

    return f"{question}\nAnswer Choices:\n" + "\n".join(option_lines)


def extract_answer_block(text: str) -> Optional[str]:
    match = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", text, re.DOTALL | re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip()


def infer_answer_letter(full_output: str, options) -> Optional[str]:
    candidates = []

    answer_block = extract_answer_block(full_output)
    if answer_block:
        candidates.append(answer_block)
    candidates.append(full_output)

    if isinstance(options, list):
        options_dict = {str(x["key"]).strip().upper(): str(x["value"]).strip() for x in options}
    else:
        options_dict = {str(k).strip().upper(): str(v).strip() for k, v in options.items()}

    normalized_option_text_to_letter = {normalize_text(v): k for k, v in options_dict.items()}

    for candidate in candidates:
        candidate = candidate.strip()

        m = re.fullmatch(r"[<\s(/]*([A-F])[>\s).:/\\-]*", candidate, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        patterns = [
            r"(?:the\s+answer\s+is|answer\s*(?:choice)?|option)\s*[:\-]?\s*([A-F])\b",
            r"\b([A-F])\b(?=\s*</ANSWER>)",
            r"^\s*([A-F])[\s\.\):,-]*$",
        ]
        for pat in patterns:
            m = re.search(pat, candidate, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()

        candidate_norm = normalize_text(candidate)
        if candidate_norm in normalized_option_text_to_letter:
            return normalized_option_text_to_letter[candidate_norm]

        candidate_norm = normalize_text(re.sub(r"[\"'`.,;:]+$", "", candidate))
        if candidate_norm in normalized_option_text_to_letter:
            return normalized_option_text_to_letter[candidate_norm]

    return None


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


def batched(items: List[dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def make_llm(model_name: str, gpu_memory_utilization: float, max_model_len: int, dtype: str) -> LLM:
    return LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
    )


def generate_long_cot_batch(llm: LLM, tokenizer, batch_examples: List[dict], max_new_tokens: int, temperature: float):
    question_texts = [format_question(ex) for ex in batch_examples]
    prompts = [build_prompt(tokenizer, q, LONG_PREFIX) for q in question_texts]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    rows = []
    for ex, qtext, prompt, out in zip(batch_examples, question_texts, prompts, outputs):
        generated_text = out.outputs[0].text
        full_output = LONG_PREFIX + generated_text
        pred_letter = infer_answer_letter(full_output, ex["options"])
        prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids is not None else 0
        generated_tokens = len(out.outputs[0].token_ids)
        total_tokens = prompt_tokens + generated_tokens

        rows.append({
            "question": ex["question"],
            "formatted_question": qtext,
            "gold_answer_idx": str(ex["answer_idx"]).strip().upper(),
            "predicted_answer_idx": pred_letter,
            "is_correct": pred_letter == str(ex["answer_idx"]).strip().upper(),
            "prompt": prompt,
            "full_output": full_output,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": total_tokens,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect LONG_COT answers with vLLM and cache them.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
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

    rows = []
    total_correct = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_tokens = 0

    for batch in batched(dataset, args.batch_size):
        batch_rows = generate_long_cot_batch(
            llm=llm,
            tokenizer=tokenizer,
            batch_examples=batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        rows.extend(batch_rows)

        for r in batch_rows:
            total_correct += int(r["is_correct"])
            total_prompt_tokens += r["prompt_tokens"]
            total_generated_tokens += r["generated_tokens"]
            total_tokens += r["total_tokens"]

        print(f"Processed {len(rows)}/{len(dataset)} examples")

    total = len(rows)
    summary = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "mode": "long_cot",
        "total": total,
        "correct": total_correct,
        "accuracy": (total_correct / total) if total else 0.0,
        "avg_prompt_tokens": (total_prompt_tokens / total) if total else 0.0,
        "avg_generated_tokens": (total_generated_tokens / total) if total else 0.0,
        "avg_total_tokens": (total_tokens / total) if total else 0.0,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    payload = {"summary": summary, "results": rows}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved cache to:", args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
