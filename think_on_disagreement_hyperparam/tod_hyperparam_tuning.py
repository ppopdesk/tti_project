#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SHORT_PREFIX = "<COT>\n"
DEFAULT_TEMPERATURES = [0.4, 0.6, 0.8, 1.0]
DEFAULT_K_VALUES = [2, 3, 4]
DEFAULT_SEEDS = [11, 22, 33, 44]


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


def load_long_reason_cache(path: str) -> Dict[int, dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["results"]
    cache = {}
    for row in rows:
        idx = int(row["idx"])
        cache[idx] = row
    return cache


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


def extract_letter(text: str, valid_letters: List[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None

    valid_letters = [x.upper() for x in valid_letters]

    m = re.search(r"<ANSWER>\s*([A-Z])\s*</ANSWER>", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    m = re.search(r"(?:answer is|answer:|final answer[:\s]+)\s*([A-Z])", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", text.upper())
    for letter in reversed(matches):
        if letter in valid_letters:
            return letter

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


def build_short_prompt(tokenizer, example: dict) -> str:
    return build_prompt(tokenizer, format_question(example), SHORT_PREFIX)


def make_llm(model_name: str, gpu_memory_utilization: float, max_model_len: int, dtype: str) -> LLM:
    return LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
    )


def batched(items: List[int], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_short_trials(
    llm: LLM,
    tokenizer,
    dataset: List[dict],
    k: int,
    temperature: float,
    seeds: List[int],
    batch_size: int,
    max_new_tokens: int,
    top_p: float,
) -> Dict[int, dict]:
    if len(seeds) < k:
        raise ValueError(f"Need at least {k} seeds, but got {len(seeds)} seeds")

    prompts = [build_short_prompt(tokenizer, ex) for ex in dataset]
    indices = list(range(len(dataset)))

    short_results = {
        idx: {
            "letters": [],
            "raw_outputs": [],
            "trial_token_usage": [],
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "total_tokens": 0,
        }
        for idx in indices
    }

    for trial_idx in range(k):
        seed = seeds[trial_idx]
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        for batch_indices in batched(indices, batch_size):
            batch_prompts = [prompts[idx] for idx in batch_indices]
            outputs = llm.generate(batch_prompts, sampling_params)

            for data_idx, out in zip(batch_indices, outputs):
                example = dataset[data_idx]
                options = example["options"]
                if isinstance(options, list):
                    valid_letters = [str(opt["key"]).strip().upper() for opt in options]
                else:
                    valid_letters = [str(k).strip().upper() for k in options.keys()]

                raw_output = SHORT_PREFIX + out.outputs[0].text
                pred_letter = extract_letter(raw_output, valid_letters)
                prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids is not None else 0
                generated_tokens = len(out.outputs[0].token_ids)
                total_tokens = prompt_tokens + generated_tokens

                short_results[data_idx]["letters"].append(pred_letter)
                short_results[data_idx]["raw_outputs"].append(raw_output)
                short_results[data_idx]["trial_token_usage"].append({
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "total_tokens": total_tokens,
                })
                short_results[data_idx]["prompt_tokens"] += prompt_tokens
                short_results[data_idx]["generated_tokens"] += generated_tokens
                short_results[data_idx]["total_tokens"] += total_tokens

        print(f"  completed short-CoT trial {trial_idx + 1}/{k} | k={k} temp={temperature:.1f} seed={seed}")

    return short_results


def unanimous_vote(letters: List[Optional[str]]) -> Optional[str]:
    clean = [x for x in letters if x is not None]
    if len(clean) != len(letters):
        return None
    first = clean[0]
    if all(x == first for x in clean):
        return first
    return None


def evaluate_grid_point(
    dataset: List[dict],
    short_results: Dict[int, dict],
    long_cache: Dict[int, dict],
    k: int,
    temperature: float,
) -> Dict[str, object]:
    per_question = []
    correct = 0
    escalated = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_tokens = 0

    for idx, example in enumerate(dataset):
        gold_answer = str(example["answer_idx"]).strip().upper()
        short_info = short_results[idx]
        unanimous_answer = unanimous_vote(short_info["letters"])

        final_prompt_tokens = short_info["prompt_tokens"]
        final_generated_tokens = short_info["generated_tokens"]
        final_total_tokens = short_info["total_tokens"]

        if unanimous_answer is not None:
            final_answer = unanimous_answer
            used_long_reason = False
            answer_source = "short_cot_unanimous"
            long_reason_row = None
        else:
            used_long_reason = True
            escalated += 1
            long_reason_row = long_cache[idx]
            final_answer = long_reason_row.get("predicted_answer_idx")
            answer_source = "long_reason_escalation"
            final_prompt_tokens += int(long_reason_row.get("token_usage", {}).get("prompt_tokens", long_reason_row.get("prompt_tokens", 0)))
            final_generated_tokens += int(long_reason_row.get("token_usage", {}).get("generated_tokens", long_reason_row.get("generated_tokens", 0)))
            final_total_tokens += int(long_reason_row.get("token_usage", {}).get("total_tokens", long_reason_row.get("total_tokens", 0)))

        is_correct = final_answer == gold_answer
        correct += int(is_correct)
        total_prompt_tokens += final_prompt_tokens
        total_generated_tokens += final_generated_tokens
        total_tokens += final_total_tokens

        row = {
            "idx": idx,
            "question": example["question"],
            "answer": gold_answer,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "answer_source": answer_source,
            "used_long_reason": used_long_reason,
            "k": k,
            "temperature": temperature,
            "short_cot_answers": short_info["letters"],
            "short_cot_raw_outputs": short_info["raw_outputs"],
            "short_cot_token_usage": short_info["trial_token_usage"],
            "token_usage": {
                "prompt_tokens": final_prompt_tokens,
                "generated_tokens": final_generated_tokens,
                "total_tokens": final_total_tokens,
            },
            "prompt_tokens": final_prompt_tokens,
            "generated_tokens": final_generated_tokens,
            "total_tokens": final_total_tokens,
        }
        if long_reason_row is not None:
            row["long_reason_answer"] = long_reason_row.get("predicted_answer_idx")
            row["long_reason_raw_output"] = long_reason_row.get("raw_output", long_reason_row.get("full_output"))
            row["long_reason_token_usage"] = long_reason_row.get("token_usage", {
                "prompt_tokens": long_reason_row.get("prompt_tokens", 0),
                "generated_tokens": long_reason_row.get("generated_tokens", 0),
                "total_tokens": long_reason_row.get("total_tokens", 0),
            })
        per_question.append(row)

    total = len(dataset)
    metrics = {
        "k": k,
        "temperature": temperature,
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "escalated": escalated,
        "escalation_rate": escalated / total if total else 0.0,
        "avg_prompt_tokens": total_prompt_tokens / total if total else 0.0,
        "avg_generated_tokens": total_generated_tokens / total if total else 0.0,
        "avg_total_tokens": total_tokens / total if total else 0.0,
    }
    return {
        "metrics": metrics,
        "per_question": per_question,
    }


def save_heatmap(df: pd.DataFrame, value_col: str, out_path: str, title: str):
    pivot = df.pivot(index="k", columns="temperature", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("temperature")
    ax.set_ylabel("k")
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grid search think-on-disagreement with cached long reasoning answers.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--long_reason_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperatures", type=str, default=",".join(str(x) for x in DEFAULT_TEMPERATURES))
    parser.add_argument("--k_values", type=str, default=",".join(str(x) for x in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_examples", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    temperatures = parse_csv_floats(args.temperatures)
    k_values = parse_csv_ints(args.k_values)
    seeds = parse_csv_ints(args.seeds)

    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading validation data from {args.data_path}")
    dataset = load_jsonl(args.data_path)
    if args.num_examples > 0:
        dataset = dataset[:args.num_examples]
    print(f"Loaded {len(dataset)} validation examples")

    print(f"Loading long reason cache from {args.long_reason_json}")
    long_cache = load_long_reason_cache(args.long_reason_json)
    if len(long_cache) < len(dataset):
        raise ValueError(
            f"Long-reason cache has {len(long_cache)} rows, but dataset has {len(dataset)} examples."
        )

    print("Initializing vLLM...")
    llm = make_llm(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    metrics_rows = []
    detailed_results = {}

    for k in k_values:
        for temperature in temperatures:
            print(f"\nRunning grid point: k={k}, temperature={temperature:.1f}")
            short_results = run_short_trials(
                llm=llm,
                tokenizer=tokenizer,
                dataset=dataset,
                k=k,
                temperature=temperature,
                seeds=seeds,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
            )
            result = evaluate_grid_point(
                dataset=dataset,
                short_results=short_results,
                long_cache=long_cache,
                k=k,
                temperature=temperature,
            )
            metrics = result["metrics"]
            metrics_rows.append(metrics)
            detailed_results[f"k{k}_temp{temperature:.1f}"] = result
            print(
                f"  accuracy={metrics['accuracy']:.4f} | "
                f"avg_total_tokens={metrics['avg_total_tokens']:.2f} | "
                f"escalation_rate={metrics['escalation_rate']:.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["accuracy", "avg_total_tokens", "escalation_rate"],
        ascending=[False, True, True],
    )

    metrics_csv = os.path.join(args.output_dir, "grid_search_metrics.csv")
    metrics_json = os.path.join(args.output_dir, "grid_search_metrics.json")
    detailed_json = os.path.join(args.output_dir, "grid_search_detailed.json")
    table_txt = os.path.join(args.output_dir, "grid_search_table.txt")

    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)
    with open(detailed_json, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f)

    with open(table_txt, "w", encoding="utf-8") as f:
        f.write(metrics_df.to_string(index=False))

    save_heatmap(
        metrics_df,
        value_col="accuracy",
        out_path=os.path.join(args.output_dir, "accuracy_heatmap.png"),
        title="Accuracy by (k, temperature)",
    )
    save_heatmap(
        metrics_df,
        value_col="avg_total_tokens",
        out_path=os.path.join(args.output_dir, "avg_total_tokens_heatmap.png"),
        title="Average Total Tokens by (k, temperature)",
    )

    print("\n=== FINAL TABLE ===")
    print(metrics_df.to_string(index=False))

    if len(metrics_df) > 0:
        best = metrics_df.iloc[0].to_dict()
        print("\nBest grid point:")
        print(json.dumps(best, indent=2))

    print(f"\nSaved metrics csv to {metrics_csv}")
    print(f"Saved metrics json to {metrics_json}")
    print(f"Saved detailed per-question results to {detailed_json}")
    print(f"Saved text table to {table_txt}")


if __name__ == "__main__":
    main()
