import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

from openai import OpenAI


PROMPT_TEMPLATE = """You are helping create training data for a medical QA model.

Given:
1. a medical multiple-choice question,
2. a long reasoning trace,
3. the correct answer,

your task is to compress the long reasoning into a short, high-yield reasoning snippet that would help a model arrive at the correct answer in the future.

Requirements:
- Write only 2 to 3 sentences.
- Keep it under 75 tokens.
- Do not restate the full question.
- Do not include the final conclusion, final diagnosis, or final treatment recommendation as a standalone answer sentence.
- Do not say phrases like “therefore,” “thus,” “the answer is,” or “so the correct option is.”
- Focus only on the key clinical clues and the minimal reasoning chain that supports the correct answer.
- Keep it medically relevant and concise.
- Do not mention distractor options unless necessary.
- Output only the compressed reasoning.

Question:
{question}

Long reasoning:
{reasoning}

Correct answer:
{answer}
"""

DATA_PATH = "medreason_train1.jsonl"
OUTPUT_PATH = "summary_full.jsonl"

BASE_URL = "http://promaxgb10-d473.eecs.umich.edu:8000/v1"
API_KEY = "api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
MODEL_NAME = "openai/gpt-oss-120b"

MAX_WORKERS = 8


def format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, str):
        return options.strip()
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    if isinstance(options, list):
        lines = []
        for opt in options:
            if isinstance(opt, dict) and "key" in opt and "value" in opt:
                lines.append(f"{opt['key']}. {opt['value']}")
            else:
                lines.append(str(opt))
        return "\n".join(lines)
    return str(options)


def build_qn_ans_reasoning(data_point: Dict[str, Any]) -> Tuple[str, str, str]:
    curr_qn = data_point.get("question", "").strip()
    curr_ans = data_point.get("answer", "").strip()
    curr_reasoning = data_point.get("reasoning", "").strip()

    options_str = format_options(data_point.get("options", ""))
    if options_str:
        curr_qn += "\n" + options_str

    return curr_qn, curr_ans, curr_reasoning


def build_prompt(question: str, reasoning: str, answer: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=question,
        reasoning=reasoning,
        answer=answer,
    )


def load_data(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_one(client: OpenAI, idx: int, data_point: Dict[str, Any]) -> Dict[str, Any]:
    question, answer, reasoning = build_qn_ans_reasoning(data_point)
    prompt = build_prompt(question=question, reasoning=reasoning, answer=answer)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    dt = time.perf_counter() - t0

    content = response.choices[0].message.content

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "sample_idx": idx,
        "dataset_name": data_point.get("dataset_name"),
        "id_in_dataset": data_point.get("id_in_dataset"),
        "question": question,
        "answer": answer,
        "short_reasoning": content,
        "latency_sec": dt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    all_data = load_data(DATA_PATH)
    total_dataset_size = len(all_data)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(call_one, client, idx, dp)
            for idx, dp in enumerate(all_data, start=1)
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{i}/{total_dataset_size}] "
                f"id={result['id_in_dataset']} "
                f"latency={result['latency_sec']:.2f}s "
                f"completion_tokens={result['completion_tokens']}"
            )

            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start

    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    prompt_token_vals = [r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None]
    completion_token_vals = [r["completion_tokens"] for r in results if r["completion_tokens"] is not None]
    total_token_vals = [r["total_tokens"] for r in results if r["total_tokens"] is not None]

    avg_prompt_tokens = sum(prompt_token_vals) / len(prompt_token_vals) if prompt_token_vals else None
    avg_completion_tokens = sum(completion_token_vals) / len(completion_token_vals) if completion_token_vals else None
    avg_total_tokens = sum(total_token_vals) / len(total_token_vals) if total_token_vals else None

    print("\n" + "=" * 60)
    print(f"Total dataset size: {total_dataset_size}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Wall-clock time: {elapsed/3600:.2f} hr")
    print(f"Average latency per datapoint: {avg_latency:.2f} sec")

    if avg_prompt_tokens is not None:
        print(f"Average prompt tokens: {avg_prompt_tokens:.2f}")
        print(f"Average completion tokens: {avg_completion_tokens:.2f}")
        print(f"Average total tokens: {avg_total_tokens:.2f}")

    print(f"Saved outputs to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()