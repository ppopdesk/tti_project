import json
import os
import re
from typing import Dict, Any, Optional, List, Tuple

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "vijayavp/medreason-qwen25-shortcot-exp2:latest")
DATA_PATH = os.getenv("EVAL_DATA_PATH", "/home/vijayavp/ttc_project/finetune/MedReason/eval_data/medqa_test.jsonl")


def ollama_generate(prompt: str, *, options: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, int]]:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return data["response"], usage


def normalize_options(options: Any) -> List[Dict[str, str]]:
    if isinstance(options, dict):
        return [{"key": str(k).strip().upper(), "value": v} for k, v in options.items()]
    return [{"key": str(opt["key"]).strip().upper(), "value": opt["value"]} for opt in options]


def build_prompt(question: str, options: List[Dict[str, str]]) -> str:
    option_str = "\n".join(f"{opt['key']}. {opt['value']}" for opt in options)
    return (
        "Answer the following multiple-choice medical question.\n"
        "First do a short chain of thought under '## Thinking' to help arrive at the answer.\n"
        "Keep it brief, relevant, and only 1-2 short lines.\n"
        "Then give the final answer under '## Final Answer' in the format 'A. Answer text'.\n"
        "Do not write anything after the Final Answer line.\n"
        "Do not add extra explanation after the final answer.\n\n"
        f"{question}\n"
        f"{option_str}\n"
    )


def extract_letter(text: str, valid_letters: List[str]) -> Optional[str]:
    pattern = r"\b(" + "|".join(map(re.escape, valid_letters)) + r")\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def extract_letter_from_final_answer(text: str, valid_letters: List[str]) -> Optional[str]:
    final_answer_match = re.search(r"##\s*Final\s*Answer", text, flags=re.IGNORECASE)
    if final_answer_match:
        text = text[final_answer_match.start():]

    patterns = [
        r"##\s*Final\s*Answer\s*:\s*\(?\s*([A-Z])\s*\)?[\.\):\s-]",
        r"##\s*Final\s*Answer\s+\(?\s*([A-Z])\s*\)?[\.\):\s-]",
        r"##\s*Final\s*Answer\s*\n+\s*\(?\s*([A-Z])\s*\)?[\.\):\s-]",
        r"The answer is\s*\(?\s*([A-Z])\s*\)?\b",
        r"answer\s+is\s*\(?\s*([A-Z])\s*\)?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter

    return None


if __name__ == "__main__":
    with open(DATA_PATH, "r") as f:
        lines = f.readlines()

    total = len(lines)
    correct = 0
    format_error = 0
    total_prompt = 0
    total_output = 0

    for idx, line in enumerate(lines, start=1):
        row = json.loads(line)
        question = row["question"]
        options = normalize_options(row["options"])
        gold = str(row["answer_idx"]).strip().upper()

        prompt = build_prompt(question, options)
        raw, usage = ollama_generate(
            prompt,
            options={
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_predict": 150,
            },
        )

        print(f"Raw: {raw}")
        print("-" * 80)

        valid_letters = [opt["key"] for opt in options]

        pred = extract_letter_from_final_answer(raw, valid_letters)

        if pred is None:
            pred = "Format Error"
            format_error += 1

        if pred == gold:
            correct += 1

        total_prompt += usage["prompt_tokens"]
        total_output += usage["completion_tokens"]

        print(f"[{idx}/{total}] pred={pred} real={gold}")

    accuracy = 100 * correct / total if total else 0.0
    error_pct = 100 * format_error / total if total else 0.0
    avg_prompt = total_prompt / total if total else 0.0
    avg_output = total_output / total if total else 0.0

    print("\n" + "=" * 50)
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Format Error percentage: {error_pct:.2f}% ({format_error}/{total})")
    print(f"Average Prompt tokens: {avg_prompt:.2f} ({total_prompt}/{total})")
    print(f"Average Output tokens: {avg_output:.2f} ({total_output}/{total})")