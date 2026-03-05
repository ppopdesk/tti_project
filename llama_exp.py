import json
import re
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple

import requests
import json as json_module
import random

from config import OLLAMA_URL, MODEL, TEST_FILE


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


def build_mcq_prompt(question: str, options: List[Dict]) -> str:
    # options like [{"key":"A","value":"Asthma"}, ...]
    opts = "\n".join([f"{opt['key']}) {opt['value']}" for opt in options])
    return (
        "You are answering a multiple-choice question.\n"
        "Return ONLY the letter of the correct option (A, B, C, D, ...).\n\n"
        f"Question:\n{question}\n\nOptions:\n{opts}\n\nAnswer:"
    )


def extract_letter(text: str, valid_letters: List[str]) -> Optional[str]:
    # Find the first standalone letter among valid choices
    # e.g., matches "A" or "A)" or "A." etc.
    pattern = r"\b(" + "|".join(map(re.escape, valid_letters)) + r")\b"
    m = re.search(pattern, text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # fallback: look for leading letter
    t = text.strip().upper()
    if t and t[0] in valid_letters:
        return t[0]

    return text


def sample_k_letters(question: str, options: List[Dict], seeds: List[int]) -> List[Optional[str]]:
    valid_letters = [opt['key'] for opt in options]
    prompt = build_mcq_prompt(question, options)

    letters = []
    for s in seeds:
        raw, usage = ollama_generate(
            prompt,
            options={
                "seed": s,
                "temperature": 0.8,
                "top_p": 0.95,
                "num_predict": 10,   # only need a letter
            },
        )
        letters.append(extract_letter(raw, valid_letters))
    return letters, usage


def long_think_answer(question: str, options: List[Dict]) -> Optional[str]:
    valid_letters = [opt["key"].strip().upper() for opt in options]
    opts = "\n".join([f"{opt['key']}) {opt['value']}" for opt in options])

    # Force explicit reasoning + constrained final output channel
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{opts}\n\n"
        "Instructions:\n"
        "1) Analyze the question and each option step-by-step.\n"
        "2) Write your reasoning inside <thought>...</thought>.\n"
        f"3) Write ONLY the final answer letter (one of {', '.join(valid_letters)}) inside <answer>...</answer>.\n\n"
        "Begin.\n"
    )

    raw, usage = ollama_generate(
        prompt,
        options={
            # More deterministic for reasoning
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            # Give room to think out loud
            "num_predict": 2000,
            # Keep model loaded for iterative experiments
            "keep_alive": "5m",
        },
    )

    # Returns the extracted letter if found, otherwise returns raw
    patterns = [
        r"<answer>\s*(.*?)\s*</answer>",                 # closed <answer>
        r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",     # closed <FINAL_ANSWER>
        r"<answer>\s*([^\n<]+)",                         # open <answer> ... (no close)
        r"<FINAL_ANSWER>\s*([^\n<]+)",                   # open <FINAL_ANSWER> ... (no close)
    ]

    for pat in patterns:
        m = re.search(pat, raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            answer_text = m.group(1).strip()
            letter = extract_letter(answer_text, valid_letters)
            if letter is not None:
                return letter, usage

    return raw, usage


def majority_vote(letters: List[Optional[str]]) -> Optional[str]:
    # Remove Nones
    clean = [x for x in letters if x is not None]
    if not clean:
        return None
    counts = Counter(clean).most_common()
    top_letter, top_count = counts[0]
    return top_letter if top_count == 3 else None  # majority for k=3


def adaptive_mcq(question: str, options: List[Dict], seeds: List[int] = [11, 22, 33]) -> Dict[str, Any]:
    letters, usage_fast = sample_k_letters(question, options, seeds)
    maj = majority_vote(letters)

    if maj is not None:
        return {
            "final": maj,
            "mode": "fast-majority",
            "usage_dict": usage_fast
        }

    final, usage_slow = long_think_answer(question, options)
    return {
        "final": final,
        "mode": "think",
        "usage_dict": usage_slow
    }


if __name__ == "__main__":
    with open(TEST_FILE, "r") as f:
        print("loading train dataset...")
        lines = f.readlines()

    total_train = len(lines)

    correct = 0
    think_count = 0
    format_error = 0
    total_prompt_count = 0
    total_output_count = 0

    for idx, line in enumerate(lines, start=1):
        row = json_module.loads(line)
        question = row["question"]
        options = row["options"]
        real_answer = str(row["answer_idx"]).strip().upper()

        out = adaptive_mcq(question, options)
        pred = (out.get("final") or "").strip().upper()
        if len(pred) != 1:
            pred = "Format Error"
            format_error+=1

        if pred == real_answer:
            correct += 1
        if out.get("mode") == "think":
            think_count += 1
        
        total_prompt_count += out.get("usage_dict").get("prompt_tokens")
        total_output_count += out.get("usage_dict").get("completion_tokens")

        print(
            f"[{idx}/{total_train}] pred={pred or 'None'} real={real_answer} mode={out.get('mode')}"
        )

    accuracy = (correct / total_train) * 100 if total_train else 0.0
    think_pct = (think_count / total_train) * 100 if total_train else 0.0
    error_pct = (format_error / total_train) * 100 if total_train else 0.0
    prompt_avg = (total_prompt_count/total_train) if total_train else 0.0
    output_avg = (total_output_count/total_train) if total_train else 0.0

    print("\n" + "=" * 50)
    print(f"Final accuracy: {accuracy:.2f}% ({correct}/{total_train})")
    print(f"Think mode percentage: {think_pct:.2f}% ({think_count}/{total_train})")
    print(f"Format Error percentage: {error_pct:.2f}% ({format_error}/{total_train})")
    print(f"Average Prompt tokens: {prompt_avg:.2f} ({total_prompt_count}/{total_train})%")
    print(f"Average Output tokens: {output_avg:.2f} ({total_output_count}/{total_train})%")
