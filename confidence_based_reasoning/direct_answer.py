import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
MODEL_NAME = "ShivaniiKum/qwen-medreason-finetuned"
VAL_SIZE = 300
SAVE_INTERVAL = 10  # Save progress every 10 questions

_SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = _SCRIPT_DIR / "med_qa_json" / "validation.json" 
OUTPUT_DIR = Path(os.environ.get("MEDQA_CALIBRATION_DIR", _SCRIPT_DIR)).resolve()
OUTPUT_FILE = OUTPUT_DIR / "medqa_direct_answer_logprobs.json"

# Token IDs for A-E in Qwen2.5
LETTER_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
IDX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# The specific prompt template provided
NO_REASONING_PROMPT = (
    "Question: {question}\nOptions:\n{options_text}\n\n"
    "You are a medical expert. "
    "Provide the letter corresponding to the correct final answer to this multiple choice question. "
    "Your output should only be the letter of your chosen output choice, nothing else."
)

# --- Utilities ---

def _atomic_write_json(path: Path, obj) -> None:
    """Writes to a temporary file then renames to avoid corruption during crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    tmp_path.replace(path)

def format_prompt(question, options_text):
    return NO_REASONING_PROMPT.format(
        question=question, 
        options_text=options_text
    )

def load_local_jsonl(path, limit):
    data = []
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data.append(json.loads(line))
    return data

# --- Main Logic ---

def main():
    print(f"Loading local MedQA data from {INPUT_FILE}...")
    examples = load_local_jsonl(INPUT_FILE, VAL_SIZE)
    
    if examples is None:
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    results = []

    print(f"Running direct answer probes on {len(examples)} questions...")
    for i, ex in enumerate(examples):
        # 1. Map text answer to letter (A-E)
        try:
            correct_answer_text = ex['answer'][0] 
            correct_idx = ex['choices'].index(correct_answer_text)
            correct_letter = IDX_TO_LETTER[correct_idx]
        except (ValueError, IndexError):
            print(f"Skipping ID {ex.get('id', i)} due to answer mismatch.")
            continue

        # 2. Format options and prompt
        opts_list = [f"{IDX_TO_LETTER[j]}) {text}" for j, text in enumerate(ex['choices'])]
        options_formatted = "\n".join(opts_list)
        prompt_text = format_prompt(ex['question'], options_formatted)
        
        # 3. Inference
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. Extract Logprobs for A-E at the next-token position
        next_token_logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        choice_lps = {
            letter: log_probs[tid].item()
            for letter, tid in LETTER_TOKEN_IDS.items()
        }

        # 5. Calculate Metrics
        sorted_lps = sorted(choice_lps.items(), key=lambda x: x[1], reverse=True)
        best_letter, best_lp = sorted_lps[0]
        second_letter, second_lp = sorted_lps[1]
        log_diff = float(best_lp - second_lp)

        results.append({
            "id": ex.get("id", i),
            "correct_letter": correct_letter,
            "best_letter": best_letter,
            "best_logprob": round(best_lp, 6),
            "second_letter": second_letter,
            "second_logprob": round(second_lp, 6),
            "log_diff": round(log_diff, 6),
            "probe_correct": bool(best_letter == correct_letter),
        })

        # --- Periodic Checkpoint ---
        current_count = i + 1
        if current_count % SAVE_INTERVAL == 0:
            _atomic_write_json(OUTPUT_FILE, results)
            print(f"   Processed {current_count}/{len(examples)} - Checkpoint saved.")
        elif current_count % 10 == 0: # Still print progress even if not saving
             print(f"   Processed {current_count}/{len(examples)}...")

    # Final Save
    _atomic_write_json(OUTPUT_FILE, results)
    print(f"\nProcessing complete. Final results written to {OUTPUT_FILE}")

    # Final summary stats
    if results:
        accuracy = sum(r["probe_correct"] for r in results) / len(results)
        print(f"Final Probe Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()