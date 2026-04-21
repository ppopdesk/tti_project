import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
MODEL_NAME = "ShivaniiKum/qwen-medreason-finetuned"
VAL_SIZE = 300
SAVE_INTERVAL = 10 

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent 

INPUT_FILE = PROJECT_ROOT / "med_qa_json" / "validation.json" 
OUTPUT_DIR = Path(os.environ.get("MEDQA_CALIBRATION_DIR", PROJECT_ROOT)).resolve()
OUTPUT_FILE = OUTPUT_DIR / "medqa_direct_answer_logprobs.json"

# Token IDs for A-E in Qwen2.5
LETTER_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36}
IDX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# --- Core Logic ---

def build_prompt(tokenizer, question_text: str, prefix: str) -> str:
    """
    Uses the model's specific chat template to format the conversation.
    continue_final_message=True ensures no extra EOT/BOS tokens are added after the prefix.
    """
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

def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    tmp_path.replace(path)

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

def main():
    print(f"Loading local MedQA data from {INPUT_FILE}...")
    examples = load_local_jsonl(INPUT_FILE, VAL_SIZE)
    if not examples:
        print("Error: Dataset not found or empty.")
        return

    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    results = []

    for i, ex in enumerate(examples):
        # 1. Map answer to letter
        try:
            correct_idx = ex['choices'].index(ex['answer'][0])
            correct_letter = IDX_TO_LETTER[correct_idx]
        except (ValueError, IndexError):
            continue

        # 2. Format options and combined user instruction
        opts = "\n".join([f"{IDX_TO_LETTER[j]}) {text}" for j, text in enumerate(ex['choices'])])
        
        # We put the detailed instruction inside the 'user' block
        user_instruction = (
            f"Question: {ex['question']}\nOptions:\n{opts}\n\n"
            "You are a medical expert. Provide the letter corresponding to the correct final answer. "
            "Your output should only be the letter, nothing else."
        )

        # 3. Build prompt using Chat Template
        # Prefix is strictly "<ANSWER>\n" as requested
        full_prompt = build_prompt(tokenizer, user_instruction, "<ANSWER>\n")
        
        # 4. Inference
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # 5. Extract Logprobs
        next_token_logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        choice_lps = {
            letter: log_probs[tid].item()
            for letter, tid in LETTER_TOKEN_IDS.items()
        }

        # 6. Metrics
        sorted_lps = sorted(choice_lps.items(), key=lambda x: x[1], reverse=True)
        best_letter, best_lp = sorted_lps[0]
        second_letter, second_lp = sorted_lps[1]

        results.append({
            "id": ex.get("id", i),
            "correct_letter": correct_letter,
            "best_letter": best_letter,
            "best_logprob": round(best_lp, 6),
            "second_letter": second_letter,
            "second_logprob": round(second_lp, 6),
            "log_diff": round(best_lp - second_lp, 6),
            "probe_correct": bool(best_letter == correct_letter),
        })

        # --- Periodic Checkpoint ---
        if (i + 1) % SAVE_INTERVAL == 0:
            _atomic_write_json(OUTPUT_FILE, results)
            print(f"Processed {i+1}/{len(examples)} - Logprob for correct ({correct_letter}): {choice_lps[correct_letter]:.4f}")

    _atomic_write_json(OUTPUT_FILE, results)
    print(f"Final results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()