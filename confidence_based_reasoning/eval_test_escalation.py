import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DirectRow:
    idx: int
    correct_letter: Optional[str]
    best_letter: Optional[str]
    log_diff: Optional[float]


@dataclass(frozen=True)
class ShortCotRow:
    idx: int
    final_answer: Optional[str]


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _norm_letter(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if len(s) != 1:
        return None
    if s not in {"A", "B", "C", "D", "E"}:
        return None
    return s


def load_direct(path: Path) -> Dict[int, DirectRow]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")

    out: Dict[int, DirectRow] = {}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        idx = _to_int(row.get("id", i))
        if idx is None:
            continue
        out[idx] = DirectRow(
            idx=idx,
            correct_letter=_norm_letter(row.get("correct_letter")),
            best_letter=_norm_letter(row.get("best_letter")),
            log_diff=_to_float(row.get("log_diff")),
        )
    return out


def load_short_cot(path: Path) -> Dict[int, ShortCotRow]:
    data = _read_json(path)
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Expected object with 'results' in {path}")

    out: Dict[int, ShortCotRow] = {}
    for row in data.get("results", []):
        if not isinstance(row, dict):
            continue
        idx = _to_int(row.get("idx"))
        if idx is None:
            continue
        out[idx] = ShortCotRow(
            idx=idx,
            final_answer=_norm_letter(row.get("final_answer")),
        )
    return out


def evaluate(direct: Dict[int, DirectRow], cot: Dict[int, ShortCotRow], threshold: float) -> Tuple[float, float, int]:
    common = sorted(set(direct.keys()) & set(cot.keys()))
    if not common:
        raise ValueError("No overlapping indices between direct and short-CoT files.")

    correct = 0
    escalated = 0
    usable = 0

    for idx in common:
        d = direct[idx]
        c = cot[idx]
        gold = d.correct_letter
        if gold is None:
            continue

        use_cot = d.log_diff is None or d.log_diff < threshold
        pred = c.final_answer if use_cot else d.best_letter
        if pred is None:
            continue

        usable += 1
        if use_cot:
            escalated += 1
        if pred == gold:
            correct += 1

    if usable == 0:
        raise ValueError("No usable rows (missing gold/preds).")

    accuracy = correct / usable
    escalation_rate = escalated / usable
    return accuracy, escalation_rate, usable


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--direct",
        default="medqa_direct_answer_logprobs_test.json",
        help="Path to test direct-answer logprobs JSON.",
    )
    p.add_argument(
        "--cot",
        default=str(Path("token_entropy") / "short_reasoning_results.json"),
        help="Path to short-CoT test results JSON.",
    )
    p.add_argument("--threshold", type=float, default=3.75)
    args = p.parse_args()

    direct_path = Path(args.direct)
    cot_path = Path(args.cot)

    direct = load_direct(direct_path)
    cot = load_short_cot(cot_path)
    acc, esc_rate, n = evaluate(direct, cot, args.threshold)

    print(
        json.dumps(
            {
                "n_common": len(set(direct.keys()) & set(cot.keys())),
                "n_used": n,
                "threshold": args.threshold,
                "accuracy": acc,
                "escalation_rate": esc_rate,
                "direct_path": str(direct_path),
                "cot_path": str(cot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

