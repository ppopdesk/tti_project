import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _coerce_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class DirectRow:
    idx: int
    log_diff: Optional[float]
    is_correct: bool


@dataclass(frozen=True)
class CotRow:
    idx: int
    is_correct: bool


def load_direct(direct_path: Path) -> Dict[int, DirectRow]:
    data = _read_json(direct_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {direct_path}, got {type(data)}")

    out: Dict[int, DirectRow] = {}
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        idx = _coerce_int(row.get("id", i))
        if idx is None:
            continue
        out[idx] = DirectRow(
            idx=idx,
            log_diff=_safe_float(row.get("log_diff")),
            is_correct=bool(row.get("probe_correct", False)),
        )
    return out


def load_cot(cot_path: Path) -> Dict[int, CotRow]:
    data = _read_json(cot_path)
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Expected an object with 'results' in {cot_path}")

    out: Dict[int, CotRow] = {}
    for row in data.get("results", []):
        if not isinstance(row, dict):
            continue
        idx = _coerce_int(row.get("idx"))
        if idx is None:
            continue
        out[idx] = CotRow(idx=idx, is_correct=bool(row.get("is_correct", False)))
    return out


def build_joined_frame(direct: Dict[int, DirectRow], cot: Dict[int, CotRow]) -> pd.DataFrame:
    common = sorted(set(direct.keys()) & set(cot.keys()))
    if not common:
        raise ValueError("No overlapping indices between direct and CoT outputs.")

    rows = []
    for idx in common:
        d = direct[idx]
        c = cot[idx]
        rows.append(
            {
                "idx": idx,
                "log_diff": d.log_diff,
                "direct_correct": bool(d.is_correct),
                "cot_correct": bool(c.is_correct),
            }
        )
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    return df


def grid_search_threshold(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    # escalate if log_diff < T (or if log_diff missing, treat as escalate)
    ld = df["log_diff"].astype(float)
    direct = df["direct_correct"].astype(bool).to_numpy()
    cot = df["cot_correct"].astype(bool).to_numpy()

    analysis: List[dict] = []
    for T in thresholds:
        escalated = (ld < T) | ld.isna()
        chosen_correct = np.where(escalated.to_numpy(), cot, direct)

        acc = float(np.mean(chosen_correct))
        esc_rate = float(np.mean(escalated))

        analysis.append(
            {
                "threshold": float(T),
                "accuracy": acc,
                "escalation_rate": esc_rate,
            }
        )
    return pd.DataFrame(analysis)


def _save_plots(df: pd.DataFrame, gs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so the script still runs headless without plotting deps installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Accuracy vs threshold (with escalation rate on secondary axis)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(gs["threshold"], gs["accuracy"], color="tab:blue", linewidth=2)
    ax1.set_xlabel("Log-diff threshold T (escalate if log_diff < T)")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(gs["threshold"], gs["escalation_rate"], color="tab:orange", linewidth=2, linestyle="--")
    ax2.set_ylabel("Escalation rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(out_dir / "threshold_accuracy_escalation.png", dpi=200)
    plt.close(fig)

    # 2) Log-diff distributions: direct correct vs incorrect
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = 40
    ax.hist(
        df.loc[df["direct_correct"], "log_diff"].dropna(),
        bins=bins,
        alpha=0.6,
        label="Direct correct",
        color="tab:green",
        density=True,
    )
    ax.hist(
        df.loc[~df["direct_correct"], "log_diff"].dropna(),
        bins=bins,
        alpha=0.6,
        label="Direct incorrect",
        color="tab:red",
        density=True,
    )
    ax.set_xlabel("log_diff (best - second best logprob)")
    ax.set_ylabel("Density")
    ax.set_title("Direct-answer confidence separation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "logdiff_hist_direct_correct_vs_incorrect.png", dpi=200)
    plt.close(fig)

    # 3) "Gain" from escalation by log-diff bucket
    bucket_edges = np.quantile(df["log_diff"].dropna(), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    bucket_edges = np.unique(bucket_edges)
    if len(bucket_edges) >= 3:
        b = pd.cut(df["log_diff"], bucket_edges, include_lowest=True)
        grouped = df.assign(bucket=b).groupby("bucket", observed=True).agg(
            n=("idx", "size"),
            direct_acc=("direct_correct", "mean"),
            cot_acc=("cot_correct", "mean"),
        )
        grouped["delta"] = grouped["cot_acc"] - grouped["direct_acc"]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(len(grouped)), grouped["delta"], color="tab:purple", alpha=0.8)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels([str(x) for x in grouped.index], rotation=30, ha="right")
        ax.set_ylabel("Accuracy gain (CoT - direct)")
        ax.set_title("Where CoT helps most (by log_diff bucket)")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "cot_gain_by_logdiff_bucket.png", dpi=200)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--direct",
        type=str,
        default=str(Path("confidence_based_reasoning") / "medqa_direct_answer_logprobs.json"),
        help="Path to direct-answer logprobs JSON (list).",
    )
    p.add_argument(
        "--cot",
        type=str,
        default=str(Path("val_set_outputs") / "cot_answers_validation.json"),
        help="Path to short-CoT outputs JSON (object with results).",
    )
    p.add_argument("--out_dir", type=str, default=str(Path("confidence_based_reasoning") / "analysis_out"))
    p.add_argument("--min_T", type=float, default=0.0)
    p.add_argument("--max_T", type=float, default=25.0)
    p.add_argument("--steps", type=int, default=251)
    args = p.parse_args()

    direct_path = Path(args.direct)
    cot_path = Path(args.cot)
    out_dir = Path(args.out_dir)

    direct = load_direct(direct_path)
    cot = load_cot(cot_path)
    df = build_joined_frame(direct, cot)

    direct_acc = float(df["direct_correct"].mean())
    cot_acc = float(df["cot_correct"].mean())

    thresholds = np.linspace(args.min_T, args.max_T, args.steps)
    gs = grid_search_threshold(df, thresholds)

    best = gs.sort_values(["accuracy", "threshold"], ascending=[False, True]).iloc[0]
    best_T = float(best["threshold"])
    best_acc = float(best["accuracy"])
    best_esc = float(best["escalation_rate"])

    out_dir.mkdir(parents=True, exist_ok=True)
    gs.to_csv(out_dir / "grid_search.csv", index=False)

    summary = {
        "direct_path": str(direct_path),
        "cot_path": str(cot_path),
        "n_common": int(len(df)),
        "direct_accuracy": direct_acc,
        "cot_accuracy": cot_acc,
        "best_threshold": best_T,
        "best_accuracy": best_acc,
        "best_escalation_rate": best_esc,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _save_plots(df, gs, out_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

