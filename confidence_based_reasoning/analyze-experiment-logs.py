import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. THE PARSER ---
def parse_ollama_logs(file_path):
    """
    Extracts structured data from the terminal output logs.
    Supports format with: Top choice, Log-Diff, Probe/Reasoning/Total output tokens,
    FINAL RESULT with optional Target.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, 'r') as f:
        content = f.read()

    # Split into question blocks (Q1 Analysis: ... ----, Q2 Analysis: ...)
    blocks = re.split(r'\n(?=Q\d+ Analysis:)', content.strip())
    if blocks and not blocks[0].strip().startswith('Q'):
        blocks = blocks[1:]  # Drop leading non-block content

    data = []
    for block in blocks:
        block = block.strip()
        if not block.startswith('Q'):
            continue

        # Extract fields with simpler patterns
        id_match = re.search(r'Q(\d+) Analysis:', block)
        top_match = re.search(r'Top choice: ([A-D])', block)
        log_match = re.search(r'Initial Log-Diff: ([\d\.]+)', block)
        final_match = re.search(r'FINAL RESULT: ([A-D]) \| (CORRECT|WRONG)(?: \(Target: ([A-D])\))?', block)
        probe_tok_match = re.search(r'Probe output tokens: (\d+)', block)
        reason_tok_match = re.search(r'Reasoning output tokens: (\d+)', block)
        total_tok_match = re.search(r'Total output tokens: (\d+)', block)

        if not all([id_match, top_match, log_match, final_match]):
            continue

        # Reasoning was used if we see "Invoking Reasoning Mode" (not "High confidence")
        used_reasoning = 'Invoking Reasoning Mode' in block

        target_letter = final_match.group(3) if final_match.group(3) else final_match.group(1)
        initial_correct = (top_match.group(1) == target_letter)

        data.append({
            "id": int(id_match.group(1)),
            "top_choice": top_match.group(1),
            "log_diff": float(log_match.group(1)),
            "used_reasoning": used_reasoning,
            "final_pred": final_match.group(1),
            "is_correct": final_match.group(2) == "CORRECT",
            "target": target_letter,
            "initial_correct": initial_correct,
            "probe_output_tokens": int(probe_tok_match.group(1)) if probe_tok_match else 0,
            "reasoning_output_tokens": int(reason_tok_match.group(1)) if reason_tok_match else 0,
            "total_output_tokens": int(total_tok_match.group(1)) if total_tok_match else 0,
        })

    return pd.DataFrame(data)

# --- 2. IMPACT CLASSIFICATION ---
def categorize_impact(row):
    """
    Categorizes how reasoning changed the outcome.
    """
    if not row['used_reasoning']:
        return "Skipped Reasoning"
    
    if row['initial_correct'] and row['is_correct']:
        return "Stayed Correct"
    if not row['initial_correct'] and not row['is_correct']:
        return "Stayed Wrong"
    if not row['initial_correct'] and row['is_correct']:
        return "Reasoning Corrected ✅"
    if row['initial_correct'] and not row['is_correct']:
        return "Reasoning Ruined ❌"
    return "Unknown"

# --- 3. MAIN EXECUTION ---
def main():
    LOG_FILE = "out-tuned.txt"
    
    print(f"📂 Loading logs from {LOG_FILE}...")
    df = parse_ollama_logs(LOG_FILE)
    
    if df.empty:
        print(f"❌ No data parsed. Ensure '{LOG_FILE}' exists and matches the format.")
        return

    df['impact'] = df.apply(categorize_impact, axis=1)

    # --- Print Summary Stats ---
    print("\n" + "="*35)
    print("      CSE 545 PROJECT SUMMARY")
    print("="*35)
    summary = {
        "Total Questions": len(df),
        "Overall Accuracy": f"{df['is_correct'].mean():.2%}",
        "Reasoning Trigger Rate": f"{df['used_reasoning'].mean():.2%}",
        "Avg Margin (Correct)": f"{df[df['is_correct']]['log_diff'].mean():.2f}",
        "Avg Margin (Wrong)": f"{df[~df['is_correct']]['log_diff'].mean():.2f}",
        "Avg Total Output Tokens": f"{df['total_output_tokens'].mean():.1f}",
        "Avg Tokens (Reasoning)": f"{df[df['used_reasoning']]['total_output_tokens'].mean():.1f}" if df['used_reasoning'].any() else "N/A",
    }
    for k, v in summary.items():
        print(f"{k:<25}: {v}")

    # --- PLOT 1: MARGIN DISTRIBUTION ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['is_correct']], x='log_diff', fill=True, color='#2ecc71', label='Correct Answers', alpha=0.5)
    sns.kdeplot(data=df[~df['is_correct']], x='log_diff', fill=True, color='#e74c3c', label='Wrong Answers', alpha=0.5)
    plt.axvline(x=10, color='#34495e', linestyle='--', label='Threshold (10.0)')
    plt.title("Is Log-Diff a Reliable Signal of Correctness?", fontsize=14)
    plt.xlabel("Log-Diff (Margin)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("margin_reliability_density.png")
    print("\n📈 Saved: margin_reliability_density.png")

    # --- PLOT 2: REASONING IMPACT BAR CHART ---
    plt.figure(figsize=(10, 6))
    # Filter only for questions where the model actually invoked reasoning
    reasoning_df = df[df['used_reasoning']].copy()
    
    if not reasoning_df.empty:
        order = ["Stayed Correct", "Stayed Wrong", "Reasoning Corrected ✅", "Reasoning Ruined ❌"]
        sns.countplot(data=reasoning_df, x='impact', palette='viridis', order=order)
        plt.title("The Value of Chain-of-Thought (CoT) Reasoning", fontsize=14)
        plt.ylabel("Count of Questions", fontsize=12)
        plt.xlabel("Outcome Change", fontsize=12)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig("cot_impact_analysis.png")
        print("📈 Saved: cot_impact_analysis.png")
    else:
        print("⚠️ No reasoning triggers found; skipping impact chart.")

    # --- PLOT 3: ACCURACY PER LOG-DIFF BIN ---
    bins = [0, 1, 2, 5, 10, 15, 25, 100]
    df['bin'] = pd.cut(df['log_diff'], bins=bins)
    bin_acc = df.groupby('bin', observed=True)['is_correct'].mean()
    
    plt.figure(figsize=(10, 6))
    bin_acc.plot(kind='bar', color='#3498db', alpha=0.8, edgecolor='black')
    plt.axhline(y=0.25, color='red', linestyle=':', label='Random Guessing (25%)')
    plt.title("Calibration: Confidence (Log-Diff) vs. Actual Accuracy", fontsize=14)
    plt.xlabel("Log-Diff Bin", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_calibration_bins.png")
    print("📈 Saved: accuracy_calibration_bins.png")

    # Final Data Export
    df.to_csv("experiment_analysis_final.csv", index=False)
    print("\n💾 Full CSV exported to experiment_analysis_final.csv")

if __name__ == "__main__":
    main()