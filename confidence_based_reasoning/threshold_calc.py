import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your local CSV
df = pd.read_csv('medqa_val_probe_results.csv')

# --- PLOT 1: ACCURACY BY BIN ---
# Group margins into readable categories
bins = [0, 0.5, 1, 2, 3, 5, 10, 20]
df['margin_bin'] = pd.cut(df['margin'], bins=bins)
bin_accuracy = df.groupby('margin_bin', observed=True)['is_correct'].mean()

plt.figure(figsize=(10, 6))
bin_accuracy.plot(kind='bar', color='#3498db', edgecolor='black', alpha=0.8)
plt.axhline(y=0.25, color='red', linestyle='--', label='Random Chance (25%)')
plt.title('Accuracy increases as Log-Diff (Margin) grows', fontsize=14)
plt.xlabel('Log-Diff Range', fontsize=12)
plt.ylabel('Fraction Correct', fontsize=12)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('report_accuracy_bins.png')

# --- PLOT 2: DISTRIBUTION DENSITY ---
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['is_correct'] == 1], x='margin', fill=True, color='green', label='Correct', alpha=0.4)
sns.kdeplot(data=df[df['is_correct'] == 0], x='margin', fill=True, color='red', label='Incorrect', alpha=0.4)
plt.title('Where does the model fail? (Distribution of Log-Diff)', fontsize=14)
plt.xlabel('Log-Diff (Margin)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('report_margin_density.png')

# --- PLOT 3: THE TRADE-OFF CURVE ---
thresholds = np.linspace(0, 12, 100)
acc_above = []
pct_sent_to_reasoning = []

for t in thresholds:
    # Accuracy of the questions we DON'T reason about
    subset = df[df['margin'] >= t]
    acc = subset['is_correct'].mean() if len(subset) > 0 else 1.0
    acc_above.append(acc)
    # % of data where margin < t
    pct_sent_to_reasoning.append((df['margin'] < t).mean() * 100)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thresholds, acc_above, color='blue', label='Probe Accuracy (Above Threshold)', linewidth=2)
ax1.set_xlabel('Log-Diff Threshold ($\tau$)', fontsize=12)
ax1.set_ylabel('Accuracy of Fast Model', color='blue', fontsize=12)

ax2 = ax1.twinx()
ax2.fill_between(thresholds, pct_sent_to_reasoning, color='orange', alpha=0.2, label='% Sent to Reasoning')
ax2.set_ylabel('% of Questions Triggering Reasoning', color='orange', fontsize=12)

plt.title('Trading Efficiency for Accuracy', fontsize=14)
plt.savefig('report_tradeoff_curve.png')

print("✅ Visualizations generated: report_accuracy_bins.png, report_margin_density.png, report_tradeoff_curve.png")