import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # <--- The library!
import matplotlib.pyplot as plt

# 1. Load the validation data you collected earlier
df_val = pd.read_csv('medqa_val_probe_results.csv')

# 2. Define Features (X) and Target (y)
# We use the Margin to predict if the 'Probe' was correct
X = df_val[['margin']].values 
y = df_val['is_correct'].values

# 3. Initialize and Train the Model
# This finds the best-fitting line (Sigmoid curve) for your data
clf = LogisticRegression()
clf.fit(X, y)

# 4. Extract the Coefficients
beta_0 = clf.intercept_[0] # The intercept
beta_1 = clf.coef_[0][0]    # The "weight" of the Log-Diff

print(f"Logistic Regression Intercept (B0): {beta_0:.4f}")
print(f"Logistic Regression Coefficient (B1): {beta_1:.4f}")

# 5. Find the Mathematical Threshold
# Let's say we want to trigger reasoning if Probability(Correct) < 80%
target_p = 0.80
logit_p = np.log(target_p / (1 - target_p))
optimal_threshold = (logit_p - beta_0) / beta_1

print(f"\n--- CONCLUSION FOR REPORT ---")
print(f"To ensure 80% confidence, set Log-Diff threshold to: {optimal_threshold:.2f}")