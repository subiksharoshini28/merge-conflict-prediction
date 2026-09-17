"""
Concrete BEFORE -> AFTER for every preprocessing step, on the real dataset.
Usage:  python preprocessing_report.py
"""
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import conflict_predictor as cp

GIT = cp.FEATURES
df = pd.read_csv("all.csv")

print("=" * 70)
print(" DATA PREPROCESSING  —  BEFORE  vs  AFTER")
print("=" * 70)

# ---- 1. Data cleaning: trivial-merge filter ----
print("\n[1] DATA CLEANING  (drop trivial merges: commits_p1==0 or commits_p2==0)")
try:
    raw = subprocess.run(["git", "-C", "repos/commons-lang", "log", "--merges",
                          "--min-parents=2", "--max-parents=2", "--format=%H"],
                         capture_output=True, text=True).stdout.split()
    kept = int(((df["repo"] == "commons-lang")).sum())
    print(f"    example (commons-lang):   BEFORE {len(raw)} two-parent merges  ->  AFTER {kept} kept")
except Exception:
    pass
print(f"    whole dataset:            raw merges mined  ->  AFTER filter = {len(df)} usable scenarios")

# ---- 2. Type coercion + missing-value imputation ----
print("\n[2] TYPE COERCION + MISSING-VALUE IMPUTATION  (to_numeric + fillna(0))")
before = df[GIT]
coerced = before.apply(pd.to_numeric, errors="coerce")
n_bad = int(coerced.isna().sum().sum())
print(f"    non-numeric / empty cells in feature columns:  BEFORE {n_bad}  ->  AFTER 0 (filled)")
print(f"    dtypes:  BEFORE mixed/object-safe  ->  AFTER all numeric float")

# ---- 3. Class-imbalance handling (SMOTE) ----
y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
X = coerced.fillna(0).to_numpy(float)
pos, neg = int(y.sum()), int((y == 0).sum())
Xa, ya = cp.smote_balance(X, y)
print("\n[3] CLASS-IMBALANCE HANDLING  (SMOTE, training set)")
print(f"    conflict : clean          BEFORE  {pos} : {neg}  (1 : {neg/pos:.1f})")
print(f"                              AFTER   {int(ya.sum())} : {int((ya==0).sum())}  (1 : 1, balanced)")
print(f"    total training rows       BEFORE  {len(y)}  ->  AFTER {len(ya)}  (+{len(ya)-len(y)} synthetic minority)")

# ---- 4. Feature scaling (for the linear model) ----
print("\n[4] FEATURE SCALING  (StandardScaler, used for Logistic Regression)")
f = "churn_p1"
col = X[:, GIT.index(f)]
sc = StandardScaler().fit(col.reshape(-1, 1)).transform(col.reshape(-1, 1)).ravel()
print(f"    feature '{f}':")
print(f"      BEFORE   min {col.min():.0f}   max {col.max():.0f}   mean {col.mean():.1f}   std {col.std():.1f}")
print(f"      AFTER    min {sc.min():.2f}   max {sc.max():.2f}   mean {sc.mean():.2f}   std {sc.std():.2f}")
print("      (Random Forest is scale-invariant, so scaling is applied only for the linear baseline.)")
print("=" * 70)
