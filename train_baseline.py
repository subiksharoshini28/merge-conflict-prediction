"""
Phase 2 - baseline conflict predictor on the flat git-history features.

Rules baked in (see CLAUDE.md):
- NO LEAKAGE: `conflict_files` and `code_conflict` describe the merge OUTCOME,
  so they are NOT features. Only pre-merge git-history signals are used.
- Class imbalance is expected -> class_weight='balanced'.
- Judge on PRECISION / RECALL / F1 (+ PR-AUC) for the CONFLICT class.
  Accuracy is printed only as a foil; a predict-clean model scores 90%+ and is useless.
- Honest evaluation: out-of-fold predictions via StratifiedKFold, so every row is
  scored by a model that never saw it. Grouped-by-repo CV is also reported to check
  the model generalizes to an unseen repo (harder, more realistic).

Usage:  python train_baseline.py [all.csv]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix, accuracy_score)

FEATURES = ["commits_p1", "commits_p2", "files_p1", "files_p2",
            "overlap_files", "overlap_ratio", "authors_p1", "authors_p2",
            "churn_p1", "churn_p2"]
# Explicitly NOT features (leakage / identifiers): conflict_files, code_conflict, repo, merge, label
TARGET = "label"


def load(path):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df


def report(name, y, pred, proba):
    p = precision_score(y, pred, zero_division=0)
    r = recall_score(y, pred, zero_division=0)
    f = f1_score(y, pred, zero_division=0)
    ap = average_precision_score(y, proba)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\n  [{name}]  conflict-class metrics")
    print(f"    precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}   PR-AUC {ap:.3f}")
    print(f"    confusion: TP={tp} FP={fp} FN={fn} TN={tn}   (accuracy {accuracy_score(y, pred):.3f} <- ignore)")
    return dict(precision=p, recall=r, f1=f, pr_auc=ap)


def oof_predict(model_fn, X, y, splitter, groups=None):
    """Out-of-fold predictions: each row scored by a fold-model that never trained on it."""
    proba = np.zeros(len(y), dtype=float)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = model_fn()
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba, (proba >= 0.5).astype(int)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    df = load(path)
    n, pos = len(df), int(df[TARGET].sum())
    print("=" * 64)
    print(f"DATASET  {path}")
    print(f"  scenarios       {n}")
    print(f"  conflicts (1)   {pos}   ({pos / n:.1%})")
    print(f"  clean     (0)   {n - pos}   ({(n - pos) / n:.1%})")
    print(f"  repos           {df['repo'].nunique()} -> {sorted(df['repo'].unique())}")
    if "code_conflict" in df.columns:
        cc = int(df["code_conflict"].sum())
        print(f"  code_conflict=1 {cc}  (Java-source conflicts; label refinement, NOT a feature)")
    print("=" * 64)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    groups = df["repo"].to_numpy()

    models = {
        "LogReg (balanced)": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest (balanced)": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=0, n_jobs=-1),
    }

    print("\n### Stratified 5-fold CV (random split) ###")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, fn in models.items():
        proba, pred = oof_predict(fn, X, y, skf)
        report(name, y, pred, proba)

    # Harder, more realistic: never train and test on the same repo.
    ngroups = df["repo"].nunique()
    if ngroups >= 2:
        print(f"\n### Leave-repo-out CV (GroupKFold, {min(ngroups,5)} folds) — generalization to an UNSEEN repo ###")
        gkf = GroupKFold(n_splits=min(ngroups, 5))
        for name, fn in models.items():
            proba, pred = oof_predict(fn, X, y, gkf, groups=groups)
            report(name, y, pred, proba)

    print("\nNote: baseline uses flat git features only. Phase 3 adds dependency-graph")
    print("features expected to lift recall on conflicts that share NO file.")


if __name__ == "__main__":
    main()
