"""
Multi-model train/test comparison on the merge-conflict dataset.

Extends train_baseline.py with more classifiers and a single ranked summary
table, so the framework's train/test performance can be read at a glance.

Same rules (see CLAUDE.md):
- NO LEAKAGE: only pre-merge git-history features; conflict_files / code_conflict
  (merge outcome) and repo / merge / label are never inputs.
- Class imbalance -> class_weight='balanced' where the model supports it.
- Judge on PRECISION / RECALL / F1 (+ PR-AUC) for the CONFLICT class; accuracy is
  a foil only.
- Honest evaluation: out-of-fold predictions. Stratified 5-fold AND leave-one-
  repository-out (generalization to an unseen project) are both reported.

Usage:  python train_models.py [all.csv]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix, accuracy_score)

FEATURES = ["commits_p1", "commits_p2", "files_p1", "files_p2",
            "overlap_files", "overlap_ratio", "authors_p1", "authors_p2",
            "churn_p1", "churn_p2"]
TARGET = "label"


def load(path):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df


def models():
    return {
        "LogReg (balanced)": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest (balanced)": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=0, n_jobs=-1),
        "ExtraTrees (balanced)": lambda: ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=0, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=0),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=0),
    }


def oof_predict(model_fn, X, y, splitter, groups=None):
    """Out-of-fold predictions: each row scored by a fold-model that never trained on it."""
    proba = np.zeros(len(y), dtype=float)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = model_fn()
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba, (proba >= 0.5).astype(int)


def score(y, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        pr_auc=average_precision_score(y, proba),
        acc=accuracy_score(y, pred),
        tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    df = load(path)
    n, pos = len(df), int(df[TARGET].sum())
    print("=" * 78)
    print(f"DATASET  {path}   |   {n} scenarios, {pos} conflicts ({pos/n:.1%}), "
          f"{df['repo'].nunique()} repos")
    print(f"base rate {pos/n:.3f}   (a trivial 'always clean' model scores "
          f"{1-pos/n:.1%} accuracy and is useless)")
    print("=" * 78)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    groups = df["repo"].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ng = df["repo"].nunique()
    gkf = GroupKFold(n_splits=min(ng, 5))

    protocols = [("stratified-5fold", skf, None)]
    if ng >= 2:
        protocols.append(("leave-repo-out", gkf, groups))

    rows = []
    for proto, splitter, grp in protocols:
        for name, fn in models().items():
            proba, pred = oof_predict(fn, X, y, splitter, grp)
            s = score(y, pred, proba)
            s.update(model=name, protocol=proto)
            rows.append(s)

    # ---- detailed per-protocol tables ----
    for proto, _, _ in protocols:
        print(f"\n### {proto} ###")
        print(f"  {'model':<26}{'P':>6}{'R':>7}{'F1':>7}{'PRAUC':>8}   confusion TP/FP/FN/TN")
        for s in sorted([r for r in rows if r['protocol'] == proto],
                        key=lambda r: -r['f1']):
            print(f"  {s['model']:<26}{s['precision']:>6.3f}{s['recall']:>7.3f}"
                  f"{s['f1']:>7.3f}{s['pr_auc']:>8.3f}   "
                  f"{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']}")

    # ---- headline: best under the strict (leave-repo-out) protocol ----
    strict = "leave-repo-out" if ng >= 2 else "stratified-5fold"
    best = max([r for r in rows if r['protocol'] == strict], key=lambda r: r['f1'])
    print("\n" + "=" * 78)
    print(f"HEADLINE ({strict}, the honest generalization number):")
    print(f"  best model = {best['model']}")
    print(f"  F1 {best['f1']:.3f} | precision {best['precision']:.3f} | "
          f"recall {best['recall']:.3f} | PR-AUC {best['pr_auc']:.3f}")
    print(f"  PR-AUC {best['pr_auc']:.3f} vs base rate {pos/n:.3f} "
          f"= {best['pr_auc']/(pos/n):.1f}x better than chance")
    print("=" * 78)


if __name__ == "__main__":
    main()
