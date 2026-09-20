"""
Multi-model train/test comparison on the merge-conflict dataset.

Usage:  python train_models.py [all.csv]
"""
import os
import sys
import warnings

# Suppress joblib/CPU warnings before any imports
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")

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

DIVIDER = "=" * 80


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
            min_samples_leaf=2, random_state=0, n_jobs=1),
        "ExtraTrees (balanced)": lambda: ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=0, n_jobs=1),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=0),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=0),
    }


def oof_predict(model_fn, X, y, splitter, groups=None):
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


def print_header():
    print(DIVIDER)
    print("  MERGE CONFLICT PREDICTION — MODEL TRAINING & EVALUATION")
    print(DIVIDER)


def print_dataset_info(path, n, pos, n_repos):
    print(f"\n  Dataset          : {path}")
    print(f"  Total scenarios  : {n}")
    print(f"  Conflicts       : {pos} ({pos/n:.1%})")
    print(f"  Clean            : {n - pos} ({(n-pos)/n:.1%})")
    print(f"  Repositories     : {n_repos}")
    print(f"  Features         : {len(FEATURES)}")
    print(f"  Base rate        : {pos/n:.3f} (trivial 'always clean' = {1-pos/n:.1%} acc)")
    print()


def print_protocol_header(proto):
    print(DIVIDER)
    print(f"  PROTOCOL: {proto.upper()}")
    if proto == "stratified-5fold":
        print("  (5 folds, same distribution — easy evaluation)")
    else:
        print("  (unseen repos — honest generalization test)")
    print(DIVIDER)
    print()


def print_results_table(results):
    # Header
    print(f"  {'Model':<26} {'Precision':>9} {'Recall':>8} {'F1':>6} {'PR-AUC':>7} {'Accuracy':>9}   {'Confusion Matrix'}")
    print(f"  {'-'*26} {'-'*9} {'-'*8} {'-'*6} {'-'*7} {'-'*9}   {'-'*20}")
    
    # Rows sorted by F1
    for s in sorted(results, key=lambda r: -r['f1']):
        cm = f"TP={s['tp']:>3} FP={s['fp']:>3} FN={s['fn']:>3} TN={s['tn']:>4}"
        print(f"  {s['model']:<26} {s['precision']:>9.3f} {s['recall']:>8.3f} "
              f"{s['f1']:>6.3f} {s['pr_auc']:>7.3f} {s['acc']:>9.3f}   {cm}")
    print()


def print_headline(strict, best, pos, n):
    print(DIVIDER)
    print("  HEADLINE RESULTS")
    print(DIVIDER)
    print(f"  Evaluation protocol : {strict}")
    print(f"  Best model          : {best['model']}")
    print()
    print(f"  +{'-'*38}+")
    print(f"  | {'Metric':<20} | {'Value':>12} |")
    print(f"  +{'-'*38}+")
    print(f"  | {'Precision':<20} | {best['precision']:>12.3f} |")
    print(f"  | {'Recall':<20} | {best['recall']:>12.3f} |")
    print(f"  | {'F1-Score':<20} | {best['f1']:>12.3f} |")
    print(f"  | {'PR-AUC':<20} | {best['pr_auc']:>12.3f} |")
    print(f"  | {'Accuracy':<20} | {best['acc']:>12.3f} |")
    print(f"  +{'-'*38}+")
    print()
    print(f"  PR-AUC {best['pr_auc']:.3f} vs base rate {pos/n:.3f} = "
          f"{best['pr_auc']/(pos/n):.1f}x better than random chance")
    print(DIVIDER)


def print_confusion_detail(best):
    print(f"\n  Confusion Matrix for {best['model']}:")
    print(f"  +{'-'*30}+")
    print(f"  |                      | Predicted Clean | Predicted Conflict |")
    print(f"  +{'-'*30}+")
    print(f"  | Actual Clean         |    {best['tn']:>6}        |       {best['fp']:>4}         |")
    print(f"  | Actual Conflict      |    {best['fn']:>6}        |       {best['tp']:>4}         |")
    print(f"  +{'-'*30}+")
    print(f"\n  Interpretation:")
    print(f"    - Caught {best['tp']} conflicts correctly (True Positives)")
    print(f"    - Missed {best['fn']} conflicts (False Negatives)")
    print(f"    - Falsely flagged {best['fp']} clean merges (False Positives)")
    print(f"    - Correctly identified {best['tn']} clean merges (True Negatives)")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    
    print_header()
    
    # Step 1: Load data
    print("  [1/4] Loading dataset...")
    df = load(path)
    n, pos = len(df), int(df[TARGET].sum())
    n_repos = df["repo"].nunique()
    print(f"        Done.\n")
    
    # Step 2: Dataset info
    print("  [2/4] Dataset Summary:")
    print_dataset_info(path, n, pos, n_repos)
    
    # Step 3: Prepare features
    print("  [3/4] Preparing features...")
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    groups = df["repo"].to_numpy()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ng = df["repo"].nunique()
    gkf = GroupKFold(n_splits=min(ng, 5))
    print(f"        Features shape: {X.shape}")
    print(f"        Stratified 5-fold: ready")
    print(f"        Leave-repo-out ({ng} groups): ready\n")
    
    # Step 4: Train and evaluate
    print("  [4/4] Training models...\n")
    
    protocols = [("stratified-5fold", skf, None)]
    if ng >= 2:
        protocols.append(("leave-repo-out", gkf, groups))
    
    rows = []
    for proto, splitter, grp in protocols:
        print_protocol_header(proto)
        model_results = []
        for name, fn in models().items():
            proba, pred = oof_predict(fn, X, y, splitter, grp)
            s = score(y, pred, proba)
            s.update(model=name, protocol=proto)
            rows.append(s)
            model_results.append(s)
        print_results_table(model_results)
    
    # Headline
    strict = "leave-repo-out" if ng >= 2 else "stratified-5fold"
    best = max([r for r in rows if r['protocol'] == strict], key=lambda r: r['f1'])
    print_headline(strict, best, pos, n)
    print_confusion_detail(best)


if __name__ == "__main__":
    main()
