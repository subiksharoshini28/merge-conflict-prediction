"""
Phase 3 evaluation - does the dependency graph beat the flat baseline?

Trains the SAME model on three feature sets and compares conflict-class F1 under
the honest leave-repo-out protocol (generalization to an unseen repo):
    GIT    - the 10 flat git-history features (the Phase-2 baseline)
    GRAPH  - the 12 dependency-graph features (Phase 3 novelty)
    ALL    - git + graph combined
The lift ALL - GIT under leave-repo-out is the paper/patent headline number.

Leakage rules unchanged: conflict_files / code_conflict are never features.

Usage:  python train_graph.py [all_graph.csv]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]
GRAPH = ["g_nodes", "g_edges", "g_changed_p1", "g_changed_p2",
         "g_fanin_p1", "g_fanout_p1", "g_fanin_p2", "g_fanout_p2",
         "cross_edges", "shared_neighbors", "min_graph_distance", "shared_modules"]
SETS = {"GIT (baseline)": GIT, "GRAPH (novelty)": GRAPH, "ALL (git+graph)": GIT + GRAPH}
TARGET = "label"


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=0, n_jobs=-1)


def logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def oof(model_fn, X, y, splitter, groups=None):
    proba = np.zeros(len(y))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = model_fn(); m.fit(X[tr], y[tr]); proba[te] = m.predict_proba(X[te])[:, 1]
    return proba, (proba >= 0.5).astype(int)


def metrics(y, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0),
            average_precision_score(y, proba), tp, fp, fn, tn)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all_graph.csv"
    df = pd.read_csv(path)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = df["repo"].to_numpy()
    n, pos = len(df), int(y.sum())
    print(f"dataset {path}: {n} scenarios, {pos} conflicts ({pos/n:.1%}), "
          f"{df['repo'].nunique()} repos")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    gkf = GroupKFold(n_splits=min(df['repo'].nunique(), 5))

    for model_name, make in [("RandomForest", rf), ("LogReg", logreg)]:
        print(f"\n{'='*70}\nMODEL: {model_name}   (conflict-class: precision / recall / F1 / PR-AUC)\n{'='*70}")
        print(f"{'feature set':<20}{'protocol':<16}{'P':>6}{'R':>7}{'F1':>7}{'PRAUC':>8}   confusion(TP/FP/FN/TN)")
        base_f1 = {}
        for proto, splitter, grp in [("stratified-5f", skf, None), ("leave-repo-out", gkf, groups)]:
            for set_name, cols in SETS.items():
                X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
                proba, pred = oof(make, X, y, splitter, grp)
                p, r, f, ap, tp, fp, fn, tn = metrics(y, pred, proba)
                tag = ""
                if proto == "leave-repo-out":
                    if set_name.startswith("GIT"):
                        base_f1[model_name] = f
                    elif set_name.startswith("ALL") and model_name in base_f1:
                        tag = f"   <= lift {f - base_f1[model_name]:+.3f} vs GIT"
                print(f"{set_name:<20}{proto:<16}{p:>6.3f}{r:>7.3f}{f:>7.3f}{ap:>8.3f}   {tp}/{fp}/{fn}/{tn}{tag}")
            print()

    print("Headline = leave-repo-out F1 of ALL minus GIT. Positive => the dependency")
    print("graph adds signal the flat git features miss (esp. no-shared-file conflicts).")


if __name__ == "__main__":
    main()
