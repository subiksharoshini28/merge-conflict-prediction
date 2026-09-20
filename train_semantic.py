"""
Option 2 model harness - predict SEMANTIC (compile-verified) conflicts.

Assembles the option-2 dataset by joining, on (repo, merge):
    labels      <- semantic.csv    (status clean=0 / SEMANTIC_CONFLICT=1; skips 'skip'/'textual')
    git feats   <- all.csv
    complexity  <- complexity.csv  (cbo, cyclomatic, dep_depth)
    graph feats <- graph_semantic.csv  (optional; used if present)

Then reports the semantic-conflict class counts and, IF there are enough positives,
compares feature sets (GIT / COMPLEXITY / GRAPH / ALL) under leave-repo-out + stratified CV,
scoring precision/recall/F1/PR-AUC on the conflict class. Degrades gracefully when
positives are too few (the usual case for stable libs) - then it points to the seeded
experiment as the claim-3 evidence instead of over-fitting a handful of points.

Usage:  python train_semantic.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]
COMPLEXITY = ["cbo_p1", "cbo_p2", "cyclomatic_p1", "cyclomatic_p2", "dep_depth_p1", "dep_depth_p2"]
GRAPH = ["g_nodes", "g_edges", "g_changed_p1", "g_changed_p2", "g_fanin_p1", "g_fanout_p1",
         "g_fanin_p2", "g_fanout_p2", "cross_edges", "shared_neighbors",
         "min_graph_distance", "shared_modules"]


def load_join():
    sem = pd.read_csv("semantic.csv")
    sem = sem[sem["status"].isin(["clean", "SEMANTIC_CONFLICT"])].copy()
    sem["y"] = (sem["status"] == "SEMANTIC_CONFLICT").astype(int)
    df = sem[["repo", "merge", "y"]]

    git = pd.read_csv("all.csv")[["repo", "merge"] + GIT]
    df = df.merge(git, on=["repo", "merge"], how="left")
    if os.path.exists("complexity.csv"):
        df = df.merge(pd.read_csv("complexity.csv"), on=["repo", "merge"], how="left")
    have_graph = os.path.exists("graph_semantic.csv")
    if have_graph:
        gcols = ["repo", "merge"] + [c for c in GRAPH if c in pd.read_csv("graph_semantic.csv", nrows=0).columns]
        df = df.merge(pd.read_csv("graph_semantic.csv")[gcols], on=["repo", "merge"], how="left")
    return df, have_graph


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=0, n_jobs=-1)


def evaluate(df, cols, splitter, groups=None):
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    y = df["y"].to_numpy(int)
    proba = np.zeros(len(y))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = rf(); m.fit(X[tr], y[tr]); proba[te] = m.predict_proba(X[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0), average_precision_score(y, proba), tp, fp, fn, tn)


def main():
    if not os.path.exists("semantic.csv"):
        print("semantic.csv not found - run semantic_label.py first."); return
    df, have_graph = load_join()
    n, pos = len(df), int(df["y"].sum())
    print("=" * 66)
    print("OPTION-2 SEMANTIC-CONFLICT DATASET")
    print("=" * 66)
    print(f"labelled scenarios : {n}   (clean + semantic-conflict, excludes skip/textual)")
    print(f"semantic conflicts : {pos}  ({pos/n:.1%})" if n else "no rows")
    print(f"by repo:\n{df.groupby('repo')['y'].agg(['count','sum']).to_string()}")

    sets = {"GIT": GIT, "COMPLEXITY": COMPLEXITY}
    if have_graph:
        sets["GRAPH"] = GRAPH
        sets["ALL"] = GIT + COMPLEXITY + GRAPH
    else:
        sets["GIT+COMPLEXITY"] = GIT + COMPLEXITY
        print("\n(graph_semantic.csv not present - GRAPH set skipped; compute it to include graph features)")

    if pos < 6:
        print(f"\nOnly {pos} positive(s): too few for a reliable model. Not training —")
        print("the SEEDED experiment (seeded.csv) is the controlled evidence for claim 3.")
        print("If real positives stay low, report 'semantic conflicts are rare in stable")
        print("libraries' as a finding and lean on the seeded proof + graph mechanism.")
        return

    nrepos = df["repo"].nunique()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print("\n### Stratified 5-fold CV (semantic-conflict class: P / R / F1 / PR-AUC) ###")
    for name, cols in sets.items():
        p, r, f, ap, tp, fp, fn, tn = evaluate(df, cols, skf)
        print(f"  {name:<16} P {p:.3f}  R {r:.3f}  F1 {f:.3f}  PR-AUC {ap:.3f}  (TP{tp}/FP{fp}/FN{fn})")
    if nrepos >= 2:
        gkf = GroupKFold(n_splits=min(nrepos, 5)); groups = df["repo"].to_numpy()
        print("\n### Leave-repo-out CV ###")
        for name, cols in sets.items():
            p, r, f, ap, tp, fp, fn, tn = evaluate(df, cols, gkf, groups)
            print(f"  {name:<16} P {p:.3f}  R {r:.3f}  F1 {f:.3f}  PR-AUC {ap:.3f}  (TP{tp}/FP{fp}/FN{fn})")


if __name__ == "__main__":
    main()
