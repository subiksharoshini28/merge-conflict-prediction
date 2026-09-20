"""
Full honest evaluation: does adding historical features raise F1 — under BOTH protocols?

Feature sets:  GIT (10)  ·  HIST (5)  ·  GIT+HIST  ·  ALL (+complexity/graph if full-coverage)
Protocols:     stratified 5-fold (within-project)  ·  leave-one-repository-out (cross-project)
Metric:        precision / recall / F1 / PR-AUC on the conflict class, threshold 0.5 (no gaming).

Joins historical.csv (and complexity/graph if present) onto all.csv by (repo, merge).
Usage:  python train_full.py [all.csv]
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]
HIST = ["hist_conflict_rate", "hist_prior_conflicts", "hist_prior_merges",
        "file_conflict_overlap", "divergence_days"]
COMPLEX = ["cbo_p1", "cbo_p2", "cyclomatic_p1", "cyclomatic_p2", "dep_depth_p1", "dep_depth_p2"]
GRAPH = ["g_nodes", "g_edges", "cross_edges", "shared_neighbors", "min_graph_distance",
         "shared_modules", "g_fanin_p1", "g_fanout_p1", "g_fanin_p2", "g_fanout_p2"]


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=0, n_jobs=-1)


def load():
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "all.csv")
    def join(fn, cols):
        if os.path.exists(fn):
            add = pd.read_csv(fn)
            keep = ["repo", "merge"] + [c for c in cols if c in add.columns]
            return df.merge(add[keep], on=["repo", "merge"], how="left"), [c for c in cols if c in add.columns]
        return df, []
    df, hist = join("historical.csv", HIST)
    df, comp = join("complexity.csv", COMPLEX)
    df, graph = join("graph_semantic.csv", GRAPH)
    all_cols = GIT + hist + comp + graph
    df[all_cols] = df[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df, hist, comp, graph


def oof(df, cols, splitter, groups=None):
    X = df[cols].to_numpy(float); y = df["label"].to_numpy(int)
    proba = np.zeros(len(y))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = rf(); m.fit(X[tr], y[tr]); proba[te] = m.predict_proba(X[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0), average_precision_score(y, proba), tp, fp, fn, tn)


def main():
    df, hist, comp, graph = load()
    y = df["label"].to_numpy(int)
    print("=" * 82)
    print(f"FULL EVALUATION  |  {len(df)} scenarios, {int(y.sum())} conflicts, {df['repo'].nunique()} repos")
    print(f"feature groups available -> git:{len(GIT)}  hist:{len(hist)}  complexity:{len(comp)}  graph:{len(graph)}")
    print("=" * 82)
    sets = {"GIT": GIT}
    if hist: sets["HIST only"] = hist; sets["GIT+HIST"] = GIT + hist
    if comp or graph: sets["ALL"] = GIT + hist + comp + graph
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    gkf = GroupKFold(n_splits=df["repo"].nunique())
    groups = df["repo"].to_numpy()
    for proto, sp, grp in [("STRATIFIED 5-fold (within-project)", skf, None),
                           ("LEAVE-ONE-REPO-OUT (cross-project)", gkf, groups)]:
        print(f"\n### {proto} ###")
        print(f"{'feature set':<14}{'P':>7}{'R':>7}{'F1':>7}{'PR-AUC':>8}   TP/FP/FN")
        for name, cols in sets.items():
            p, r, f, ap, tp, fp, fn, tn = oof(df, cols, sp, grp)
            print(f"{name:<14}{p:>7.3f}{r:>7.3f}{f:>7.3f}{ap:>8.3f}   {tp}/{fp}/{fn}")
    print("\nHonest headline = LEAVE-REPO-OUT F1. Within-project (stratified) is legitimately higher")
    print("and reflects a per-project CI-assistant use case — report both, labelled clearly.")


if __name__ == "__main__":
    main()
