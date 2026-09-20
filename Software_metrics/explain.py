"""
Phase 4 (part 1) - EXPLAINABILITY for the tabular conflict predictor.

Offline, no external deps beyond sklearn. Two layers:
  GLOBAL  - permutation importance: which features drive conflict prediction overall
            (model-agnostic; more reliable than impurity importances).
  LOCAL   - per-scenario "why": counterfactual attribution - for a flagged merge,
            replace each feature with its typical CLEAN-merge value and measure how
            much the conflict probability drops. Big drop => that feature is a driver.
            Each top driver is mapped to a RECOMMENDED ACTION.

Leakage rules unchanged: conflict_files / code_conflict are never features.
Uses graph features automatically if present (all_graph.csv), else git-only (all.csv).

Usage:  python explain.py [all_graph.csv | all.csv]
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]
GRAPH = ["g_nodes", "g_edges", "g_changed_p1", "g_changed_p2",
         "g_fanin_p1", "g_fanout_p1", "g_fanin_p2", "g_fanout_p2",
         "cross_edges", "shared_neighbors", "min_graph_distance", "shared_modules"]

# Map a driving feature to a human recommended action.
ACTION = {
    "overlap_files":  "Both branches edit the same file(s) - coordinate on them or merge the smaller branch first.",
    "overlap_ratio":  "High share of overlapping files - split the work or integrate early in small steps.",
    "churn_p1":       "Branch 1 changes a large volume of code - merge it sooner / in smaller increments.",
    "churn_p2":       "Branch 2 changes a large volume of code - merge it sooner / in smaller increments.",
    "files_p1":       "Branch 1 touches many files - review integration surface early.",
    "files_p2":       "Branch 2 touches many files - review integration surface early.",
    "commits_p1":     "Branch 1 has diverged over many commits - rebase/integrate more frequently.",
    "commits_p2":     "Branch 2 has diverged over many commits - rebase/integrate more frequently.",
    "authors_p1":     "Many contributors on branch 1 - align via code owners before merging.",
    "authors_p2":     "Many contributors on branch 2 - align via code owners before merging.",
    "cross_edges":    "The two branches edit DEPENDENT files (import links) though not the same file - coordinate across the dependency.",
    "shared_neighbors":"Both branches depend on common files - a shared component couples them; review it jointly.",
    "min_graph_distance":"Changes are close in the dependency graph - a small architectural distance raises interaction risk.",
    "shared_modules": "Both branches modify the same module/package - align design within that module.",
    "g_fanin_p1": "Branch 1 edits highly depended-on files - changes ripple widely; test dependents.",
    "g_fanin_p2": "Branch 2 edits highly depended-on files - changes ripple widely; test dependents.",
    "g_fanout_p1": "Branch 1 edits files with many dependencies - fragile to upstream change.",
    "g_fanout_p2": "Branch 2 edits files with many dependencies - fragile to upstream change.",
}


def risk_band(p):
    """Map a conflict probability to a discrete risk score (the framework's user-facing output)."""
    return "HIGH" if p >= 0.50 else "MEDIUM" if p >= 0.20 else "LOW"


def load(path):
    df = pd.read_csv(path)
    feats = GIT + [c for c in GRAPH if c in df.columns]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df, feats


def local_explain(model, x, clean_baseline, feats, k=4):
    """Counterfactual attribution: prob drop when each feature -> typical clean value."""
    p_full = model.predict_proba(x.reshape(1, -1))[0, 1]
    contribs = []
    for i, f in enumerate(feats):
        xc = x.copy(); xc[i] = clean_baseline[i]
        drop = p_full - model.predict_proba(xc.reshape(1, -1))[0, 1]
        contribs.append((f, drop, x[i], clean_baseline[i]))
    contribs.sort(key=lambda t: t[1], reverse=True)
    return p_full, contribs[:k]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "all_graph.csv" if os.path.exists("all_graph.csv") else "all.csv")
    df, feats = load(path)
    X = df[feats].to_numpy(float); y = df["label"].to_numpy(int)
    print("=" * 66)
    print(f"EXPLAINABILITY  |  {path}  |  {len(df)} scenarios, {int(y.sum())} conflicts")
    print(f"features ({len(feats)}): {'git+graph' if len(feats) > len(GIT) else 'git only'}")
    print("=" * 66)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, y)

    # ---- GLOBAL ----
    print("\n### GLOBAL feature importance (permutation, drop in avg-precision) ###")
    pi = permutation_importance(model, X, y, scoring="average_precision",
                                n_repeats=20, random_state=0, n_jobs=-1)
    order = np.argsort(pi.importances_mean)[::-1]
    for i in order:
        if pi.importances_mean[i] <= 0:
            continue
        bar = "#" * max(1, int(pi.importances_mean[i] / pi.importances_mean[order[0]] * 30))
        print(f"  {feats[i]:<18}{pi.importances_mean[i]:.4f}  {bar}")

    # ---- LOCAL ----
    clean_baseline = np.median(X[y == 0], axis=0)
    proba = model.predict_proba(X)[:, 1]

    # ---- RISK SCORE distribution (Low / Medium / High banding) ----
    bands = [risk_band(p) for p in proba]
    print("\n### RISK SCORE banding (probability -> Low/Medium/High) ###")
    for b, lo, hi in [("HIGH", 0.50, 1.0), ("MEDIUM", 0.20, 0.50), ("LOW", 0.0, 0.20)]:
        idx = [i for i, bb in enumerate(bands) if bb == b]
        conf = int(sum(y[i] for i in idx))
        print(f"  {b:<7} [{lo:.2f}-{hi:.2f}): {len(idx):>4} scenarios, {conf} actual conflicts"
              f"  ({conf/len(idx):.0%} hit-rate)" if idx else f"  {b:<7}: 0")

    risky = np.argsort(proba)[::-1][:3]     # 3 highest-risk scenarios to explain
    print("\n### LOCAL explanations for the highest-risk merges (why + action) ###")
    for idx in risky:
        row = df.iloc[idx]
        p, drivers = local_explain(model, X[idx], clean_baseline, feats)
        actual = "CONFLICT" if y[idx] == 1 else "clean"
        print(f"\n  {row['repo']} @ {row['merge'][:10]}  |  conflict prob = {p:.2f}"
              f"  |  RISK: {risk_band(p)}  |  actual: {actual}")
        for f, drop, val, base in drivers:
            if drop <= 0:
                continue
            print(f"    - {f} = {val:g} (typical clean {base:g})  ->  {ACTION.get(f, 'notable driver')}")

    print("\nNote: SHAP/GNNExplainer can be layered on later for exact additive")
    print("attributions; this counterfactual method needs no extra dependencies.")


if __name__ == "__main__":
    main()
