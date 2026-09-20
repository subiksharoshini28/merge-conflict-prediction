"""
Tier-1 LEGITIMATE F1 improvement experiment (no leakage, no downloads).

Compares, under the honest leave-one-repository-out protocol:
  - feature sets:  GIT (10 raw)  vs  GIT+ENG (engineered ratios/totals/interactions)
  - models:        RandomForest  vs  HistGradientBoosting
  - threshold:     fixed 0.5      vs  TUNED (chosen on TRAIN only, via inner CV)

Rules kept:
  * Only pre-merge git features; conflict_files/code_conflict/repo/merge/label are never inputs.
  * Threshold is tuned on an inner split of the TRAINING repos, never on the held-out repo.
  * PR-AUC (threshold-free) reported alongside F1.

Everything runs in this folder. Usage:  python train_improve.py [all.csv]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]


def add_engineered(df):
    """Leakage-free features derived only from existing pre-merge columns."""
    e = pd.DataFrame(index=df.index)
    e["churn_total"] = df.churn_p1 + df.churn_p2
    e["churn_max"] = df[["churn_p1", "churn_p2"]].max(axis=1)
    e["churn_ratio"] = df[["churn_p1", "churn_p2"]].min(axis=1) / (e["churn_max"] + 1)
    e["files_total"] = df.files_p1 + df.files_p2
    e["files_ratio"] = df[["files_p1", "files_p2"]].min(axis=1) / (df[["files_p1", "files_p2"]].max(axis=1) + 1)
    e["commits_total"] = df.commits_p1 + df.commits_p2
    e["authors_total"] = df.authors_p1 + df.authors_p2
    e["overlap_x_churn"] = df.overlap_files * e["churn_total"]
    e["overlap_x_files"] = df.overlap_files * e["files_total"]
    e["churn_imbalance"] = (df.churn_p1 - df.churn_p2).abs()
    return e


def best_threshold(y, proba):
    """Threshold in [0.05,0.95] that maximizes F1 on (y, proba)."""
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t


def rf():
    return RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=0, n_jobs=-1)


def hgb():
    # HistGradientBoosting has no class_weight; use sample weights for imbalance at fit time.
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          max_leaf_nodes=31, random_state=0)


def evaluate(X, y, groups, make, balanced_sw=False):
    """Leave-one-repo-out. Returns (F1@0.5, F1@tuned, PR-AUC, tuned-threshold info)."""
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    proba = np.zeros(len(y))
    pred_tuned = np.zeros(len(y), dtype=int)
    for tr, te in gkf.split(X, y, groups):
        m = make()
        sw = None
        if balanced_sw:
            pos = y[tr].sum(); neg = len(tr) - pos
            w = np.where(y[tr] == 1, neg / max(pos, 1), 1.0)
            sw = w
        m.fit(X[tr], y[tr], sample_weight=sw) if sw is not None else m.fit(X[tr], y[tr])
        # tune threshold on TRAIN via inner CV (never sees test fold)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        ip = np.zeros(len(tr))
        for itr, ite in inner.split(X[tr], y[tr]):
            mm = make()
            if balanced_sw:
                p2 = y[tr][itr].sum(); n2 = len(itr) - p2
                w2 = np.where(y[tr][itr] == 1, n2 / max(p2, 1), 1.0)
                mm.fit(X[tr][itr], y[tr][itr], sample_weight=w2)
            else:
                mm.fit(X[tr][itr], y[tr][itr])
            ip[ite] = mm.predict_proba(X[tr][ite])[:, 1]
        thr = best_threshold(y[tr], ip)
        pt = m.predict_proba(X[te])[:, 1]
        proba[te] = pt
        pred_tuned[te] = (pt >= thr).astype(int)
    f1_05 = f1_score(y, (proba >= 0.5).astype(int), zero_division=0)
    f1_tuned = f1_score(y, pred_tuned, zero_division=0)
    return f1_05, f1_tuned, average_precision_score(y, proba), \
        recall_score(y, pred_tuned, zero_division=0), precision_score(y, pred_tuned, zero_division=0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    df = pd.read_csv(path)
    for c in GIT:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = df["repo"].to_numpy()
    eng = add_engineered(df)
    Xg = df[GIT].to_numpy(float)
    Xge = np.hstack([Xg, eng.to_numpy(float)])

    print("=" * 78)
    print(f"TIER-1 IMPROVEMENT EXPERIMENT  |  {len(df)} scenarios, {int(y.sum())} conflicts, "
          f"{len(np.unique(groups))} repos")
    print("Honest leave-one-repository-out. Threshold tuned on TRAIN only.")
    print("=" * 78)
    print(f"{'model':<16}{'features':<12}{'F1@0.5':>8}{'F1@tuned':>10}{'PR-AUC':>9}{'R@tuned':>9}{'P@tuned':>9}")
    combos = [
        ("RandomForest", "GIT", Xg, rf, False),
        ("RandomForest", "GIT+ENG", Xge, rf, False),
        ("HistGradBoost", "GIT", Xg, hgb, True),
        ("HistGradBoost", "GIT+ENG", Xge, hgb, True),
    ]
    best = None
    for name, feats, X, make, sw in combos:
        f05, ft, ap, rec, prec = evaluate(X, y, groups, make, balanced_sw=sw)
        print(f"{name:<16}{feats:<12}{f05:>8.3f}{ft:>10.3f}{ap:>9.3f}{rec:>9.3f}{prec:>9.3f}")
        if best is None or ft > best[0]:
            best = (ft, name, feats)
    print("-" * 78)
    print(f"BEST honest F1: {best[0]:.3f}  ({best[1]} + {best[2]}, tuned threshold)")
    print("Baseline for reference (RandomForest / GIT / 0.5): see F1@0.5 above.")


if __name__ == "__main__":
    main()
