"""
Feature Selection — pick a subset of the original features (they keep their meaning).
Covers all three families the syllabus lists, on the real dataset (all.csv).

  FILTER   : Variance threshold, Correlation pruning, Mutual Information, ANOVA F-test
  WRAPPER  : Recursive Feature Elimination (RFE)
  EMBEDDED : Random Forest importance, Lasso (L1) logistic regression

Outputs a combined ranking (feature_ranking.txt) and a chart (feature_importance.png).
No downloads — numpy/pandas/sklearn/matplotlib only.

Usage:  python feature_selection.py [all.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (mutual_info_classif, f_classif,
                                       VarianceThreshold, RFE)

FEATURES = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
            "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]


def load(path):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    return df[FEATURES].to_numpy(float), y


def ranks_from_scores(scores):
    """Higher score = better -> rank 1..n (1 best)."""
    order = np.argsort(scores)[::-1]
    r = np.empty(len(scores), int)
    for pos, idx in enumerate(order):
        r[idx] = pos + 1
    return r


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    X, y = load(path)
    Xs = StandardScaler().fit_transform(X)
    n = len(FEATURES)
    print("=" * 74)
    print(f"FEATURE SELECTION  |  {X.shape[0]} scenarios, {n} features, {int(y.sum())} conflicts")
    print("=" * 74)

    # ---------- FILTER ----------
    variances = X.var(axis=0)
    near_const = [FEATURES[i] for i in range(n) if variances[i] < 1e-9]
    print("\n[FILTER] Variance threshold")
    print(f"   near-constant features (dropped): {near_const or 'none'}")

    corr = pd.DataFrame(X, columns=FEATURES).corr().abs()
    high = [(FEATURES[i], FEATURES[j], round(corr.iloc[i, j], 2))
            for i in range(n) for j in range(i + 1, n) if corr.iloc[i, j] > 0.85]
    print("\n[FILTER] Correlation pruning (|r| > 0.85)")
    print(f"   highly-correlated pairs: {high or 'none'}")

    mi = mutual_info_classif(X, y, random_state=0)
    fval, _ = f_classif(X, y)
    fval = np.nan_to_num(fval)

    # ---------- WRAPPER ----------
    rfe = RFE(RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                     random_state=0, n_jobs=-1), n_features_to_select=5)
    rfe.fit(X, y)
    rfe_score = (n - rfe.ranking_).astype(float)   # higher = kept earlier

    # ---------- EMBEDDED ----------
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, y)
    rf_imp = rf.feature_importances_
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                               class_weight="balanced", max_iter=2000).fit(Xs, y)
    lasso_imp = np.abs(lasso.coef_[0])

    methods = {"MutualInfo(filter)": mi, "ANOVA-F(filter)": fval,
               "RFE(wrapper)": rfe_score, "RandomForest(embedded)": rf_imp,
               "Lasso-L1(embedded)": lasso_imp}

    # per-method table
    print("\n[SCORES] per method (higher = more important)")
    print(f"{'feature':<16}" + "".join(f"{m.split('(')[0]:>14}" for m in methods))
    for i, f in enumerate(FEATURES):
        print(f"{f:<16}" + "".join(f"{methods[m][i]:>14.3f}" for m in methods))

    # combined ranking = average rank across methods
    rank_mat = np.array([ranks_from_scores(s) for s in methods.values()])
    avg_rank = rank_mat.mean(axis=0)
    order = np.argsort(avg_rank)
    print("\n[COMBINED RANKING]  (lower average rank = more important)")
    lines = ["FEATURE RANKING - Combined (Filter + Wrapper + Embedded)", "=" * 56, ""]
    for pos, i in enumerate(order, 1):
        line = f"{pos:2}. {FEATURES[i]:<16} avg-rank {avg_rank[i]:.1f}"
        print("   " + line); lines.append(line)
    lasso_dropped = [FEATURES[i] for i in range(n) if lasso_imp[i] == 0]
    lines += ["", f"Lasso zeroed out: {lasso_dropped or 'none'}",
              f"Recommended subset (top 6): {[FEATURES[i] for i in order[:6]]}"]
    open("feature_ranking.txt", "w").write("\n".join(lines) + "\n")
    print(f"\nLasso zeroed out: {lasso_dropped or 'none'}")
    print(f"Recommended subset (top 6): {[FEATURES[i] for i in order[:6]]}")

    # chart
    fig, ax = plt.subplots(figsize=(8, 5))
    comb = (rf_imp / rf_imp.max())[order][::-1]
    names = [FEATURES[i] for i in order][::-1]
    ax.barh(range(n), comb, color="#2c78d6")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Random Forest importance (normalised)")
    ax.set_title("Feature Importance / Selection")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=130)
    print("\nSaved: feature_ranking.txt, feature_importance.png")


if __name__ == "__main__":
    main()
