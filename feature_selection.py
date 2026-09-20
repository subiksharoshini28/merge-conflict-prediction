"""
Feature Selection — Identify and rank the most important features.

Usage:  python feature_selection.py [features_extracted.csv]
Output: feature_importance.png
        feature_ranking.txt
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.feature_selection import (
    mutual_info_classif,
    VarianceThreshold,
    SelectKBest,
    f_classif,
    RFE,
    mutual_info_classif,
)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

DIVIDER = "=" * 80
C_ACCENT = "#2c3e50"

FEATURES = [
    "commits_p1", "commits_p2", "files_p1", "files_p2",
    "overlap_files", "overlap_ratio", "authors_p1", "authors_p2",
    "churn_p1", "churn_p2"
]

FEATURE_DESCRIPTIONS = {
    "commits_p1": "Number of commits branch 1 diverged by",
    "commits_p2": "Number of commits branch 2 diverged by",
    "files_p1": "Number of files changed on branch 1",
    "files_p2": "Number of files changed on branch 2",
    "overlap_files": "Files edited by BOTH branches",
    "overlap_ratio": "Overlap as fraction of all touched files",
    "authors_p1": "Distinct authors on branch 1",
    "authors_p2": "Distinct authors on branch 2",
    "churn_p1": "Lines added + deleted by branch 1",
    "churn_p2": "Lines added + deleted by branch 2"
}


def load_data(path):
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


def rf_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        min_samples_leaf=2, random_state=0, n_jobs=1
    )
    rf.fit(X, y)
    return rf.feature_importances_, np.argsort(rf.feature_importances_)[::-1]


def gb_importance(X, y):
    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X, y)
    return gb.feature_importances_, np.argsort(gb.feature_importances_)[::-1]


def mi_importance(X, y):
    mi = mutual_info_classif(X, y, random_state=0)
    return mi, np.argsort(mi)[::-1]


def lr_importance(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    lr.fit(X_scaled, y)
    imp = np.abs(lr.coef_[0])
    return imp, np.argsort(imp)[::-1]


# --- Filter Methods ---

def variance_importance(X, y):
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X)
    imp = selector.variances_
    return imp, np.argsort(imp)[::-1]


def correlation_importance(X, y):
    corr = np.array([np.abs(spearmanr(X[:, i], y)[0]) for i in range(X.shape[1])])
    return corr, np.argsort(corr)[::-1]


def anova_importance(X, y):
    selector = SelectKBest(f_classif, k="all")
    selector.fit(X, y)
    imp = selector.scores_
    return imp, np.argsort(imp)[::-1]


# --- Wrapper Method ---

def rfe_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample",
        min_samples_leaf=2, random_state=0, n_jobs=1
    )
    rfe = RFE(rf, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    imp = np.array([len(FEATURES) - r + 1 for r in rfe.ranking_], dtype=float)
    return imp, np.argsort(imp)[::-1]


# --- Embedded Method ---

def lasso_importance(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=0, max_iter=5000)
    lasso.fit(X_scaled, y)
    imp = np.abs(lasso.coef_)
    return imp, np.argsort(imp)[::-1]


def ablation_analysis(X, y):
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample",
        min_samples_leaf=2, random_state=0, n_jobs=1
    )
    y_pred = cross_val_predict(rf, X, y, cv=5)
    baseline_f1 = f1_score(y, y_pred)

    drops = []
    for i in range(X.shape[1]):
        X_reduced = np.delete(X, i, axis=1)
        y_pred_r = cross_val_predict(rf, X_reduced, y, cv=5)
        drops.append(baseline_f1 - f1_score(y, y_pred_r))

    drops = np.array(drops)
    return drops, np.argsort(drops)[::-1], baseline_f1


def plot_results(all_results, output_path):
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    fig.suptitle("Feature Selection - Filter / Wrapper / Embedded Methods",
                 fontsize=16, fontweight="bold", color=C_ACCENT, y=1.01)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4"]

    for idx, (method_name, (importances, indices)) in enumerate(all_results.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        sorted_features = [FEATURES[i] for i in indices]
        sorted_importances = importances[indices]
        if sorted_importances.max() > 0:
            sorted_importances = sorted_importances / sorted_importances.max()

        bars = ax.barh(range(len(sorted_features)), sorted_importances,
                       color=colors[idx % len(colors)], edgecolor="white", alpha=0.8)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, fontsize=9)
        ax.set_title(method_name, fontweight="bold", fontsize=11, pad=10)
        ax.set_xlabel("Normalized Importance", fontsize=9)
        ax.invert_yaxis()

        for bar, val in zip(bars, sorted_importances):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    for idx in range(len(all_results), 12):
        axes[idx // 3, idx % 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Chart saved: {output_path}")
    plt.close()


def print_ranking_table(all_results):
    print(f"\n{'-' * 80}")
    print("  COMBINED FEATURE RANKING (Average across all methods)")
    print(f"{'-' * 80}")

    ranks = np.zeros(len(FEATURES))
    for method_name, (importances, indices) in all_results.items():
        for rank, idx in enumerate(indices):
            ranks[idx] += rank + 1
    ranks /= len(all_results)
    final_indices = np.argsort(ranks)

    print(f"\n  {'Rank':<6} {'Feature':<18} {'Avg Rank':<10} {'Description'}")
    print(f"  {'-'*6} {'-'*18} {'-'*10} {'-'*40}")

    for rank, idx in enumerate(final_indices, 1):
        feature = FEATURES[idx]
        avg_rank = ranks[idx]
        desc = FEATURE_DESCRIPTIONS[feature]
        print(f"  {rank:<6} {feature:<18} {avg_rank:<10.1f} {desc}")

    return final_indices


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "features_extracted.csv"

    print(DIVIDER)
    print("  FEATURE SELECTION - Identify Most Important Features")
    print(DIVIDER)

    print("\n  [1/3] Loading dataset...")
    df = load_data(path)
    X = df[FEATURES].to_numpy(float)
    y = df["label"].to_numpy(int)
    print(f"        Loaded: {len(df)} rows, {len(FEATURES)} features")
    print(f"        Conflicts: {int(y.sum())} ({y.mean()*100:.1f}%)")

    print("\n  [1/10] Computing Variance (filter)...")
    var_imp, var_idx = variance_importance(X, y)

    print("  [2/10] Computing Correlation (filter)...")
    corr_imp, corr_idx = correlation_importance(X, y)

    print("  [3/10] Computing Mutual Information (filter)...")
    mi_imp, mi_idx = mi_importance(X, y)

    print("  [4/10] Computing ANOVA F-test (filter)...")
    anova_imp, anova_idx = anova_importance(X, y)

    print("  [5/10] Computing Recursive Feature Elimination (wrapper)...")
    rfe_imp, rfe_idx = rfe_importance(X, y)

    print("  [6/10] Computing Random Forest importance (embedded)...")
    rf_imp, rf_idx = rf_importance(X, y)

    print("  [7/10] Computing Gradient Boosting importance (embedded)...")
    gb_imp, gb_idx = gb_importance(X, y)

    print("  [8/10] Computing Logistic Regression coefficients (embedded)...")
    lr_imp, lr_idx = lr_importance(X, y)

    print("  [9/10] Computing Lasso L1 (embedded)...")
    lasso_imp, lasso_idx = lasso_importance(X, y)

    print("  [10/10] Running ablation analysis...")
    abl_imp, abl_idx, baseline_f1 = ablation_analysis(X, y)

    all_results = {
        "Variance (filter)": (var_imp, var_idx),
        "Correlation (filter)": (corr_imp, corr_idx),
        "Mutual Information (filter)": (mi_imp, mi_idx),
        "ANOVA F-test (filter)": (anova_imp, anova_idx),
        "RFE (wrapper)": (rfe_imp, rfe_idx),
        "Random Forest (embedded)": (rf_imp, rf_idx),
        "Gradient Boosting (embedded)": (gb_imp, gb_idx),
        "Logistic Regression (embedded)": (lr_imp, lr_idx),
        "Lasso L1 (embedded)": (lasso_imp, lasso_idx),
        "Ablation Analysis": (abl_imp, abl_idx),
    }

    print(f"\n{'-' * 80}")
    print("  METHOD COMPARISON")
    print(f"{'-' * 80}")
    print(f"\n  Baseline F1 (all features): {baseline_f1:.3f}")

    final_indices = print_ranking_table(all_results)

    output_chart = "reports/feature_importance_10methods.png"
    plot_results(all_results, output_chart)

    output_txt = "reports/feature_ranking.txt"
    with open(output_txt, "w") as f:
        f.write("FEATURE RANKING - Combined Methods\n")
        f.write("=" * 60 + "\n\n")
        for rank, idx in enumerate(final_indices, 1):
            feature = FEATURES[idx]
            desc = FEATURE_DESCRIPTIONS[feature]
            f.write(f"{rank}. {feature:<18} - {desc}\n")

    print(f"  Ranking saved: {output_txt}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
