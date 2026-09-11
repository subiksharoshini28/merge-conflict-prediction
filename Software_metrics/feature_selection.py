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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif

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
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Feature Selection - Multiple Methods Comparison",
                 fontsize=16, fontweight="bold", color=C_ACCENT, y=1.02)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

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
        ax.set_title(method_name, fontweight="bold", fontsize=12, pad=10)
        ax.set_xlabel("Normalized Importance", fontsize=9)
        ax.invert_yaxis()

        for bar, val in zip(bars, sorted_importances):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    if len(all_results) < 6:
        axes[1, 2].axis("off")

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

    print("\n  [1/5] Loading dataset...")
    df = load_data(path)
    X = df[FEATURES].to_numpy(float)
    y = df["label"].to_numpy(int)
    print(f"        Loaded: {len(df)} rows, {len(FEATURES)} features")
    print(f"        Conflicts: {int(y.sum())} ({y.mean()*100:.1f}%)")

    print("\n  [2/5] Computing Random Forest importance...")
    rf_imp, rf_idx = rf_importance(X, y)

    print("  [3/5] Computing Gradient Boosting importance...")
    gb_imp, gb_idx = gb_importance(X, y)

    print("  [4/5] Computing Mutual Information...")
    mi_imp, mi_idx = mi_importance(X, y)

    print("  [5/5] Computing Logistic Regression coefficients...")
    lr_imp, lr_idx = lr_importance(X, y)

    print("\n  Running ablation analysis...")
    abl_imp, abl_idx, baseline_f1 = ablation_analysis(X, y)

    all_results = {
        "Random Forest": (rf_imp, rf_idx),
        "Gradient Boosting": (gb_imp, gb_idx),
        "Mutual Information": (mi_imp, mi_idx),
        "Logistic Regression": (lr_imp, lr_idx),
        "Ablation Analysis": (abl_imp, abl_idx),
    }

    print(f"\n{'-' * 80}")
    print("  METHOD COMPARISON")
    print(f"{'-' * 80}")
    print(f"\n  Baseline F1 (all features): {baseline_f1:.3f}")

    final_indices = print_ranking_table(all_results)

    output_chart = "feature_importance.png"
    plot_results(all_results, output_chart)

    output_txt = "feature_ranking.txt"
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
