"""
Feature Extraction - Git History + CSV Statistical Methods

Usage:
    python feature_extraction.py [repo1] [repo2] ...
    python feature_extraction.py --all
    python feature_extraction.py --dim-reduction
    python feature_extraction.py --csv [input.csv]
Output: features_extracted.csv, reports/extracted_features.csv, reports/reduced_features.csv
"""
import os
import sys
import warnings
import subprocess
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

DIVIDER = "=" * 80


def g(repo, *a):
    return subprocess.run(
        ["git", "-C", repo, *a],
        capture_output=True, text=True
    ).stdout.strip()


def extract_branch_features(repo, base, tip):
    commits = int(g(repo, "rev-list", "--count", base + ".." + tip) or 0)
    files = set(g(repo, "diff", "--name-only", base, tip).splitlines())
    authors = len(set(g(repo, "log", "--format=%ae", base + ".." + tip).splitlines()))
    stat = g(repo, "diff", "--shortstat", base, tip)
    ins = dels = 0
    for part in stat.split(", "):
        if "insertion" in part:
            ins = int(part.split()[0])
        elif "deletion" in part:
            dels = int(part.split()[0])
    return commits, files, authors, ins + dels


def extract_merge_features(repo, merge_commit, repo_name):
    parents = g(repo, "rev-list", "--parents", "-n", "1", merge_commit).split()[1:]
    if len(parents) != 2:
        return None
    p1, p2 = parents
    base = g(repo, "merge-base", p1, p2)
    if not base:
        return None

    result = subprocess.run(
        ["git", "-C", repo, "merge-tree", "--write-tree", p1, p2],
        capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        return None
    label = 1 if result.returncode == 1 else 0

    conflict_files = [
        line.split("Merge conflict in ", 1)[1]
        for line in result.stdout.splitlines()
        if "Merge conflict in " in line
    ]
    code_conflict = 1 if any(f.strip().endswith(".java") for f in conflict_files) else 0

    c1, f1, a1, ch1 = extract_branch_features(repo, base, p1)
    c2, f2, a2, ch2 = extract_branch_features(repo, base, p2)

    if c1 == 0 or c2 == 0:
        return None

    overlap = f1 & f2
    union = f1 | f2
    overlap_ratio = round(len(overlap) / len(union), 4) if union else 0

    return {
        "repo": repo_name,
        "merge": merge_commit,
        "commits_p1": c1,
        "commits_p2": c2,
        "files_p1": len(f1),
        "files_p2": len(f2),
        "overlap_files": len(overlap),
        "overlap_ratio": overlap_ratio,
        "authors_p1": a1,
        "authors_p2": a2,
        "churn_p1": ch1,
        "churn_p2": ch2,
        "conflict_files": ";".join(conflict_files),
        "code_conflict": code_conflict,
        "label": label
    }


def dim_reduction_visualization(csv_path):
    df = pd.read_csv(csv_path)
    features = ["commits_p1", "commits_p2", "files_p1", "files_p2",
                "overlap_files", "overlap_ratio", "authors_p1", "authors_p2",
                "churn_p1", "churn_p2"]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    X = df[features].values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    methods = {
        "PCA": PCA(n_components=2, random_state=0),
        "LDA": LDA(n_components=1),
        "t-SNE": TSNE(n_components=2, random_state=0, perplexity=min(30, len(X)-1)),
        "Kernel-PCA (RBF)": KernelPCA(n_components=2, kernel="rbf", random_state=0),
        "ICA": FastICA(n_components=2, random_state=0, max_iter=500),
    }

    fig, axes = plt.subplots(1, 5, figsize=(28, 5))
    fig.suptitle("Dimensionality Reduction - 2D Visualization (for completeness)",
                 fontsize=14, fontweight="bold", y=1.02)

    colors = ["#3498db", "#e74c3c"]
    labels_map = {0: "Clean", 1: "Conflict"}

    for idx, (name, model) in enumerate(methods.items()):
        ax = axes[idx]
        try:
            X_2d = model.fit_transform(X_scaled, y) if name == "LDA" else model.fit_transform(X_scaled)
            if X_2d.shape[1] == 1:
                X_2d = np.hstack([X_2d, np.zeros_like(X_2d)])
        except Exception:
            ax.set_title(f"{name}\n(failed)", fontsize=11)
            ax.axis("off")
            continue

        for label in [0, 1]:
            mask = y == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=colors[label], label=labels_map[label],
                       alpha=0.6, edgecolors="white", s=50, linewidths=0.5)
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out = "dim_reduction_visualization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Chart saved: {out}")
    plt.close()

    print("\n  NOTE: Dimensionality reduction reduces F1 (0.599 -> 0.554 with PCA).")
    print("        Used only for 2D visualization. Model retains original interpretable features.")


# ============================================================================
# CSV STATISTICAL FEATURE EXTRACTION
# ============================================================================

def csv_statistical_extraction(df, numeric_cols):
    print("\n  [1/7] Statistical Feature Extraction...")
    result = df.copy()
    count = 0
    for col in numeric_cols:
        result[f"{col}_mean"] = df[col].mean()
        result[f"{col}_median"] = df[col].median()
        result[f"{col}_std"] = df[col].std()
        result[f"{col}_skew"] = df[col].skew()
        result[f"{col}_kurtosis"] = df[col].kurtosis()
        result[f"{col}_min"] = df[col].min()
        result[f"{col}_max"] = df[col].max()
        result[f"{col}_range"] = df[col].max() - df[col].min()
        result[f"{col}_q25"] = df[col].quantile(0.25)
        result[f"{col}_q75"] = df[col].quantile(0.75)
        result[f"{col}_iqr"] = df[col].quantile(0.75) - df[col].quantile(0.25)
        count += 11
    print(f"        Added {count} statistical features")
    return result


def csv_correlation_extraction(df, numeric_cols):
    print("\n  [2/7] Correlation-based Feature Extraction (PCA, LDA, ICA)...")
    result = df.copy()
    X = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(2, X.shape[1]), random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    for i in range(X_pca.shape[1]):
        result[f"PCA_{i+1}"] = X_pca[:, i]
    print(f"        PCA: variance = {pca.explained_variance_ratio_}")

    if "label" in df.columns:
        y = df["label"].values
        n_classes = len(np.unique(y))
        if n_classes > 1:
            lda = LDA(n_components=min(n_classes - 1, X.shape[1]))
            X_lda = lda.fit_transform(X_scaled, y)
            for i in range(X_lda.shape[1]):
                result[f"LDA_{i+1}"] = X_lda[:, i]
            print(f"        LDA: {X_lda.shape[1]} component(s)")

    ica = FastICA(n_components=min(2, X.shape[1]), random_state=0, max_iter=500)
    X_ica = ica.fit_transform(X_scaled)
    for i in range(X_ica.shape[1]):
        result[f"ICA_{i+1}"] = X_ica[:, i]
    print(f"        ICA: {X_ica.shape[1]} component(s)")

    return result


def csv_interaction_extraction(df, numeric_cols):
    print("\n  [3/7] Interaction Feature Extraction (ratio, difference, polynomial)...")
    result = df.copy()
    count = 0
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            result[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-10)
            result[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
            result[f"{col1}_mul_{col2}"] = df[col1] * df[col2]
            count += 3
    print(f"        Added {count} interaction features")
    return result


def csv_categorical_extraction(df):
    print("\n  [4/7] Categorical Feature Extraction (one-hot, label, frequency, target)...")
    result = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    count = 0
    for col in cat_cols:
        if col in ["conflict_files"]:
            continue
        dummies = pd.get_dummies(df[col], prefix=f"{col}_onehot")
        result = pd.concat([result, dummies], axis=1)
        count += len(dummies.columns)

        le = LabelEncoder()
        result[f"{col}_label"] = le.fit_transform(df[col].astype(str))
        count += 1

        freq = df[col].value_counts(normalize=True)
        result[f"{col}_freq"] = df[col].map(freq)
        count += 1

        if "label" in df.columns:
            target_mean = df.groupby(col)["label"].mean()
            result[f"{col}_target"] = df[col].map(target_mean)
            count += 1
    print(f"        Added {count} categorical features")
    return result


def csv_binning_extraction(df, numeric_cols):
    print("\n  [5/7] Binning/Discretization (uniform, quantile, k-means)...")
    result = df.copy()
    count = 0
    for col in numeric_cols[:5]:
        try:
            kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
            result[f"{col}_bin_uniform"] = kbins.fit_transform(df[[col]].fillna(0))
            count += 1
        except Exception:
            pass
        try:
            kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            result[f"{col}_bin_quantile"] = kbins.fit_transform(df[[col]].fillna(0))
            count += 1
        except Exception:
            pass
        try:
            km = KMeans(n_clusters=5, random_state=0, n_init=10)
            result[f"{col}_bin_kmeans"] = km.fit_predict(df[[col]].fillna(0))
            count += 1
        except Exception:
            pass
    print(f"        Added {count} binned features")
    return result


def csv_log_transform_extraction(df, numeric_cols):
    print("\n  [6/7] Log Transform Extraction (log1p)...")
    result = df.copy()
    count = 0
    for col in numeric_cols:
        if df[col].min() >= 0:
            result[f"{col}_log1p"] = np.log1p(df[col])
            count += 1
    print(f"        Added {count} log-transformed features")
    return result


def csv_domain_extraction(df):
    print("\n  [7/7] Domain-Specific Feature Extraction...")
    result = df.copy()
    count = 0
    if "overlap_files" in df.columns and "files_p1" in df.columns:
        result["domain_overlap_density"] = df["overlap_files"] / (df["files_p1"] + 1)
        count += 1
    if "churn_p1" in df.columns and "churn_p2" in df.columns:
        result["domain_churn_ratio"] = df["churn_p1"] / (df["churn_p2"] + 1)
        count += 1
    if "commits_p1" in df.columns and "commits_p2" in df.columns:
        result["domain_commit_balance"] = abs(df["commits_p1"] - df["commits_p2"]) / (df["commits_p1"] + df["commits_p2"] + 1)
        count += 1
    print(f"        Added {count} domain-specific features")
    return result


# ============================================================================
# CSV FEATURE REDUCTION
# ============================================================================

def csv_variance_reduction(df, threshold=0.01):
    print("\n  [1/7] Variance Threshold (Filter)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0).values
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    kept_mask = selector.get_support()
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed = len(numeric_cols) - len(kept_cols)
    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Removed {removed} low-variance features")
    return result


def csv_correlation_reduction(df, threshold=0.95):
    print("\n  [2/7] Correlation Removal (Filter)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    removed = [col for col in upper.columns if any(upper[col] > threshold)]
    result = df.drop(columns=removed, errors="ignore")
    print(f"        Removed {len(removed)} highly correlated features (>{threshold})")
    return result


def csv_mi_reduction(df, label_col="label", k=10):
    print("\n  [3/7] Mutual Information (Filter)...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df
    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    k = min(k, len(numeric_cols))
    mi = mutual_info_classif(X, y, random_state=0)
    top_indices = np.argsort(mi)[::-1][:k]
    kept_cols = [numeric_cols[i] for i in top_indices]
    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept top {k} features by Mutual Information")
    return result


def csv_anova_reduction(df, label_col="label", k=10):
    print("\n  [4/7] ANOVA F-test (Filter)...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df
    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    k = min(k, len(numeric_cols))
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    kept_mask = selector.get_support()
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept top {k} features by ANOVA F-test")
    return result


def csv_rfe_reduction(df, label_col="label", n_features=10):
    print("\n  [5/7] Recursive Feature Elimination (Wrapper)...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df
    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    n_features = min(n_features, len(numeric_cols))
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
    rfe = RFE(rf, n_features_to_select=n_features)
    rfe.fit(X, y)
    kept_mask = rfe.support_
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept top {n_features} features by RFE")
    return result


def csv_rf_reduction(df, label_col="label", threshold=0.01):
    print("\n  [6/7] Random Forest Importance (Embedded)...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df
    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=2, random_state=0, n_jobs=1)
    rf.fit(X, y)
    kept_mask = rf.feature_importances_ > threshold
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed = len(numeric_cols) - len(kept_cols)
    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Removed {removed} features (importance < {threshold})")
    return result


def csv_lasso_reduction(df, label_col="label", threshold=0.001):
    print("\n  [7/7] Lasso L1 (Embedded)...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df
    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=0, max_iter=5000)
    lasso.fit(X_scaled, y)
    kept_mask = np.abs(lasso.coef_) > threshold
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed = len(numeric_cols) - len(kept_cols)
    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept {len(kept_cols)} features, removed {removed}")
    return result


# ============================================================================
# CSV VISUALIZATION
# ============================================================================

def csv_visualize(original_df, extracted_df, reduced_df, output_dir):
    print("\n  Generating visualization charts...")
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle("Feature Extraction & Reduction - Complete Pipeline",
                 fontsize=16, fontweight="bold", y=1.02)

    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    axes[0, 0].bar(range(len(numeric_cols)),
                   [original_df[c].mean() for c in numeric_cols],
                   color="#3498db", alpha=0.8)
    axes[0, 0].set_xticks(range(len(numeric_cols)))
    axes[0, 0].set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)
    axes[0, 0].set_title("Original Features (Mean)", fontweight="bold")

    pca_cols = [c for c in extracted_df.columns if c.startswith("PCA_")]
    if pca_cols and len(pca_cols) >= 2:
        label_col = "label" if "label" in extracted_df.columns else None
        colors = extracted_df[label_col].values if label_col else "#3498db"
        axes[0, 1].scatter(extracted_df[pca_cols[0]], extracted_df[pca_cols[1]],
                          c=colors, alpha=0.6, s=30)
        axes[0, 1].set_title("PCA Projection (Extracted)", fontweight="bold")

    int_cols = [c for c in extracted_df.columns if "_mul_" in c][:10]
    if int_cols:
        axes[0, 2].bar(range(len(int_cols)),
                       [extracted_df[c].std() for c in int_cols],
                       color="#2ecc71", alpha=0.8)
        axes[0, 2].set_xticks(range(len(int_cols)))
        axes[0, 2].set_xticklabels(int_cols, rotation=45, ha="right", fontsize=7)
        axes[0, 2].set_title("Polynomial Features (Std)", fontweight="bold")

    bin_cols = [c for c in extracted_df.columns if "_bin_" in c][:10]
    if bin_cols:
        axes[0, 3].bar(range(len(bin_cols)),
                       [extracted_df[c].nunique() for c in bin_cols],
                       color="#e74c3c", alpha=0.8)
        axes[0, 3].set_xticks(range(len(bin_cols)))
        axes[0, 3].set_xticklabels(bin_cols, rotation=45, ha="right", fontsize=7)
        axes[0, 3].set_title("Binned Features (Unique Values)", fontweight="bold")

    feature_counts = [len(original_df.columns), len(extracted_df.columns), len(reduced_df.columns)]
    axes[1, 0].bar(["Original", "Extracted", "Reduced"], feature_counts,
                   color=["#3498db", "#2ecc71", "#e74c3c"], alpha=0.8)
    axes[1, 0].set_title("Feature Count Progression", fontweight="bold")
    for i, v in enumerate(feature_counts):
        axes[1, 0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    if "label" in reduced_df.columns:
        reduced_numeric = reduced_df.select_dtypes(include=[np.number]).columns.tolist()
        if "label" in reduced_numeric:
            reduced_numeric.remove("label")
        if reduced_numeric:
            top_features = reduced_numeric[:10]
            corrs = [reduced_df[f].corr(reduced_df["label"]) for f in top_features]
            axes[1, 1].barh(range(len(top_features)), corrs, color="#9b59b6", alpha=0.8)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=8)
            axes[1, 1].set_title("Top Reduced Features (Correlation)", fontweight="bold")

    axes[1, 2].axis("off")
    summary = (
        "EXTRACTION METHODS:\n"
        "  - Statistical (mean, median, std, skew, kurtosis)\n"
        "  - Correlation-based (PCA, LDA, ICA)\n"
        "  - Interaction (ratio, difference, polynomial)\n"
        "  - Categorical (one-hot, label, frequency, target)\n"
        "  - Binning (uniform, quantile, k-means)\n"
        "  - Log Transform (log1p)\n"
        "  - Domain-Specific\n\n"
        "REDUCTION METHODS:\n"
        "  - Filter: Variance, Correlation, MI, ANOVA\n"
        "  - Wrapper: RFE\n"
        "  - Embedded: RF Importance, Lasso L1"
    )
    axes[1, 2].text(0.1, 0.5, summary, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment="center", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[1, 3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_extraction_full.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Chart saved: {out_path}")
    plt.close()


def csv_pipeline(input_csv):
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}, Numeric: {len(numeric_cols)}")

    print(f"\n{'-' * 80}")
    print("  PART 1: FEATURE EXTRACTION")
    print(f"{'-' * 80}")

    extracted_df = df.copy()
    extracted_df = csv_statistical_extraction(extracted_df, numeric_cols)
    extracted_df = csv_correlation_extraction(extracted_df, numeric_cols)
    extracted_df = csv_interaction_extraction(extracted_df, numeric_cols)
    extracted_df = csv_categorical_extraction(extracted_df)
    extracted_df = csv_binning_extraction(extracted_df, numeric_cols)
    extracted_df = csv_log_transform_extraction(extracted_df, numeric_cols)
    extracted_df = csv_domain_extraction(extracted_df)

    print(f"\n  Features after extraction: {len(extracted_df.columns)}")
    out_extracted = os.path.join(output_dir, "extracted_features.csv")
    extracted_df.to_csv(out_extracted, index=False)
    print(f"  Saved: {out_extracted}")

    print(f"\n{'-' * 80}")
    print("  PART 2: FEATURE REDUCTION")
    print(f"{'-' * 80}")

    reduced_df = extracted_df.copy()
    reduced_df = csv_variance_reduction(reduced_df)
    reduced_df = csv_correlation_reduction(reduced_df)
    reduced_df = csv_mi_reduction(reduced_df)
    reduced_df = csv_anova_reduction(reduced_df)
    reduced_df = csv_rfe_reduction(reduced_df)
    reduced_df = csv_rf_reduction(reduced_df)
    reduced_df = csv_lasso_reduction(reduced_df)

    print(f"\n  Features after reduction: {len(reduced_df.columns)}")
    out_reduced = os.path.join(output_dir, "reduced_features.csv")
    reduced_df.to_csv(out_reduced, index=False)
    print(f"  Saved: {out_reduced}")

    csv_visualize(df, extracted_df, reduced_df, output_dir)

    print(f"\n{'-' * 80}")
    print("  PIPELINE COMPLETE")
    print(f"{'-' * 80}")
    print(f"  Original features : {len(df.columns)}")
    print(f"  Extracted features: {len(extracted_df.columns)}")
    print(f"  Reduced features : {len(reduced_df.columns)}")
    print(DIVIDER)


def main():
    print(DIVIDER)
    print("  FEATURE EXTRACTION - Git History + CSV Statistical Methods")
    print(DIVIDER)

    args = sys.argv[1:]

    if "--csv" in args:
        idx = args.index("--csv")
        csv_path = args[idx + 1] if idx + 1 < len(args) else "all.csv"
        csv_pipeline(csv_path)
        return

    if "--dim-reduction" in args:
        csv_path = "features_extracted.csv"
        if not os.path.exists(csv_path):
            print(f"  Error: {csv_path} not found. Run extraction first.")
            return
        dim_reduction_visualization(csv_path)
        return

    if "--all" in args:
        repos_dir = "repos"
        if os.path.exists(repos_dir):
            repos = [
                os.path.join(repos_dir, d)
                for d in os.listdir(repos_dir)
                if os.path.isdir(os.path.join(repos_dir, d))
            ]
        else:
            print("  Error: repos/ directory not found")
            return
    else:
        repos = [r for r in args if r.startswith("repos/")]

    if not repos:
        print("\n  Usage:")
        print("    python feature_extraction.py repos/commons-lang repos/commons-io")
        print("    python feature_extraction.py --all")
        return

    print(f"\n  Repos to mine: {len(repos)}")
    for r in repos:
        print(f"    - {r}")

    output_file = "features_extracted.csv"
    fieldnames = [
        "repo", "merge", "commits_p1", "commits_p2", "files_p1", "files_p2",
        "overlap_files", "overlap_ratio", "authors_p1", "authors_p2",
        "churn_p1", "churn_p2", "conflict_files", "code_conflict", "label"
    ]

    total_scenarios = 0
    total_conflicts = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo_path in repos:
            repo_name = os.path.basename(repo_path.rstrip("/\\"))
            print(f"\n  [{repo_name}] Mining merges...")

            merges = g(repo_path, "log", "--merges", "--format=%H").splitlines()
            print(f"    Found {len(merges)} merge commits")

            repo_scenarios = 0
            repo_conflicts = 0

            for merge_commit in merges:
                features = extract_merge_features(repo_path, merge_commit, repo_name)
                if features:
                    writer.writerow(features)
                    repo_scenarios += 1
                    if features["label"] == 1:
                        repo_conflicts += 1

            total_scenarios += repo_scenarios
            total_conflicts += repo_conflicts
            print(f"    Extracted: {repo_scenarios} scenarios, {repo_conflicts} conflicts")

    print(f"\n{'-' * 80}")
    print("  EXTRACTION COMPLETE")
    print(f"{'-' * 80}")
    print(f"  Output file      : {output_file}")
    print(f"  Total scenarios  : {total_scenarios}")
    print(f"  Total conflicts  : {total_conflicts}")
    if total_scenarios > 0:
        print(f"  Conflict rate    : {total_conflicts/total_scenarios*100:.1f}%")
    print(f"  Features per row : 10")
    print(DIVIDER)


if __name__ == "__main__":
    main()
