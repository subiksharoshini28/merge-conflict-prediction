"""
Statistical Feature Extraction - Apply statistical methods to CSV files.

Usage:  python statistical_feature_extraction.py [input.csv]
Output: statistical_features.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

DIVIDER = "=" * 80


def load_data(path):
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df, numeric_cols


def statistical_extraction(df, numeric_cols):
    print("\n  [1/8] Statistical Feature Extraction...")
    result = df.copy()

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

    print(f"        Added {len([c for c in result.columns if c not in df.columns])} statistical features")
    return result


def correlation_extraction(df, numeric_cols):
    print("\n  [2/8] Correlation-based Feature Extraction...")
    result = df.copy()

    corr_matrix = df[numeric_cols].corr()
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))

    print(f"        High correlation pairs (>0.7): {len(high_corr_pairs)}")
    return result, high_corr_pairs


def pca_extraction(df, numeric_cols, n_components=2):
    print("\n  [3/8] PCA Feature Extraction...")
    X = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    result = df.copy()
    for i in range(X_pca.shape[1]):
        result[f"PCA_{i+1}"] = X_pca[:, i]

    print(f"        Explained variance: {pca.explained_variance_ratio_}")
    return result, pca


def lda_extraction(df, numeric_cols, label_col="label"):
    print("\n  [4/8] LDA Feature Extraction...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df, None

    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_classes = len(np.unique(y))
    n_components = min(n_classes - 1, X.shape[1])
    if n_components < 1:
        print("        Skipped: insufficient classes")
        return df, None

    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)

    result = df.copy()
    for i in range(X_lda.shape[1]):
        result[f"LDA_{i+1}"] = X_lda[:, i]

    print(f"        Components: {X_lda.shape[1]}")
    return result, lda


def ica_extraction(df, numeric_cols, n_components=2):
    print("\n  [5/8] ICA Feature Extraction...")
    X = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ica = FastICA(n_components=min(n_components, X.shape[1]), random_state=0, max_iter=500)
    X_ica = ica.fit_transform(X_scaled)

    result = df.copy()
    for i in range(X_ica.shape[1]):
        result[f"ICA_{i+1}"] = X_ica[:, i]

    print(f"        Components: {X_ica.shape[1]}")
    return result, ica


def interaction_extraction(df, numeric_cols):
    print("\n  [6/8] Interaction Feature Extraction...")
    result = df.copy()
    new_features = []

    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]

            result[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-10)
            new_features.append(f"{col1}_div_{col2}")

            result[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
            new_features.append(f"{col1}_minus_{col2}")

    print(f"        Added {len(new_features)} interaction features")
    return result


def categorical_extraction(df):
    print("\n  [7/8] Categorical Feature Extraction...")
    result = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    new_features = []

    for col in cat_cols:
        if col in ["conflict_files"]:
            continue

        dummies = pd.get_dummies(df[col], prefix=col)
        result = pd.concat([result, dummies], axis=1)
        new_features.extend(dummies.columns.tolist())

    print(f"        Categorical columns: {len(cat_cols)}")
    print(f"        Added {len(new_features)} one-hot features")
    return result


def log_transform_extraction(df, numeric_cols):
    print("\n  [8/8] Log Transform Extraction...")
    result = df.copy()
    new_features = []

    for col in numeric_cols:
        if df[col].min() >= 0:
            result[f"{col}_log1p"] = np.log1p(df[col])
            new_features.append(f"{col}_log1p")

    print(f"        Added {len(new_features)} log-transformed features")
    return result


def variance_reduction(df, threshold=0.01):
    print("\n  [1/5] Variance Threshold Reduction...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0).values

    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    kept_mask = selector.get_support()
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Removed {len(removed_cols)} low-variance features")
    print(f"        Remaining: {len(kept_cols)} numeric features")
    return result, removed_cols


def correlation_reduction(df, threshold=0.95):
    print("\n  [2/5] Correlation-based Reduction...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    removed = [col for col in upper.columns if any(upper[col] > threshold)]

    result = df.drop(columns=removed, errors="ignore")
    print(f"        Removed {len(removed)} highly correlated features (>{threshold})")
    print(f"        Remaining: {len(result.columns)} features")
    return result, removed


def selectkbest_reduction(df, label_col="label", k=10):
    print("\n  [3/5] SelectKBest Reduction...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df, []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df, []

    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    k = min(k, len(numeric_cols))

    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    kept_mask = selector.get_support()
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept top {k} features by ANOVA F-test")
    print(f"        Removed {len(removed_cols)} features")
    return result, removed_cols


def rfe_reduction(df, label_col="label", n_features=10):
    print("\n  [4/5] RFE Reduction...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df, []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df, []

    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values
    n_features = min(n_features, len(numeric_cols))

    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
    rfe = RFE(rf, n_features_to_select=n_features)
    rfe.fit(X, y)
    kept_mask = rfe.support_
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept top {n_features} features by RFE")
    print(f"        Removed {len(removed_cols)} features")
    return result, removed_cols


def lasso_reduction(df, label_col="label", threshold=0.001):
    print("\n  [5/5] Lasso L1 Reduction...")
    if label_col not in df.columns:
        print(f"        Skipped: {label_col} not found")
        return df, []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        return df, []

    X = df[numeric_cols].fillna(0).values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=0, max_iter=5000)
    lasso.fit(X_scaled, y)

    kept_mask = np.abs(lasso.coef_) > threshold
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept {len(kept_cols)} features (coef > {threshold})")
    print(f"        Removed {len(removed_cols)} zero-coefficient features")
    return result, removed_cols


def visualize_extraction(original_df, enhanced_df, numeric_cols, output_dir):
    print("\n  Generating visualization charts...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Feature Extraction - Statistical Methods Applied to CSV",
                 fontsize=16, fontweight="bold", y=1.02)

    axes[0, 0].bar(range(len(numeric_cols)),
                   [original_df[c].mean() for c in numeric_cols],
                   color="#3498db", alpha=0.8)
    axes[0, 0].set_xticks(range(len(numeric_cols)))
    axes[0, 0].set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_title("Original Features (Mean)", fontweight="bold")

    axes[0, 1].bar(range(len(numeric_cols)),
                   [original_df[c].std() for c in numeric_cols],
                   color="#e74c3c", alpha=0.8)
    axes[0, 1].set_xticks(range(len(numeric_cols)))
    axes[0, 1].set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_title("Original Features (Std Dev)", fontweight="bold")

    axes[0, 2].bar(range(len(numeric_cols)),
                   [original_df[c].skew() for c in numeric_cols],
                   color="#2ecc71", alpha=0.8)
    axes[0, 2].set_xticks(range(len(numeric_cols)))
    axes[0, 2].set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    axes[0, 2].set_title("Original Features (Skewness)", fontweight="bold")

    pca_cols = [c for c in enhanced_df.columns if c.startswith("PCA_")]
    if pca_cols:
        for i, col in enumerate(pca_cols[:2]):
            axes[1, 0].scatter(enhanced_df[col], enhanced_df[pca_cols[1] if len(pca_cols) > 1 else 0],
                             c=enhanced_df.get("label", "#3498db"), alpha=0.6, s=30)
        axes[1, 0].set_title("PCA Projection", fontweight="bold")
        axes[1, 0].set_xlabel(pca_cols[0])
        if len(pca_cols) > 1:
            axes[1, 0].set_ylabel(pca_cols[1])

    log_cols = [c for c in enhanced_df.columns if c.endswith("_log1p")]
    if log_cols:
        axes[1, 1].bar(range(min(10, len(log_cols))),
                       [enhanced_df[c].mean() for c in log_cols[:10]],
                       color="#f39c12", alpha=0.8)
        axes[1, 1].set_xticks(range(min(10, len(log_cols))))
        axes[1, 1].set_xticklabels([c.replace("_log1p", "") for c in log_cols[:10]],
                                   rotation=45, ha="right", fontsize=8)
        axes[1, 1].set_title("Log-Transformed Features (Mean)", fontweight="bold")

    int_cols = [c for c in enhanced_df.columns if "_div_" in c or "_minus_" in c]
    if int_cols:
        axes[1, 2].bar(range(min(10, len(int_cols))),
                       [enhanced_df[c].std() for c in int_cols[:10]],
                       color="#9b59b6", alpha=0.8)
        axes[1, 2].set_xticks(range(min(10, len(int_cols))))
        axes[1, 2].set_xticklabels(int_cols[:10], rotation=45, ha="right", fontsize=8)
        axes[1, 2].set_title("Interaction Features (Std Dev)", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_extraction_statistical.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Chart saved: {out_path}")
    plt.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    print(DIVIDER)
    print("  STATISTICAL FEATURE EXTRACTION - CSV Files")
    print(DIVIDER)

    print(f"\n  Loading: {path}")
    df, numeric_cols = load_data(path)
    print(f"  Rows: {len(df)}, Numeric columns: {len(numeric_cols)}")

    enhanced_df = df.copy()

    enhanced_df, high_corr = correlation_extraction(enhanced_df, numeric_cols)
    enhanced_df, pca_model = pca_extraction(enhanced_df, numeric_cols)
    enhanced_df, lda_model = lda_extraction(enhanced_df, numeric_cols)
    enhanced_df, ica_model = ica_extraction(enhanced_df, numeric_cols)
    enhanced_df = interaction_extraction(enhanced_df, numeric_cols)
    enhanced_df = categorical_extraction(enhanced_df)
    enhanced_df = log_transform_extraction(enhanced_df, numeric_cols)

    print(f"\n  Features after extraction: {len(enhanced_df.columns)}")

    print(f"\n{'-' * 80}")
    print("  FEATURE REDUCTION")
    print(f"{'-' * 80}")

    reduced_df, var_removed = variance_reduction(enhanced_df)
    reduced_df, corr_removed = correlation_reduction(reduced_df)
    reduced_df, kb_removed = selectkbest_reduction(reduced_df)
    reduced_df, rfe_removed = rfe_reduction(reduced_df)
    reduced_df, lasso_removed = lasso_reduction(reduced_df)

    print(f"\n  Features after reduction: {len(reduced_df.columns)}")

    out_extracted = os.path.join(output_dir, "statistical_features_extracted.csv")
    enhanced_df.to_csv(out_extracted, index=False)
    print(f"\n  Extracted features saved: {out_extracted}")

    out_reduced = os.path.join(output_dir, "statistical_features_reduced.csv")
    reduced_df.to_csv(out_reduced, index=False)
    print(f"  Reduced features saved: {out_reduced}")
    print(f"  Original features: {len(df.columns)}")
    print(f"  Enhanced features: {len(enhanced_df.columns)}")

    visualize_extraction(df, enhanced_df, numeric_cols, output_dir)

    print(f"\n{'-' * 80}")
    print("  EXTRACTION COMPLETE")
    print(f"{'-' * 80}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
