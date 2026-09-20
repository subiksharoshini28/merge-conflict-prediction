"""
Complete Feature Extraction & Reduction Methods for CSV Files

Extraction: Statistical, Correlation-based, Interaction, Categorical, Binning
Reduction: Filter (Variance, Correlation, MI, ANOVA), Wrapper (RFE), Embedded (RF, Lasso)

Usage:  python csv_feature_methods.py [input.csv]
Output: reports/feature_extraction_full.png
        reports/extracted_features.csv
        reports/reduced_features.csv
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
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

DIVIDER = "=" * 80


# ============================================================================
# PART 1: FEATURE EXTRACTION
# ============================================================================

def statistical_extraction(df, numeric_cols):
    print("\n  [1/7] Statistical Feature Extraction...")
    result = df.copy()
    stats_features = []

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
        stats_features.extend([
            f"{col}_mean", f"{col}_median", f"{col}_std", f"{col}_skew",
            f"{col}_kurtosis", f"{col}_min", f"{col}_max", f"{col}_range",
            f"{col}_q25", f"{col}_q75", f"{col}_iqr"
        ])

    print(f"        Added {len(stats_features)} statistical features")
    return result


def correlation_based_extraction(df, numeric_cols):
    print("\n  [2/7] Correlation-based Feature Extraction...")
    result = df.copy()
    X = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(2, X.shape[1]), random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    for i in range(X_pca.shape[1]):
        result[f"PCA_{i+1}"] = X_pca[:, i]
    print(f"        PCA: explained variance = {pca.explained_variance_ratio_}")

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


def interaction_extraction(df, numeric_cols):
    print("\n  [3/7] Interaction Feature Extraction...")
    result = df.copy()
    new_features = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]

            result[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-10)
            new_features.append(f"{col1}_div_{col2}")

            result[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
            new_features.append(f"{col1}_minus_{col2}")

            result[f"{col1}_mul_{col2}"] = df[col1] * df[col2]
            new_features.append(f"{col1}_mul_{col2}")

    print(f"        Added {len(new_features)} interaction features (ratio, difference, polynomial)")
    return result


def categorical_extraction(df):
    print("\n  [4/7] Categorical Feature Extraction...")
    result = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    new_features = []

    for col in cat_cols:
        if col in ["conflict_files"]:
            continue

        dummies = pd.get_dummies(df[col], prefix=f"{col}_onehot")
        result = pd.concat([result, dummies], axis=1)
        new_features.extend(dummies.columns.tolist())

        le = LabelEncoder()
        result[f"{col}_label"] = le.fit_transform(df[col].astype(str))
        new_features.append(f"{col}_label")

        freq = df[col].value_counts(normalize=True)
        result[f"{col}_freq"] = df[col].map(freq)
        new_features.append(f"{col}_freq")

        target_mean = df.groupby(col)["label"].mean() if "label" in df.columns else None
        if target_mean is not None:
            result[f"{col}_target"] = df[col].map(target_mean)
            new_features.append(f"{col}_target")

    print(f"        Categorical columns: {len(cat_cols)}")
    print(f"        Methods: One-Hot, Label, Frequency, Target Encoding")
    print(f"        Added {len(new_features)} categorical features")
    return result


def binning_extraction(df, numeric_cols):
    print("\n  [5/7] Binning/Discretization Extraction...")
    result = df.copy()
    new_features = []

    for col in numeric_cols[:5]:
        try:
            kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
            result[f"{col}_bin_uniform"] = kbins.fit_transform(df[[col]].fillna(0))
            new_features.append(f"{col}_bin_uniform")
        except Exception:
            pass

        try:
            kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            result[f"{col}_bin_quantile"] = kbins.fit_transform(df[[col]].fillna(0))
            new_features.append(f"{col}_bin_quantile")
        except Exception:
            pass

        try:
            km = KMeans(n_clusters=5, random_state=0, n_init=10)
            result[f"{col}_bin_kmeans"] = km.fit_predict(df[[col]].fillna(0))
            new_features.append(f"{col}_bin_kmeans")
        except Exception:
            pass

    print(f"        Methods: Equal-Width, Equal-Frequency, K-Means")
    print(f"        Added {len(new_features)} binned features")
    return result


def log_transform_extraction(df, numeric_cols):
    print("\n  [6/7] Log Transform Extraction...")
    result = df.copy()
    new_features = []

    for col in numeric_cols:
        if df[col].min() >= 0:
            result[f"{col}_log1p"] = np.log1p(df[col])
            new_features.append(f"{col}_log1p")

    print(f"        Added {len(new_features)} log-transformed features")
    return result


def domain_specific_extraction(df):
    print("\n  [7/7] Domain-Specific Feature Extraction...")
    result = df.copy()
    new_features = []

    if "overlap_files" in df.columns and "files_p1" in df.columns:
        result["domain_overlap_density"] = df["overlap_files"] / (df["files_p1"] + 1)
        new_features.append("domain_overlap_density")

    if "churn_p1" in df.columns and "churn_p2" in df.columns:
        result["domain_churn_ratio"] = df["churn_p1"] / (df["churn_p2"] + 1)
        new_features.append("domain_churn_ratio")

    if "commits_p1" in df.columns and "commits_p2" in df.columns:
        result["domain_commit_balance"] = abs(df["commits_p1"] - df["commits_p2"]) / (df["commits_p1"] + df["commits_p2"] + 1)
        new_features.append("domain_commit_balance")

    print(f"        Added {len(new_features)} domain-specific features")
    return result


# ============================================================================
# PART 2: FEATURE REDUCTION
# ============================================================================

def variance_reduction(df, threshold=0.01):
    print("\n  [1/7] Variance Threshold (Filter)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].fillna(0).values

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    kept_mask = selector.get_support()
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Removed {len(removed)} low-variance features (threshold={threshold})")
    return result


def correlation_reduction(df, threshold=0.95):
    print("\n  [2/7] Correlation Removal (Filter)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    removed = [col for col in upper.columns if any(upper[col] > threshold)]

    result = df.drop(columns=removed, errors="ignore")
    print(f"        Removed {len(removed)} highly correlated features (>{threshold})")
    return result


def mutual_information_reduction(df, label_col="label", k=10):
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


def anova_reduction(df, label_col="label", k=10):
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


def rfe_reduction(df, label_col="label", n_features=10):
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
    print(f"        Kept top {n_features} features by RFE (RandomForest)")
    return result


def rf_importance_reduction(df, label_col="label", threshold=0.01):
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
    importances = rf.feature_importances_
    kept_mask = importances > threshold
    kept_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if kept_mask[i]]
    removed = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Removed {len(removed)} features (importance < {threshold})")
    return result


def lasso_reduction(df, label_col="label", threshold=0.001):
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
    removed = [numeric_cols[i] for i in range(len(numeric_cols)) if not kept_mask[i]]

    result = df[kept_cols + [c for c in df.columns if c not in numeric_cols]].copy()
    print(f"        Kept {len(kept_cols)} features (coef > {threshold})")
    print(f"        Removed {len(removed)} zero-coefficient features")
    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(original_df, extracted_df, reduced_df, output_dir):
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
        axes[0, 1].set_xlabel(pca_cols[0])
        axes[0, 1].set_ylabel(pca_cols[1])

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

    reduction_methods = ["Variance", "Correlation", "MI", "ANOVA", "RFE", "RF", "Lasso"]
    feature_counts = [
        len(original_df.columns),
        len(extracted_df.columns),
        len(reduced_df.columns)
    ]
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
            axes[1, 1].barh(range(len(top_features)),
                           [reduced_df[f].corr(reduced_df["label"]) for f in top_features],
                           color="#9b59b6", alpha=0.8)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=8)
            axes[1, 1].set_title("Top Reduced Features (Correlation with Label)", fontweight="bold")
            axes[1, 1].set_xlabel("Correlation")

    axes[1, 2].axis("off")
    summary_text = (
        f"EXTRACTION METHODS:\n"
        f"  - Statistical (mean, median, std, skew, kurtosis)\n"
        f"  - Correlation-based (PCA, LDA, ICA)\n"
        f"  - Interaction (ratio, difference, polynomial)\n"
        f"  - Categorical (one-hot, label, frequency, target)\n"
        f"  - Binning (uniform, quantile, k-means)\n"
        f"  - Log Transform (log1p)\n"
        f"  - Domain-Specific\n\n"
        f"REDUCTION METHODS:\n"
        f"  - Filter: Variance, Correlation, MI, ANOVA\n"
        f"  - Wrapper: RFE\n"
        f"  - Embedded: RF Importance, Lasso L1"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment="center", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[1, 3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_extraction_full.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Chart saved: {out_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    print(DIVIDER)
    print("  FEATURE EXTRACTION & REDUCTION - Complete Pipeline")
    print(DIVIDER)

    print(f"\n  Loading: {path}")
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}, Numeric: {len(numeric_cols)}")

    print(f"\n{'-' * 80}")
    print("  PART 1: FEATURE EXTRACTION")
    print(f"{'-' * 80}")

    extracted_df = df.copy()
    extracted_df = statistical_extraction(extracted_df, numeric_cols)
    extracted_df = correlation_based_extraction(extracted_df, numeric_cols)
    extracted_df = interaction_extraction(extracted_df, numeric_cols)
    extracted_df = categorical_extraction(extracted_df)
    extracted_df = binning_extraction(extracted_df, numeric_cols)
    extracted_df = log_transform_extraction(extracted_df, numeric_cols)
    extracted_df = domain_specific_extraction(extracted_df)

    print(f"\n  Features after extraction: {len(extracted_df.columns)}")

    out_extracted = os.path.join(output_dir, "extracted_features.csv")
    extracted_df.to_csv(out_extracted, index=False)
    print(f"  Saved: {out_extracted}")

    print(f"\n{'-' * 80}")
    print("  PART 2: FEATURE REDUCTION")
    print(f"{'-' * 80}")

    reduced_df = extracted_df.copy()
    reduced_df = variance_reduction(reduced_df)
    reduced_df = correlation_reduction(reduced_df)
    reduced_df = mutual_information_reduction(reduced_df)
    reduced_df = anova_reduction(reduced_df)
    reduced_df = rfe_reduction(reduced_df)
    reduced_df = rf_importance_reduction(reduced_df)
    reduced_df = lasso_reduction(reduced_df)

    print(f"\n  Features after reduction: {len(reduced_df.columns)}")

    out_reduced = os.path.join(output_dir, "reduced_features.csv")
    reduced_df.to_csv(out_reduced, index=False)
    print(f"  Saved: {out_reduced}")

    visualize_results(df, extracted_df, reduced_df, output_dir)

    print(f"\n{'-' * 80}")
    print("  PIPELINE COMPLETE")
    print(f"{'-' * 80}")
    print(f"  Original features : {len(df.columns)}")
    print(f"  Extracted features: {len(extracted_df.columns)}")
    print(f"  Reduced features : {len(reduced_df.columns)}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
