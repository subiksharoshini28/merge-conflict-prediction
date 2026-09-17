"""
Feature Extraction — transform the original features into new, derived ones
(dimensionality reduction). Covers the syllabus methods on the real dataset.

  PCA        linear, unsupervised (max variance)
  LDA        linear, supervised (max class separability)
  t-SNE      nonlinear, for 2-D visualisation
  KernelPCA  nonlinear PCA (RBF kernel)
  ICA        independent components

Produces a 2-D visualisation (feature_extraction.png), saves the PCA-reduced data
(features_extracted.csv), and HONESTLY reports whether reduction helps the Random
Forest (F1, stratified 5-fold) versus the original features.

No downloads — sklearn/matplotlib only (UMAP/autoencoder omitted as they need extra libs).
Usage:  python feature_extraction.py [all.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

FEATURES = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
            "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]


def rf_f1(X, y):
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                               min_samples_leaf=2, random_state=0, n_jobs=-1)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(m, X, y, cv=skf, method="predict")
    return f1_score(y, pred, zero_division=0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    df = pd.read_csv(path)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    X = StandardScaler().fit_transform(df[FEATURES].to_numpy(float))
    print("=" * 72)
    print(f"FEATURE EXTRACTION  |  {X.shape[0]} scenarios, {X.shape[1]} features -> reduced")
    print("=" * 72)

    # ---- PCA ----
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.argmax(cum >= 0.95) + 1)
    print(f"\n[PCA] variance explained: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
          f"PC1-2 {cum[1]:.1%}; {k95} components reach 95%")
    Xp2 = PCA(n_components=2, random_state=0).fit_transform(X)
    Xp5 = PCA(n_components=5, random_state=0).fit_transform(X)

    # ---- LDA (supervised, 1 component for binary) ----
    Xl = LDA(n_components=1).fit_transform(X, y)
    print(f"[LDA] projected to 1 supervised component (class-separating)")

    # ---- t-SNE (visualisation) ----
    print("[t-SNE] computing 2-D embedding (this takes a moment)...")
    Xt = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X)

    # ---- KernelPCA + ICA ----
    Xk = KernelPCA(n_components=2, kernel="rbf", gamma=0.1, random_state=0).fit_transform(X)
    Xi = FastICA(n_components=2, random_state=0, max_iter=500).fit_transform(X)

    # ---- honest effect on the model ----
    print("\n[EFFECT ON MODEL] Random Forest F1 (stratified 5-fold, conflict class)")
    base = rf_f1(X, y)
    print(f"   original 10 features        F1 = {base:.3f}")
    print(f"   PCA (5 components)          F1 = {rf_f1(Xp5, y):.3f}")
    print(f"   LDA (1 component)           F1 = {rf_f1(Xl, y):.3f}")
    print("   (For tree models, PCA usually does NOT beat the original interpretable")
    print("    features; extraction is used here mainly for visualisation.)")

    # ---- save PCA-reduced dataset ----
    out = df[["repo", "merge"]].copy()
    for i in range(5):
        out[f"pca_{i+1}"] = Xp5[:, i]
    out["lda_1"] = Xl[:, 0]; out["label"] = y
    out.to_csv("features_extracted.csv", index=False)

    # ---- visualisation ----
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for A, XX, title in [(ax[0], Xp2, "PCA (2 components)"),
                         (ax[2], Xt, "t-SNE (2-D)")]:
        A.scatter(XX[y == 0, 0], XX[y == 0, 1], s=8, c="#9aa0a6", label="clean", alpha=.6)
        A.scatter(XX[y == 1, 0], XX[y == 1, 1], s=14, c="#d63b3b", label="conflict", alpha=.8)
        A.set_title(title); A.legend(fontsize=8)
    ax[1].hist(Xl[y == 0, 0], bins=40, color="#9aa0a6", alpha=.7, label="clean")
    ax[1].hist(Xl[y == 1, 0], bins=40, color="#d63b3b", alpha=.7, label="conflict")
    ax[1].set_title("LDA (1 supervised component)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("feature_extraction.png", dpi=130)
    print("\nSaved: features_extracted.csv, feature_extraction.png")


if __name__ == "__main__":
    main()
