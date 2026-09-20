"""
Project visualizations for the merge-conflict predictor (current 11-feature model
with the log+IQR transform and divergence_days).

Generates a coherent figure set into figures/:
  1. transform_effect.png       raw vs log+IQR distributions (why we transform)
  2. feature_importance.png      Random-Forest importance, 11 features
  3. shap_summary.png            global SHAP mean|value| (TreeSHAP)
  4. feature_extraction.png      PCA and LDA 2-D projection, clean vs conflict
  5. correlation_heatmap.png     feature correlation (sequential, single hue)
  6. divergence_days.png         the adopted feature's signal, by class
  7. confusion_matrix.png        leave-one-repository-out confusion matrix

Usage:  python visualize.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import conflict_predictor as cp

# ---- house style (colorblind-safe Okabe-Ito pair, validated) --------------
CLEAN, CONFLICT = "#0072B2", "#D55E00"
HUE = "#0072B2"
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#e6e6e6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.edgecolor": "#888", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 130, "savefig.dpi": 150,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
})
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f7fb", HUE])
OUT = "figures"; os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", p)


def load():
    df = pd.read_csv(cp.DATA)
    for c in cp.FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


# 1. transform effect --------------------------------------------------------
def fig_transform(df):
    bundle = cp.load_model()
    cols = ["churn_p1", "overlap_files"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    for r, col in enumerate(cols):
        raw = df[col].to_numpy(float)
        j = cp.FEATURES.index(col)
        lo, hi = bundle["bounds"][j]
        t = np.log1p(np.clip(raw, lo, hi))
        axes[r, 0].hist(raw, bins=40, color=MUTED, alpha=.85)
        axes[r, 0].set_title(f"{col} — raw (skew {pd.Series(raw).skew():.1f})")
        axes[r, 1].hist(t, bins=40, color=HUE, alpha=.9)
        axes[r, 1].set_title(f"{col} — after IQR cap + log(x+1) (skew {pd.Series(t).skew():.2f})")
        for c in (0, 1):
            axes[r, c].set_ylabel("merges"); axes[r, c].set_yscale("log")
    fig.suptitle("Feature transformation squashes the long tail of skewed Git counts",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "transform_effect.png")


# 2. RF importance -----------------------------------------------------------
def fig_importance():
    bundle = cp.load_model()
    imp = sorted(zip(cp.FEATURES, bundle["model"].feature_importances_),
                 key=lambda t: t[1])
    names = [n for n, _ in imp]; vals = [v for _, v in imp]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(names, vals, color=HUE, height=.66)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * .01, y, f"{v:.3f}", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("Random-Forest importance"); ax.grid(axis="y", visible=False)
    ax.set_title("Feature importance (11-feature model)")
    ax.margins(x=.12)
    _save(fig, "feature_importance.png")


# 3. SHAP summary ------------------------------------------------------------
def fig_shap(df):
    import shap
    bundle = cp.load_model()
    X = cp.apply_transform(df[cp.FEATURES].to_numpy(float), bundle.get("bounds"))
    ex = shap.TreeExplainer(bundle["model"])
    arr = np.array(ex.shap_values(X))
    pos = arr[:, :, 1] if arr.ndim == 3 and arr.shape[-1] >= 2 else arr
    mean_abs = np.abs(pos).mean(axis=0)
    order = np.argsort(mean_abs)
    names = [cp.FEATURES[i] for i in order]; vals = mean_abs[order]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(names, vals, color=CONFLICT, height=.66)
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * .01, y, f"{v:.3f}", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("mean |SHAP value|  (impact on conflict probability)")
    ax.grid(axis="y", visible=False); ax.margins(x=.14)
    ax.set_title("Global explainability — exact TreeSHAP")
    _save(fig, "shap_summary.png")


# 4. PCA + LDA projection ----------------------------------------------------
def fig_extraction(df):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    bundle = cp.load_model()
    X = StandardScaler().fit_transform(cp.apply_transform(df[cp.FEATURES].to_numpy(float),
                                                          bundle.get("bounds")))
    y = df["label"].to_numpy()
    pca = PCA(2).fit_transform(X)
    lda1 = LinearDiscriminantAnalysis().fit(X, y).transform(X)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for cls, col, lab in [(0, CLEAN, "clean"), (1, CONFLICT, "conflict")]:
        m = y == cls
        axes[0].scatter(pca[m, 0], pca[m, 1], s=14, c=col, alpha=.55,
                        edgecolors="white", linewidths=.3, label=lab)
    axes[0].set_title("PCA projection (2 components)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(frameon=False)
    # LDA is 1-D for 2 classes -> show class-separated jittered strip / density
    rng = np.random.default_rng(0)
    for cls, col, lab in [(0, CLEAN, "clean"), (1, CONFLICT, "conflict")]:
        m = y == cls
        axes[1].scatter(lda1[m], rng.normal(cls, .06, m.sum()), s=14, c=col, alpha=.5,
                        edgecolors="white", linewidths=.3, label=lab)
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["clean", "conflict"])
    axes[1].set_xlabel("LDA discriminant"); axes[1].set_title("LDA projection (1 discriminant)")
    axes[1].legend(frameon=False)
    fig.suptitle("Feature extraction is used for visualization only — trees keep the original features",
                 fontsize=12.5, fontweight="bold", y=1.03)
    _save(fig, "feature_extraction.png")


# 5. correlation heatmap -----------------------------------------------------
def fig_corr(df):
    C = df[cp.FEATURES].corr().to_numpy()
    n = len(cp.FEATURES)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(C, cmap=SEQ, vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cp.FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cp.FEATURES, fontsize=8)
    for i in range(n):
        for jj in range(n):
            ax.text(jj, i, f"{C[i, jj]:.1f}", ha="center", va="center",
                    fontsize=6.5, color="#333")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("correlation")
    ax.set_title("Feature correlation (11 features)")
    _save(fig, "correlation_heatmap.png")


# 6. divergence_days signal --------------------------------------------------
def fig_divergence(df):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    data = [np.log1p(df.loc[df.label == 0, "divergence_days"]),
            np.log1p(df.loc[df.label == 1, "divergence_days"])]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=.55,
                    labels=["clean", "conflict"], showfliers=False)
    for patch, col in zip(bp["boxes"], [CLEAN, CONFLICT]):
        patch.set_facecolor(col); patch.set_alpha(.75); patch.set_edgecolor("#333")
    for med in bp["medians"]:
        med.set_color("#111"); med.set_linewidth(1.4)
    ax.set_ylabel("log(1 + divergence_days)")
    ax.set_title("Branch age (divergence_days) is higher for conflicting merges")
    ax.grid(axis="x", visible=False)
    _save(fig, "divergence_days.png")


# 7. confusion matrix (leave-repo-out) --------------------------------------
def fig_confusion(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import confusion_matrix
    X = df[cp.FEATURES].to_numpy(float); y = df["label"].to_numpy(); g = df["repo"].to_numpy()
    proba = np.zeros(len(y))
    for tr, te in GroupKFold(min(df["repo"].nunique(), 5)).split(X, y, g):
        b = cp.fit_transform_bounds(X[tr])
        Xa, ya = cp.smote_balance(cp.apply_transform(X[tr], b), y[tr])
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xa, ya)
        proba[te] = m.predict_proba(cp.apply_transform(X[te], b))[:, 1]
    cm = confusion_matrix(y, (proba >= .5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap=SEQ)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["clean", "conflict"]); ax.set_yticklabels(["clean", "conflict"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    mx = cm.max()
    for i in range(2):
        for jj in range(2):
            ax.text(jj, i, f"{cm[i, jj]}", ha="center", va="center", fontsize=15,
                    color="white" if cm[i, jj] > mx * .5 else INK, fontweight="bold")
    ax.grid(False)
    ax.set_title("Confusion matrix — leave-one-repository-out")
    _save(fig, "confusion_matrix.png")


def main():
    df = load()
    print("Generating figures into", OUT, "/ ...")
    fig_transform(df)
    fig_importance()
    fig_shap(df)
    fig_extraction(df)
    fig_corr(df)
    fig_divergence(df)
    fig_confusion(df)
    print("done — 7 figures in", OUT)


if __name__ == "__main__":
    main()
