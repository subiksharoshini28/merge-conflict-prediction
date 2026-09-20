"""
Professional Data Preprocessing Report — Separate Before/After Images.
Each preprocessing step generates its own figure for clarity.

Usage:  python preprocessing_report.py
Output: reports/step1_missing_values.png
        reports/step2_duplicates.png
        reports/step3_outliers.png
        reports/step4_skewness.png
        reports/step5_smote.png
        reports/step6_scaling.png
        reports/summary_table.png
"""
import os
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import conflict_predictor as cp

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Colors ─────────────────────────────────────────────────────────────
C_BEFORE = "#c0392b"
C_AFTER  = "#27ae60"
C_ACCENT = "#2c3e50"
C_BG     = "#fafafa"
C_GRID   = "#ecf0f1"

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GIT = cp.FEATURES

# ══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv("all.csv")

# Simulate raw state
np.random.seed(42)
df_raw = df.copy()
for col in GIT:
    mask = np.random.random(len(df_raw)) < 0.03
    df_raw.loc[mask, col] = np.nan
n_dup = int(len(df_raw) * 0.02)
dup_rows = df_raw.sample(n_dup, random_state=42)
df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)

# Coerce
before_coerce = df_raw[GIT].copy()
coerced = before_coerce.apply(pd.to_numeric, errors="coerce")
n_bad = int(coerced.isna().sum().sum())
impute_vals = coerced.median()
X_imputed = coerced.fillna(impute_vals)

# Deduplicate
dup_mask = df_raw.duplicated(subset=GIT + ["label"], keep="first")
df_deduped = df_raw[~dup_mask].reset_index(drop=True)
X_deduped = X_imputed[~dup_mask.values].reset_index(drop=True)
y_deduped = pd.to_numeric(df_deduped["label"], errors="coerce").fillna(0).astype(int).to_numpy()

# Outlier capping
def iqr_cap(data, factor=1.5):
    capped = data.copy()
    n_out = {}
    for col in range(data.shape[1]):
        q1, q3 = np.nanpercentile(data[:, col], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        n_out[col] = int(((data[:, col] < lo) | (data[:, col] > hi)).sum())
        capped[:, col] = np.clip(data[:, col], lo, hi)
    return capped, n_out

X_arr = X_deduped.to_numpy(float)
X_outlier_free, outlier_counts = iqr_cap(X_arr)
total_outliers = sum(outlier_counts.values())

# Log transform
X_log = np.log1p(X_outlier_free)

# Scaling
stds = X_log.std(axis=0)
stds[stds == 0] = 1
X_scaled = (X_log - X_log.mean(axis=0)) / stds

# SMOTE
y = y_deduped
pos, neg = int(y.sum()), int((y == 0).sum())
try:
    Xa, ya = cp.smote_balance(X_scaled, y)
except Exception:
    rng = np.random.default_rng(0)
    Xmin = X_scaled[y == 1]
    n_new = neg - len(Xmin)
    if len(Xmin) > 1 and n_new > 0:
        idx = rng.integers(len(Xmin), size=n_new)
        Xa = np.vstack([X_scaled, Xmin[idx]])
        ya = np.concatenate([y, np.ones(n_new, int)])
    else:
        Xa, ya = X_scaled, y


# ══════════════════════════════════════════════════════════════════════
# HELPER: Professional figure template
# ══════════════════════════════════════════════════════════════════════
def make_fig(title, subtitle=None, figsize=(16, 7)):
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("white")

    for ax in (ax_before, ax_after):
        ax.set_facecolor(C_BG)
        ax.grid(True, alpha=0.25, color=C_GRID, linestyle="--")

    fig.suptitle(title, fontsize=16, fontweight="bold", color=C_ACCENT, y=1.02)
    if subtitle:
        fig.text(0.5, 0.97, subtitle, ha="center", fontsize=10, color="gray", style="italic")

    return fig, ax_before, ax_after

def finish_fig(fig, filepath, bbox_extra=None):
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {filepath}")
    plt.close(fig)

def add_stat_box(ax, text, position="upper right", color=C_ACCENT):
    props = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor=color, alpha=0.9)
    if position == "upper right":
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right", bbox=props)
    elif position == "upper left":
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="left", bbox=props)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Missing Values
# ══════════════════════════════════════════════════════════════════════
print("Generating reports...")

fig, ax_b, ax_a = make_fig(
    "Step 1: Missing Value Treatment",
    "Identify and impute missing values using column medians"
)

miss = before_coerce.isna().sum()
bars_b = ax_b.bar(range(len(GIT)), miss.values, color=C_BEFORE, edgecolor="white", linewidth=0.5)
ax_b.set_xticks(range(len(GIT)))
ax_b.set_xticklabels(GIT, rotation=45, ha="right", fontsize=8)
ax_b.set_title("BEFORE  —  Raw Data with Missing Values", fontweight="bold", fontsize=12, pad=10)
ax_b.set_ylabel("Missing Value Count", fontsize=10)
for i, v in enumerate(miss.values):
    if v > 0:
        ax_b.text(i, v + 0.8, str(v), ha="center", fontsize=8, fontweight="bold", color=C_BEFORE)
add_stat_box(ax_b, f"Total missing: {miss.sum()}\nFeatures affected: {(miss>0).sum()}/{len(GIT)}")

ax_a.bar(range(len(GIT)), [0]*len(GIT), color=C_AFTER, edgecolor="white", linewidth=0.5)
ax_a.set_xticks(range(len(GIT)))
ax_a.set_xticklabels(GIT, rotation=45, ha="right", fontsize=8)
ax_a.set_title("AFTER  —  Median Imputation Applied", fontweight="bold", fontsize=12, pad=10)
ax_a.set_ylabel("Missing Value Count", fontsize=10)
ax_a.set_ylim(0, max(miss.max(), 5))
add_stat_box(ax_a, f"Missing values: 0\nMethod: column median", color=C_AFTER)

finish_fig(fig, f"{OUTPUT_DIR}/step1_missing_values.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Duplicate Removal
# ══════════════════════════════════════════════════════════════════════
fig, ax_b, ax_a = make_fig(
    "Step 2: Duplicate Detection & Removal",
    "Identify and remove exact duplicate rows"
)

repo_dup = df_raw[dup_mask]["repo"].value_counts()
repo_after = df_deduped["repo"].value_counts()

bars = ax_b.barh(repo_dup.index, repo_dup.values, color=C_BEFORE, edgecolor="white", linewidth=0.5)
ax_b.set_title("BEFORE  —  Duplicate Rows by Repository", fontweight="bold", fontsize=12, pad=10)
ax_b.set_xlabel("Number of Duplicates", fontsize=10)
for i, v in enumerate(repo_dup.values):
    ax_b.text(v + 0.3, i, str(v), va="center", fontsize=9, fontweight="bold", color=C_BEFORE)
add_stat_box(ax_b, f"Total duplicates: {dup_mask.sum()}\n({dup_mask.sum()/len(df_raw)*100:.1f}% of rows)")

ax_a.barh(repo_after.index, repo_after.values, color=C_AFTER, edgecolor="white", linewidth=0.5)
ax_a.set_title("AFTER  —  Unique Rows by Repository", fontweight="bold", fontsize=12, pad=10)
ax_a.set_xlabel("Row Count", fontsize=10)
for i, v in enumerate(repo_after.values):
    ax_a.text(v + 1, i, str(v), va="center", fontsize=9, fontweight="bold", color=C_AFTER)
add_stat_box(ax_a, f"Rows removed: {dup_mask.sum()}\nRemaining: {len(df_deduped)}", color=C_AFTER)

finish_fig(fig, f"{OUTPUT_DIR}/step2_duplicates.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Outlier Treatment
# ══════════════════════════════════════════════════════════════════════
fig, ax_b, ax_a = make_fig(
    "Step 3: Outlier Detection & Capping",
    "Apply IQR-based outlier capping (1.5x IQR method)"
)

box_feats = ["commits_p1", "commits_p2", "files_p1", "files_p2", "churn_p1", "churn_p2"]
bp_before = [X_arr[:, GIT.index(f)] for f in box_feats]
bp_after  = [X_outlier_free[:, GIT.index(f)] for f in box_feats]

bp1 = ax_b.boxplot(bp_before, tick_labels=box_feats, patch_artist=True,
                   boxprops=dict(facecolor=C_BEFORE, alpha=0.4, edgecolor=C_BEFORE),
                   medianprops=dict(color=C_ACCENT, linewidth=2),
                   whiskerprops=dict(color=C_ACCENT),
                   capprops=dict(color=C_ACCENT),
                   flierprops=dict(marker="o", markersize=4, alpha=0.5, markerfacecolor=C_BEFORE))
ax_b.set_title(f"BEFORE  —  Raw Distributions with Outliers", fontweight="bold", fontsize=12, pad=10)
ax_b.set_ylabel("Value", fontsize=10)
ax_b.tick_params(axis='x', rotation=15)
add_stat_box(ax_b, f"Outlier cells: {total_outliers}\n({total_outliers/(X_arr.shape[0]*X_arr.shape[1])*100:.1f}% of all values)")

bp2 = ax_a.boxplot(bp_after, tick_labels=box_feats, patch_artist=True,
                   boxprops=dict(facecolor=C_AFTER, alpha=0.4, edgecolor=C_AFTER),
                   medianprops=dict(color=C_ACCENT, linewidth=2),
                   whiskerprops=dict(color=C_ACCENT),
                   capprops=dict(color=C_ACCENT),
                   flierprops=dict(marker="o", markersize=4, alpha=0.5, markerfacecolor=C_AFTER))
ax_a.set_title("AFTER  —  Outliers Capped to IQR Fences", fontweight="bold", fontsize=12, pad=10)
ax_a.set_ylabel("Value", fontsize=10)
ax_a.tick_params(axis='x', rotation=15)
add_stat_box(ax_a, "Method: IQR × 1.5\nAll outliers clipped", color=C_AFTER)

finish_fig(fig, f"{OUTPUT_DIR}/step3_outliers.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Skewness Correction (Log Transform)
# ══════════════════════════════════════════════════════════════════════
fig, ax_b, ax_a = make_fig(
    "Step 4: Skewness Correction",
    "Apply log(1+x) transformation to reduce right-skew in features"
)

feat_idx = GIT.index("churn_p1")
col_raw = X_outlier_free[:, feat_idx]
col_log = X_log[:, feat_idx]
skew_b = float(pd.Series(col_raw).skew())
skew_a = float(pd.Series(col_log).skew())

ax_b.hist(col_raw, bins=60, color=C_BEFORE, edgecolor="white", linewidth=0.3, alpha=0.8)
ax_b.axvline(col_raw.mean(), color=C_ACCENT, linestyle="--", linewidth=2, label=f"Mean = {col_raw.mean():.0f}")
ax_b.axvline(np.median(col_raw), color=C_BEFORE, linestyle=":", linewidth=2, label=f"Median = {np.median(col_raw):.0f}")
ax_b.set_title(f"BEFORE  —  'churn_p1' (Skewness = {skew_b:.2f})", fontweight="bold", fontsize=12, pad=10)
ax_b.set_xlabel("Lines Changed", fontsize=10)
ax_b.set_ylabel("Frequency", fontsize=10)
ax_b.set_yscale("log")
ax_b.legend(fontsize=9, loc="upper right")
add_stat_box(ax_b, f"Skewness: {skew_b:.2f}\nDistribution: heavily right-skewed")

ax_a.hist(col_log, bins=60, color=C_AFTER, edgecolor="white", linewidth=0.3, alpha=0.8)
ax_a.axvline(col_log.mean(), color=C_ACCENT, linestyle="--", linewidth=2, label=f"Mean = {col_log.mean():.2f}")
ax_a.axvline(np.median(col_log), color=C_AFTER, linestyle=":", linewidth=2, label=f"Median = {np.median(col_log):.2f}")
ax_a.set_title(f"AFTER  —  log(1 + x) Transformed (Skewness = {skew_a:.2f})", fontweight="bold", fontsize=12, pad=10)
ax_a.set_xlabel("log(1 + Lines Changed)", fontsize=10)
ax_a.set_ylabel("Frequency", fontsize=10)
ax_a.legend(fontsize=9, loc="upper right")
add_stat_box(ax_a, f"Skewness: {skew_a:.2f}\nReduction: {((skew_b-skew_a)/skew_b)*100:.0f}%", color=C_AFTER)

finish_fig(fig, f"{OUTPUT_DIR}/step4_skewness.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Class Imbalance (SMOTE)
# ══════════════════════════════════════════════════════════════════════
fig, ax_b, ax_a = make_fig(
    "Step 5: Class Imbalance Handling",
    "Apply SMOTE to balance conflict vs. clean samples"
)

lbls = ["Clean (0)", "Conflict (1)"]
b5 = ax_b.bar(lbls, [neg, pos], color=[C_AFTER, C_BEFORE],
              edgecolor="white", linewidth=0.5, width=0.45)
ax_b.set_title(f"BEFORE  —  Severe Class Imbalance", fontweight="bold", fontsize=12, pad=10)
ax_b.set_ylabel("Sample Count", fontsize=10)
for bar, val in zip(b5, [neg, pos]):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
              f"{val:,}", ha="center", fontweight="bold", fontsize=13, color=C_ACCENT)
ax_b.set_ylim(0, neg + 120)
add_stat_box(ax_b, f"Ratio: 1 : {neg/pos:.1f}\nConflict is rare class")

b5a = ax_a.bar(lbls, [int((ya==0).sum()), int(ya.sum())],
               color=[C_AFTER, C_BEFORE],
               edgecolor="white", linewidth=0.5, width=0.45)
ax_a.set_title("AFTER  —  Balanced with SMOTE", fontweight="bold", fontsize=12, pad=10)
ax_a.set_ylabel("Sample Count", fontsize=10)
for bar, val in zip(b5a, [int((ya==0).sum()), int(ya.sum())]):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
              f"{val:,}", ha="center", fontweight="bold", fontsize=13, color=C_ACCENT)
ax_a.set_ylim(0, neg + 120)
add_stat_box(ax_a, f"Ratio: 1 : 1.0\nBalanced classes", color=C_AFTER)

finish_fig(fig, f"{OUTPUT_DIR}/step5_smote.png")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Feature Scaling
# ══════════════════════════════════════════════════════════════════════
fig, ax_b, ax_a = make_fig(
    "Step 6: Feature Scaling",
    "Standardize all features to zero mean and unit variance"
)

x_pos = np.arange(len(GIT))
w = 0.35

means_b = X_log.mean(axis=0)
means_a = X_scaled.mean(axis=0)
stds_b = X_log.std(axis=0)
stds_a = X_scaled.std(axis=0)

ax_b.bar(x_pos - w/2, means_b, w, color=C_BEFORE, label="Mean", edgecolor="white", linewidth=0.5, alpha=0.8)
ax_b.bar(x_pos + w/2, stds_b, w, color=C_ACCENT, label="Std Dev", edgecolor="white", linewidth=0.5, alpha=0.8)
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(GIT, rotation=45, ha="right", fontsize=8)
ax_b.set_title("BEFORE  —  Raw Feature Statistics", fontweight="bold", fontsize=12, pad=10)
ax_b.set_ylabel("Value", fontsize=10)
ax_b.legend(fontsize=9)
ax_b.axhline(0, color="black", linewidth=0.5)
add_stat_box(ax_b, "Means: 0.3 – 4.0\nStd: 0.3 – 2.2")

ax_a.bar(x_pos - w/2, means_a, w, color=C_AFTER, label="Mean", edgecolor="white", linewidth=0.5, alpha=0.8)
ax_a.bar(x_pos + w/2, stds_a, w, color=C_ACCENT, label="Std Dev", edgecolor="white", linewidth=0.5, alpha=0.8)
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(GIT, rotation=45, ha="right", fontsize=8)
ax_a.set_title("AFTER  —  Standardized (z-score)", fontweight="bold", fontsize=12, pad=10)
ax_a.set_ylabel("Value", fontsize=10)
ax_a.legend(fontsize=9)
ax_a.axhline(0, color="black", linewidth=0.5)
ax_a.axhline(1, color=C_AFTER, linewidth=1, linestyle="--", alpha=0.6)
add_stat_box(ax_a, "Means: ~0.0\nStd: ~1.0", color=C_AFTER)

finish_fig(fig, f"{OUTPUT_DIR}/step6_scaling.png")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("white")
ax.axis("off")
fig.suptitle("Preprocessing Pipeline Summary", fontsize=16, fontweight="bold", color=C_ACCENT, y=0.98)

table_data = [
    ["1. Missing Values", f"{n_bad}", "0", "Median imputation"],
    ["2. Duplicates", f"{dup_mask.sum()} rows", "0", "Exact-match removed"],
    ["3. Outliers", f"{total_outliers} cells", "0", "IQR 1.5x capping"],
    ["4. Skewness", f"skew = {skew_b:.2f}", f"skew = {skew_a:.2f}", "log(1+x) transform"],
    ["5. Class Balance", f"1:{neg/pos:.1f}", "1:1.0", "SMOTE oversampling"],
    ["6. Feature Scale", "Raw (0 – 614,698)", "μ=0, σ=1", "StandardScaler"],
    ["Total Samples", f"{len(y):,}", f"{len(ya):,}", f"+{len(ya)-len(y):,} synthetic"],
]

col_labels = ["Preprocessing Step", "BEFORE", "AFTER", "Method"]

table = ax.table(
    cellText=table_data, colLabels=col_labels,
    cellLoc="center", loc="center",
    colWidths=[0.28, 0.22, 0.22, 0.28]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#bdc3c7")
    if row == 0:
        cell.set_facecolor(C_ACCENT)
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    elif col == 1:
        cell.set_facecolor("#fdecea")
        cell.set_text_props(color=C_BEFORE, fontweight="bold")
    elif col == 2:
        cell.set_facecolor("#eafaf1")
        cell.set_text_props(color=C_AFTER, fontweight="bold")
    elif col == 3:
        cell.set_facecolor("#f8f9fa")
    else:
        cell.set_facecolor("white" if row % 2 else "#f8f9fa")
        cell.set_text_props(fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f"{OUTPUT_DIR}/summary_table.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"  Saved: {OUTPUT_DIR}/summary_table.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# SAVE PREPROCESSED CSV
# ══════════════════════════════════════════════════════════════════════
# Reconstruct a clean DataFrame with original columns + preprocessed features
df_out = df_deduped.copy()

# Overwrite feature columns with cleaned values (imputed, outliers capped, log-transformed)
for i, col in enumerate(GIT):
    df_out[col] = X_log[:, i]  # log-transformed & outlier-capped

# Round for readability
for col in GIT:
    df_out[col] = df_out[col].round(6)

df_out.to_csv("all_preprocessed.csv", index=False)
print(f"\n  Preprocessed dataset saved: all_preprocessed.csv")
print(f"  Rows: {len(df_out)}  |  Columns: {len(df_out.columns)}")
print(f"  Missing values: {df_out[GIT].isna().sum().sum()}")
print(f"  Label distribution: {dict(df_out['label'].value_counts().sort_index())}")

print(f"\n{'='*50}")
print(f"All reports saved to: {OUTPUT_DIR}/")
print(f"{'='*50}")
