"""
Honest test: does SMOTE data augmentation help? (no downloads — numpy + sklearn only)

SMOTE is applied STRICTLY INSIDE each training fold (never on the test fold, never before
the split), then the conflict-class F1 is compared with/without augmentation under both
stratified and leave-one-repository-out. This is the only leakage-free way to evaluate it.

Usage:  python train_augment.py [all.csv]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]
rng = np.random.default_rng(0)


def smote(Xmin, n_new, k=5):
    if len(Xmin) <= 1 or n_new <= 0:
        return np.empty((0, Xmin.shape[1]))
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    out = []
    for _ in range(n_new):
        i = rng.integers(len(Xmin))
        nbrs = nn.kneighbors(Xmin[i:i + 1], return_distance=False)[0][1:]
        j = nbrs[rng.integers(len(nbrs))] if len(nbrs) else i
        out.append(Xmin[i] + rng.random() * (Xmin[j] - Xmin[i]))
    return np.array(out)


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=0, n_jobs=-1)


def run(X, y, splitter, groups=None, augment=False):
    proba = np.zeros(len(y))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        Xtr, ytr = X[tr], y[tr]
        if augment:
            Xmin = Xtr[ytr == 1]; nmaj = int((ytr == 0).sum()); nmin = len(Xmin)
            syn = smote(Xmin, max(0, nmaj - nmin))          # balance minority up to majority
            if len(syn):
                Xtr = np.vstack([Xtr, syn]); ytr = np.concatenate([ytr, np.ones(len(syn), int)])
        m = rf(); m.fit(Xtr, ytr); proba[te] = m.predict_proba(X[te])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return (precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
            f1_score(y, pred, zero_division=0), average_precision_score(y, proba))


def main():
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "all.csv")
    for c in GIT:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = df[GIT].to_numpy(float); y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    g = df["repo"].to_numpy()
    print("=" * 72)
    print(f"SMOTE AUGMENTATION TEST  |  {len(df)} scenarios, {int(y.sum())} conflicts")
    print("SMOTE applied inside training folds only. Conflict-class metrics.")
    print("=" * 72)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    gkf = GroupKFold(n_splits=df["repo"].nunique())
    for proto, sp, grp in [("STRATIFIED (within-project)", skf, None),
                           ("LEAVE-REPO-OUT (cross-project)", gkf, g)]:
        print(f"\n### {proto} ###")
        print(f"{'setting':<22}{'P':>7}{'R':>7}{'F1':>7}{'PR-AUC':>8}")
        for name, aug in [("baseline (class-weight)", False), ("+ SMOTE augmentation", True)]:
            p, r, f, ap = run(X, y, sp, grp, augment=aug)
            print(f"{name:<22}{p:>7.3f}{r:>7.3f}{f:>7.3f}{ap:>8.3f}")


if __name__ == "__main__":
    main()
