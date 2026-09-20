"""
================================================================================
 MERGE CONFLICT PREDICTOR  -  unified framework entry point
================================================================================
One command-line tool that ties the whole pipeline together:

  python conflict_predictor.py train
        Train the model on the mined dataset (all.csv) and save it (model.pkl).

  python conflict_predictor.py predict <repo> <branchA> <branchB>
        The product: for two branches of a real repo, predict BEFORE merging -
        conflict probability + risk score (LOW/MEDIUM/HIGH) + why + recommended action.

  python conflict_predictor.py demo
        Run a full prediction report on the highest-risk merge already in the
        dataset (shows the tool working without needing two live branches).

Features are computed at the merge base (point-in-time; no leakage). Textual-conflict
model; the dependency-graph / semantic layer is documented in Dataset_and_Findings.
================================================================================
"""
import subprocess, sys, os, pickle
import numpy as np

MODEL = "model.pkl"
DATA = "all.csv"
FEATURES = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
            "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2",
            "divergence_days"]

ACTION = {
    "overlap_files":  "Both branches edit the same file(s) - coordinate on them or merge the smaller branch first.",
    "overlap_ratio":  "High share of overlapping files - split the work or integrate early in small steps.",
    "churn_p1":       "Branch A changes a large volume of code - merge it sooner / in smaller increments.",
    "churn_p2":       "Branch B changes a large volume of code - merge it sooner / in smaller increments.",
    "files_p1":       "Branch A touches many files - review the integration surface early.",
    "files_p2":       "Branch B touches many files - review the integration surface early.",
    "commits_p1":     "Branch A has diverged over many commits - rebase / integrate more frequently.",
    "commits_p2":     "Branch B has diverged over many commits - rebase / integrate more frequently.",
    "authors_p1":     "Many contributors on branch A - align via code owners before merging.",
    "authors_p2":     "Many contributors on branch B - align via code owners before merging.",
}


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos")


def resolve_repo(repo, fetch=False):
    """Accept a local path OR a git URL. For a URL, clone it (blobless) into a cache and
    return the local path, so predictions can run against a real GitHub repo directly."""
    if repo.startswith(("http://", "https://", "git@", "ssh://")):
        os.makedirs(CACHE, exist_ok=True)
        name = repo.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        dest = os.path.join(CACHE, name)
        if not os.path.isdir(os.path.join(dest, ".git")):
            print(f"cloning {repo} ...", file=sys.stderr)
            subprocess.run(["git", "clone", "--filter=blob:none", "--quiet", repo, dest], check=False)
        elif fetch:
            subprocess.run(["git", "-C", dest, "fetch", "--quiet", "--all"], check=False)
        return dest
    return repo


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def resolve_ref(repo, ref):
    """Map a branch name to a resolvable ref: try it directly, else origin/<ref>
    (a fresh clone keeps non-default branches only as remote-tracking refs)."""
    if g(repo, "rev-parse", "--verify", "-q", ref + "^{commit}"):
        return ref
    if g(repo, "rev-parse", "--verify", "-q", "origin/" + ref + "^{commit}"):
        return "origin/" + ref
    return ref


def side(repo, base, tip):
    """git-history features for one branch relative to the base."""
    n = g(repo, "rev-list", "--count", base + ".." + tip)
    files = set(g(repo, "diff", "--name-only", base, tip).splitlines())
    authors = len(set(g(repo, "log", "--format=%ae", base + ".." + tip).splitlines()))
    ins = dels = 0
    for p in g(repo, "diff", "--shortstat", base, tip).split(", "):
        if "insertion" in p: ins = int(p.split()[0])
        elif "deletion" in p: dels = int(p.split()[0])
    return int(n or 0), files, authors, ins + dels


def _cdate(repo, sha):
    """Unix commit timestamp of a ref (0 if unknown)."""
    out = g(repo, "show", "-s", "--format=%ct", sha)
    try:
        return int(out.splitlines()[0])
    except Exception:
        return 0


def feature_vector(repo, base, a, b):
    c1, f1, a1, ch1 = side(repo, base, a)
    c2, f2, a2, ch2 = side(repo, base, b)
    ov, un = f1 & f2, f1 | f2
    # branch age: base -> newest branch tip. Leak-free and available before the merge
    # (a, b are the un-merged branch tips), matching how the training column was built.
    div = max(0, max(_cdate(repo, a), _cdate(repo, b)) - _cdate(repo, base)) / 86400.0
    return {"commits_p1": c1, "commits_p2": c2, "files_p1": len(f1), "files_p2": len(f2),
            "overlap_files": len(ov), "overlap_ratio": round(len(ov) / len(un), 4) if un else 0,
            "authors_p1": a1, "authors_p2": a2, "churn_p1": ch1, "churn_p2": ch2,
            "divergence_days": round(div, 3)}


def risk_band(p):
    return "HIGH" if p >= 0.50 else "MEDIUM" if p >= 0.20 else "LOW"


def smote_balance(X, y, k=5, seed=0):
    """SMOTE: interpolate synthetic minority (conflict) samples up to the majority count.
    Leakage-free ONLY when applied to a training set (never to test data)."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]; n_new = int((y == 0).sum()) - len(Xmin)
    if len(Xmin) <= 1 or n_new <= 0:
        return X, y
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    out = []
    for _ in range(n_new):
        i = rng.integers(len(Xmin))
        nbrs = nn.kneighbors(Xmin[i:i + 1], return_distance=False)[0][1:]
        j = nbrs[rng.integers(len(nbrs))] if len(nbrs) else i
        out.append(Xmin[i] + rng.random() * (Xmin[j] - Xmin[i]))
    return np.vstack([X, np.array(out)]), np.concatenate([y, np.ones(len(out), int)])


# ---- feature transform: IQR outlier capping + log(x+1) on skewed count features ----
LOG_COLS = [c for c in FEATURES if c != "overlap_ratio"]
LOG_IDX = [FEATURES.index(c) for c in LOG_COLS]


def fit_transform_bounds(X):
    """Learn per-feature IQR caps on the TRAINING data (stored in the model)."""
    b = []
    for j in range(X.shape[1]):
        q1, q3 = np.percentile(X[:, j], 25), np.percentile(X[:, j], 75)
        iqr = q3 - q1
        b.append((max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr))
    return b


def apply_transform(X, bounds):
    """Clip to the learned IQR bounds, then log(x+1) the count features."""
    if not bounds:
        return np.array(X, float)
    X = np.array(X, float).copy()
    for j, (lo, hi) in enumerate(bounds):
        X[:, j] = np.clip(X[:, j], lo, hi)
    for j in LOG_IDX:
        X[:, j] = np.log1p(X[:, j])
    return X


# ------------------------------------------------------------------ train
def train(data=DATA, out=MODEL):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(data)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    X = df[FEATURES].to_numpy(float)
    bounds = fit_transform_bounds(X)          # IQR caps learned on training data
    Xt = apply_transform(X, bounds)           # cap + log(x+1)
    Xa, ya = smote_balance(Xt, y)             # SMOTE on the transformed features
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xa, ya)
    clean_baseline = np.median(Xt[y == 0], axis=0)   # baseline in TRANSFORMED space
    pickle.dump({"model": model, "features": FEATURES, "clean_baseline": clean_baseline,
                 "bounds": bounds}, open(out, "wb"))
    print(f"trained on {len(df)} scenarios ({int(y.sum())} conflicts) -> saved {out}")


def load_model(path=MODEL):
    if not os.path.exists(path):
        print(f"No model found ({path}). Run:  python conflict_predictor.py train")
        sys.exit(1)
    return pickle.load(open(path, "rb"))


def explain_vector(bundle, vec):
    """Counterfactual attribution: prob drop when each feature -> typical clean value."""
    model, feats, base = bundle["model"], bundle["features"], bundle["clean_baseline"]
    xr = np.array([vec[f] for f in feats], float)                 # raw values (for display)
    x = apply_transform(xr.reshape(1, -1), bundle.get("bounds"))[0]  # transformed (for the model)
    p = model.predict_proba(x.reshape(1, -1))[0, 1]
    drivers = []
    for i, f in enumerate(feats):
        xc = x.copy(); xc[i] = base[i]
        drop = p - model.predict_proba(xc.reshape(1, -1))[0, 1]
        drivers.append((f, drop, xr[i], base[i]))               # show RAW value
    drivers.sort(key=lambda t: t[1], reverse=True)
    return p, drivers


def shap_vector(bundle, vec):
    """Exact TreeSHAP attribution (positive/conflict class). Returns [(feature, shap, value)]."""
    import shap
    ex = shap.TreeExplainer(bundle["model"])
    feats = bundle["features"]
    x = apply_transform(np.array([[vec[f] for f in feats]], float), bundle.get("bounds"))
    arr = np.array(ex.shap_values(x))
    v = arr[0, :, 1] if arr.ndim == 3 and arr.shape[-1] >= 2 else (arr[0] if arr.ndim == 2 else arr)
    return [(f, float(s), float(vec[f])) for f, s in zip(feats, v)]


def _print_attr(title, triples):
    """triples: [(feature, impact, value)] -> text tornado."""
    triples = sorted(triples, key=lambda t: -abs(t[1]))
    mx = max((abs(i) for _, i, _ in triples), default=1) or 1
    print(f"\n  {title}")
    print(f"    {'feature':<15}{'value':>11}{'impact':>9}   attribution")
    for f, imp, val in triples:
        n = int(round(abs(imp) / mx * 22))
        arrow = "-> CONFLICT" if imp > 0 else "-> clean" if imp < 0 else ""
        print(f"    {f:<15}{val:>11.3g}{imp:>+9.3f}   {'#' * n} {arrow}")


def report(header, vec, bundle):
    p, drivers = explain_vector(bundle, vec)
    band = risk_band(p)
    bar = {"HIGH": "##########", "MEDIUM": "#####     ", "LOW": "##        "}[band]
    print("=" * 60)
    print(" MERGE CONFLICT PREDICTION")
    for line in header:
        print(" " + line)
    print("=" * 60)
    print(f"  Conflict probability : {p:.2f}")
    print(f"  Risk score           : {band}  [{bar}]")
    print("\n  Why (top drivers):")
    shown = 0
    for f, drop, val, b in drivers:
        if drop <= 0 or shown >= 3:
            continue
        print(f"    - {f} = {val:g}  (typical clean {b:g})")
        shown += 1
    if shown == 0:
        print("    - no strong risk drivers; features look like a typical clean merge")
    print("\n  Recommended action:")
    if band == "LOW" or drivers[0][1] <= 0:
        print("    -> Low risk - the merge looks safe; proceed with standard review.")
    else:
        print(f"    -> {ACTION.get(drivers[0][0], 'Monitor the change surface during integration.')}")
    print("=" * 60)


# ------------------------------------------------------------------ predict
def predict(repo, a, b):
    repo = resolve_repo(repo, fetch=True)   # accepts a GitHub URL or a local path
    if not os.path.isdir(os.path.join(repo, ".git")):
        print(f"Not a git repo (or clone failed): {repo}"); sys.exit(1)
    a, b = resolve_ref(repo, a), resolve_ref(repo, b)
    base = g(repo, "merge-base", a, b)
    if not base:
        print(f"No common ancestor for {a} and {b} (are the branch names correct?)"); sys.exit(1)
    vec = feature_vector(repo, base, a, b)
    bundle = load_model()
    report([f"repo       : {os.path.basename(repo.rstrip('/\\'))}",
            f"branch A   : {a}", f"branch B   : {b}",
            f"merge base : {base[:12]}"], vec, bundle)


# ------------------------------------------------------------------ demo
def demo(data=DATA):
    import pandas as pd
    bundle = load_model()
    df = pd.read_csv(data)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = apply_transform(df[FEATURES].to_numpy(float), bundle.get("bounds"))
    proba = bundle["model"].predict_proba(X)[:, 1]
    idx = int(np.argmax(proba))
    row = df.iloc[idx]
    vec = {f: row[f] for f in FEATURES}
    report([f"(demo on dataset row - highest predicted risk)",
            f"repo   : {row['repo']}", f"merge  : {row['merge'][:12]}",
            f"actual : {'CONFLICT' if int(row['label']) == 1 else 'clean'}"], vec, bundle)


def explain(repo, a, b):
    """Predict + print BOTH counterfactual and exact SHAP attribution in the terminal."""
    repo = resolve_repo(repo, fetch=True)
    if not os.path.isdir(os.path.join(repo, ".git")):
        print(f"Not a git repo (or clone failed): {repo}"); sys.exit(1)
    a, b = resolve_ref(repo, a), resolve_ref(repo, b)
    base = g(repo, "merge-base", a, b)
    if not base:
        print(f"No common ancestor for {a} and {b}"); sys.exit(1)
    vec = feature_vector(repo, base, a, b)
    bundle = load_model()
    p, drivers = explain_vector(bundle, vec)
    print("=" * 62)
    print(f" XAI EXPLANATION  |  {os.path.basename(repo.rstrip('/\\'))}  A:{a[:14]}  B:{b[:14]}")
    print("=" * 62)
    print(f"  Conflict probability : {p:.2f}   Risk : {risk_band(p)}")
    _print_attr("Counterfactual attribution (prob drop vs. typical clean):",
                [(f, dr, v) for f, dr, v, cb in drivers])
    try:
        _print_attr("Exact SHAP attribution (TreeSHAP, sums to the prediction):",
                    [(f, s, v) for f, s, v in shap_vector(bundle, vec)])
    except ImportError:
        print("\n  (SHAP not installed -> pip install shap  for exact Shapley attribution)")
    print("=" * 62)


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    cmd = a[0]
    if cmd == "train":
        train()
    elif cmd == "predict":
        if len(a) < 4:
            print("usage: python conflict_predictor.py predict <repo> <branchA> <branchB>"); return
        predict(a[1], a[2], a[3])
    elif cmd == "explain":
        if len(a) < 4:
            print("usage: python conflict_predictor.py explain <repo> <branchA> <branchB>"); return
        explain(a[1], a[2], a[3])
    elif cmd == "demo":
        demo()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
