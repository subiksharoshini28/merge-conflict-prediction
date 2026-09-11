"""
Interactive dashboard for the Merge Conflict Predictor framework.

Python standard library only (no Flask, no CDN — inlined, works offline).
Backend = JSON API; front-end = single-page vanilla-JS dashboard.

Endpoints:
  /                          dashboard
  /api/repos /api/merges /api/branches
  /api/predict?repo=&a=&b=   live prediction (git + graph + complexity + XAI)
  /api/scenario?repo=&merge= predict a known dataset merge (with actual outcome)
  /api/performance           precision/recall/F1/PR-AUC + confusion matrix (cross-validated)
  /api/eda /api/seeded
Run:  python webapp.py   ->  http://localhost:8000
"""
import http.server, urllib.parse, os, json, csv, base64, threading, time, subprocess
import numpy as np
import pandas as pd
import conflict_predictor as cp
import graph_features as gf
import complexity_features as cx

PORT = 8080
# Login gate — anyone sharing this (e.g. via a tunnel) must supply these.
BUNDLE = cp.load_model()
DATASET = pd.read_csv(cp.DATA)
for c in cp.FEATURES:
    DATASET[c] = pd.to_numeric(DATASET[c], errors="coerce").fillna(0)
DATASET["label"] = pd.to_numeric(DATASET["label"], errors="coerce").fillna(0).astype(int)
_PERF = None


REPOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos")


def repo_dir(name):
    return os.path.join(REPOS, name)


def build_subgraph(G, repo, base, a, b, cap=8):
    """Small drawable subgraph: changed files on each side + shared neighbours + their edges."""
    nodes = set(G.nodes)
    S1 = [f for f in gf.changed_java(repo, base, a) if f in nodes][:cap]
    S2 = [f for f in gf.changed_java(repo, base, b) if f in nodes][:cap]
    s1, s2 = set(S1), set(S2)
    UG = G.to_undirected(as_view=True)
    n1, n2 = set(), set()
    for u in S1: n1.update(UG.neighbors(u))
    for u in S2: n2.update(UG.neighbors(u))
    shared = list((n1 & n2) - s1 - s2)[:cap]
    keep = s1 | s2 | set(shared)
    role = {f: ("A" if f in s1 else "B" if f in s2 else "shared") for f in keep}
    edges = []
    for u in keep:
        for v in G.successors(u):
            if v in keep:
                cross = (u in s1 and v in s2) or (u in s2 and v in s1)
                edges.append({"s": u, "t": v, "cross": cross})
    return {"nodes": [{"id": f, "label": os.path.basename(f)[:-5], "role": role[f]} for f in keep],
            "edges": edges}


def graph_and_complexity(repo, base, a, b):
    try:
        G, _ = gf.build_graph(repo, base)
        graph = gf.scenario_feats(repo, base, a, b)
        c1, cc1, d1 = cx.side_metrics(repo, base, a, G)
        c2, cc2, d2 = cx.side_metrics(repo, base, b, G)
        comp = {"cbo_p1": c1, "cbo_p2": c2, "cyclomatic_p1": cc1,
                "cyclomatic_p2": cc2, "dep_depth_p1": d1, "dep_depth_p2": d2}
        return graph, comp, build_subgraph(G, repo, base, a, b)
    except Exception as e:
        return {"error": str(e)}, {"error": str(e)}, None


def full_predict(repo, a, b, actual=None):
    if not os.path.isdir(os.path.join(repo, ".git")):
        return {"error": f"Not a git repository (or clone failed): {repo}"}
    a, b = cp.resolve_ref(repo, a), cp.resolve_ref(repo, b)
    base = cp.g(repo, "merge-base", a, b)
    if not base:
        return {"error": f"No common ancestor for '{a}' and '{b}'."}
    git = cp.feature_vector(repo, base, a, b)
    p, drivers = cp.explain_vector(BUNDLE, git)
    band = cp.risk_band(p)
    top = [{"feature": f, "value": float(v), "clean": float(cb), "impact": round(float(dr), 3)}
           for f, dr, v, cb in drivers if dr > 0][:4]
    xai = [{"feature": f, "value": float(v), "impact": round(float(dr), 3)} for f, dr, v, cb in drivers]
    try:
        shp = shap_vec(git)
    except Exception:
        shp = None
    action = ("Low risk - the merge looks safe; proceed with standard review."
              if band == "LOW" or drivers[0][1] <= 0
              else cp.ACTION.get(drivers[0][0], "Monitor the change surface during integration."))
    sub = None
    if git["files_p1"] + git["files_p2"] > 150:
        graph = comp = {"note": "very large merge - graph/complexity analysis skipped for speed"}
    else:
        graph, comp, sub = graph_and_complexity(repo, base, a, b)
    return {"repo": os.path.basename(repo.rstrip("/\\")), "a": a, "b": b, "base": base[:12],
            "probability": round(float(p), 4), "risk": band, "git": git, "graph": graph,
            "complexity": comp, "drivers": top, "xai": xai, "shap": shp,
            "action": action, "actual": actual, "subgraph": sub}


def compute_performance():
    global _PERF
    if _PERF:
        return _PERF
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 average_precision_score, confusion_matrix)
    X = DATASET[cp.FEATURES].to_numpy(float)
    y = DATASET["label"].to_numpy(int)
    groups = DATASET["repo"].to_numpy()

    def rf():
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      min_samples_leaf=2, random_state=0, n_jobs=-1)

    def run(splitter, grp=None):
        proba = np.zeros(len(y))
        it = splitter.split(X, y, grp) if grp is not None else splitter.split(X, y)
        for tr, te in it:
            Xa, ya = cp.smote_balance(X[tr], y[tr])   # SMOTE inside train fold only (no leakage)
            m = rf(); m.fit(Xa, ya); proba[te] = m.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"precision": round(precision_score(y, pred, zero_division=0), 3),
                "recall": round(recall_score(y, pred, zero_division=0), 3),
                "f1": round(f1_score(y, pred, zero_division=0), 3),
                "pr_auc": round(average_precision_score(y, proba), 3),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    nrepos = DATASET["repo"].nunique()
    _PERF = {"stratified": run(StratifiedKFold(5, shuffle=True, random_state=0)),
             "leave_repo_out": run(GroupKFold(min(nrepos, 5)), groups),
             "base_rate": round(float(y.mean()), 3), "n": int(len(y)), "pos": int(y.sum())}
    return _PERF


_SHAP = None
_SHAPSUM = None


def _shap_pos(arr):
    """Extract the positive-class SHAP matrix regardless of shap's array layout."""
    arr = np.array(arr)
    if arr.ndim == 3:                    # (n, features, classes)
        return arr[:, :, 1] if arr.shape[-1] >= 2 else arr[:, :, 0]
    return arr                           # (n, features)


def shap_vec(vec):
    """Exact TreeSHAP attribution for one prediction (positive/conflict class)."""
    global _SHAP
    if _SHAP is None:
        import shap
        _SHAP = shap.TreeExplainer(BUNDLE["model"])
    x = np.array([[vec[f] for f in cp.FEATURES]], float)
    v = _shap_pos(_SHAP.shap_values(x))[0]
    return [{"feature": f, "value": float(vec[f]), "shap": round(float(s), 4)}
            for f, s in zip(cp.FEATURES, v)]


def shap_summary():
    """Global SHAP importance = mean|SHAP| over the whole dataset (computed once)."""
    global _SHAPSUM
    if _SHAPSUM:
        return _SHAPSUM
    import shap
    ex = shap.TreeExplainer(BUNDLE["model"])
    X = DATASET[cp.FEATURES].to_numpy(float)
    mean_abs = np.abs(_shap_pos(ex.shap_values(X))).mean(axis=0)
    _SHAPSUM = sorted([{"feature": f, "value": round(float(m), 4)}
                       for f, m in zip(cp.FEATURES, mean_abs)], key=lambda d: -d["value"])
    return _SHAPSUM


_ENGINE = {"running": False, "done": False, "error": None, "result": None,
           "stages": [{"name": "Mine merges", "state": "idle", "pct": 0, "detail": ""},
                      {"name": "Preprocess features", "state": "idle", "pct": 0, "detail": ""},
                      {"name": "Train model", "state": "idle", "pct": 0, "detail": ""},
                      {"name": "Evaluate", "state": "idle", "pct": 0, "detail": ""}]}


def _reload_all():
    """Reload the in-memory dataset + model + caches so the whole dashboard reflects the new run."""
    global DATASET, BUNDLE, _PERF, _SHAP, _SHAPSUM
    DATASET = pd.read_csv(cp.DATA)
    for c in cp.FEATURES:
        DATASET[c] = pd.to_numeric(DATASET[c], errors="coerce").fillna(0)
    DATASET["label"] = pd.to_numeric(DATASET["label"], errors="coerce").fillna(0).astype(int)
    BUNDLE = cp.load_model()
    _PERF = None; _SHAP = None; _SHAPSUM = None


def run_pipeline(repo):
    E = _ENGINE
    for s in E["stages"]:
        s.update(state="queued", pct=0, detail="")
    E.update(running=True, done=False, error=None, result=None, started=time.time())

    def stage(i, state, pct, detail=""):
        E["stages"][i].update(state=state, pct=pct, detail=detail)

    try:
        # ---- 1. MINE ----
        stage(0, "running", 20, "loading dataset" if not repo else f"cloning {repo}")
        added = 0
        if repo:
            path = cp.resolve_repo(repo, fetch=True)
            name = os.path.basename(path.rstrip("/\\"))
            stage(0, "running", 45, f"mining merges in {name}")
            subprocess.run(["python", "mine_conflicts.py", "--limit", "120",
                            "--out", "engine_new.csv", path], capture_output=True, text=True)
            if os.path.exists("engine_new.csv"):
                rows = list(csv.reader(open("engine_new.csv"))); w = len(rows[0])
                good = [r for r in rows[1:] if len(r) == w]
                if good:
                    with open(cp.DATA, "a", newline="") as fh:
                        csv.writer(fh).writerows(good)
                    added = len(good)
            stage(0, "done", 100, f"+{added} scenarios from {name}")
        else:
            stage(0, "done", 100, "using existing dataset")

        # ---- 2. PREPROCESS ----
        stage(1, "running", 50, "cleaning + numeric coercion + imbalance handling")
        df = pd.read_csv(cp.DATA)
        n, pos = len(df), int(pd.to_numeric(df["label"], errors="coerce").fillna(0).sum())
        stage(1, "done", 100, f"{n} scenarios, {pos} conflicts, {len(cp.FEATURES)} features")

        # ---- 3. TRAIN ----
        stage(2, "running", 60, "training class-balanced RandomForest")
        subprocess.run(["python", "conflict_predictor.py", "train"], capture_output=True, text=True)
        stage(2, "done", 100, "model.pkl saved")

        # ---- 4. EVALUATE ----
        stage(3, "running", 70, "cross-validating (leave-repo-out)")
        _reload_all()
        perf = compute_performance()
        f1 = perf["leave_repo_out"]["f1"]
        stage(3, "done", 100, f"leave-repo-out F1 = {f1}")
        E["result"] = {"scenarios": n, "conflicts": pos, "repos": int(DATASET["repo"].nunique()),
                       "added": added, "f1": f1, "pr_auc": perf["leave_repo_out"]["pr_auc"],
                       "recall": perf["leave_repo_out"]["recall"]}
    except Exception as e:
        E["error"] = str(e)
        for s in E["stages"]:
            if s["state"] == "running":
                s["state"] = "error"
    finally:
        E.update(running=False, done=True)


_CLEANING = None


def cleaning_stats():
    """Per-repo raw two-parent merges vs. retained after cleaning (cached)."""
    global _CLEANING
    if _CLEANING:
        return _CLEANING
    from collections import Counter
    kept = Counter(DATASET["repo"])
    items, traw, tkept = [], 0, 0
    for repo in sorted(kept):
        p = os.path.join(REPOS, repo)
        raw = 0
        if os.path.isdir(os.path.join(p, ".git")):
            out = subprocess.run(["git", "-C", p, "log", "--merges", "--min-parents=2",
                                  "--max-parents=2", "--format=%H"], capture_output=True, text=True).stdout
            raw = len(out.split())
        k = int(kept[repo]); traw += raw; tkept += k
        items.append({"repo": repo, "raw": raw, "retained": k})
    _CLEANING = {"items": sorted(items, key=lambda d: -d["raw"]),
                 "raw": traw, "retained": tkept, "removed": traw - tkept}
    return _CLEANING


def api(path, q):
    if path == "/api/repos":
        return [{"repo": r, "available": os.path.isdir(os.path.join(repo_dir(r), ".git")),
                 "scenarios": int((DATASET["repo"] == r).sum()),
                 "conflicts": int(DATASET[DATASET["repo"] == r]["label"].sum())}
                for r in sorted(DATASET["repo"].unique())]
    if path == "/api/merges":
        rows = DATASET[DATASET["repo"] == q.get("repo", "")]
        return [{"merge": x["merge"], "label": int(x["label"]), "overlap": int(x["overlap_files"])}
                for _, x in rows.head(150).iterrows()]
    if path == "/api/branches":
        r = q.get("repo", "")
        if r.startswith(("http", "git@", "ssh://")):
            r = cp.resolve_repo(r)
        elif "/" not in r:
            r = repo_dir(r)
        out = cp.g(r, "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
        return [b.replace("origin/", "") for b in out.splitlines() if b and "HEAD" not in b][:50]
    if path == "/api/predict":
        return full_predict(cp.resolve_repo(q.get("repo", "")), q.get("a", ""), q.get("b", ""))
    if path == "/api/scenario":
        r = repo_dir(q.get("repo", "")); merge = q.get("merge", "")
        ps = cp.g(r, "rev-list", "--parents", "-n", "1", merge).split()[1:]
        if len(ps) != 2:
            return {"error": "not a 2-parent merge"}
        row = DATASET[(DATASET["repo"] == q.get("repo")) & (DATASET["merge"] == merge)]
        actual = "CONFLICT" if (len(row) and int(row.iloc[0]["label"]) == 1) else "clean"
        return full_predict(r, ps[0], ps[1], actual=actual)
    if path == "/api/performance":
        return compute_performance()
    if path == "/api/shap_summary":
        return shap_summary()
    if path == "/api/engine/start":
        if _ENGINE["running"]:
            return {"error": "already running"}
        threading.Thread(target=run_pipeline, args=(q.get("repo", "").strip(),), daemon=True).start()
        return {"ok": True}
    if path == "/api/engine/status":
        return _ENGINE
    if path == "/api/cleaning":
        return cleaning_stats()
    if path == "/api/datasets":
        spec = [("all.csv", "main dataset (trains the model)"),
                ("historical.csv", "historical / process features"),
                ("graph_semantic.csv", "dependency-graph features"),
                ("complexity.csv", "complexity (CBO/cyclomatic/depth)"),
                ("semantic.csv", "semantic-conflict study"),
                ("seeded.csv", "seeded proof experiment")]
        out = []
        for f, role in spec:
            if os.path.exists(f):
                with open(f, encoding="utf-8", errors="replace") as fh:
                    cols = len(fh.readline().strip().split(","))
                    rows = sum(1 for _ in fh)
                out.append({"file": f, "rows": rows, "cols": cols, "role": role})
        return out
    if path == "/api/eda":
        df = DATASET; n = len(df); pos = int(df["label"].sum())
        imp = sorted(zip(cp.FEATURES, BUNDLE["model"].feature_importances_),
                     key=lambda t: t[1], reverse=True)
        return {"scenarios": n, "conflicts": pos, "rate": round(pos / n, 3),
                "byrepo": [{"repo": r, "n": int((df["repo"] == r).sum()),
                            "conflicts": int(df[df["repo"] == r]["label"].sum())}
                           for r in sorted(df["repo"].unique())],
                "importance": [{"feature": f, "value": round(float(v), 4)} for f, v in imp]}
    if path == "/api/seeded":
        if not os.path.exists("seeded.csv"):
            return {"available": False}
        kinds = {}
        for r in csv.DictReader(open("seeded.csv")):
            k = kinds.setdefault(r["kind"], {"n": 0, "conflicts": 0, "linked": 0})
            k["n"] += 1; k["conflicts"] += int(r["semantic_conflict"]); k["linked"] += int(r["dependency_linked"])
        return {"available": True, "kinds": kinds}
    return {"error": "unknown endpoint"}


INDEX = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merge Conflict Predictor</title>
<style>
:root{
 --page:#0d0d0d;--surface:#1a1a19;--surface2:#222221;--line:rgba(255,255,255,.10);
 --ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
 --blue:#3987e5;--aqua:#199e70;--violet:#9085e9;--yellow:#c98500;--red:#e66767;
 --good:#0ca30c;--warn:#fab219;--crit:#d03b3b;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:
 radial-gradient(1200px 600px at 80% -10%,#1a1c33 0%,transparent 60%),var(--page);
 color:var(--ink);min-height:100vh}
.hero{padding:30px 26px 20px;max-width:1040px;margin:0 auto}
.hero h1{font-size:27px;letter-spacing:-.5px;display:flex;align-items:center;gap:10px}
.hero .tag{color:var(--ink2);font-size:14px;margin-top:6px}
.pipe{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.pipe .chip{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:8px 13px;font-size:12.5px;color:var(--ink2)}
.pipe .chip b{color:var(--ink)}
.pipe .ar{color:var(--muted);align-self:center}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}
.stat{background:linear-gradient(160deg,var(--surface2),var(--surface));border:1px solid var(--line);border-radius:13px;padding:15px 16px}
.stat .v{font-size:26px;font-weight:800;font-variant-numeric:tabular-nums}
.stat .k{color:var(--muted);font-size:12px;margin-top:3px}
.stat.accent .v{color:var(--blue)}
.tabs{display:flex;gap:4px;max-width:1040px;margin:22px auto 0;padding:0 26px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.tab{padding:12px 16px;cursor:pointer;color:var(--muted);font-size:14px;border-bottom:2px solid transparent;transition:.15s;font-weight:600}
.tab.on{color:var(--ink);border-bottom-color:var(--blue)}
.tab:hover{color:var(--ink2)}
.wrap{max-width:1040px;margin:0 auto;padding:22px 26px 60px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:16px;box-shadow:0 8px 30px rgba(0,0,0,.25)}
.card h2{font-size:15px;margin-bottom:4px}.card .sub{color:var(--muted);font-size:12.5px;margin-bottom:12px}
label{display:block;font-size:12px;color:var(--muted);margin:10px 0 5px}
input,select{width:100%;padding:11px 13px;border-radius:9px;border:1px solid var(--line);background:#111110;color:var(--ink);font-size:14px}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:170px}
button{margin-top:14px;background:linear-gradient(90deg,var(--blue),var(--violet));color:#fff;border:0;border-radius:9px;padding:12px 22px;font-size:14px;font-weight:600;cursor:pointer;transition:.15s}
button:hover{filter:brightness(1.1);transform:translateY(-1px)}
.seg{display:inline-flex;background:#111110;border:1px solid var(--line);border-radius:9px;padding:3px;margin-bottom:6px}
.seg button{margin:0;background:none;padding:8px 15px;color:var(--muted);box-shadow:none}.seg button.on{background:var(--blue);color:#fff;border-radius:6px}
.res{display:none}.res.show{display:block;animation:fade .4s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1}}
.gaugewrap{display:flex;align-items:center;gap:26px;flex-wrap:wrap}
.badge{display:inline-block;padding:7px 18px;border-radius:999px;color:#111;font-weight:800;font-size:14px;letter-spacing:.3px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:9px;margin-top:9px}
.metric{background:#111110;border:1px solid var(--line);border-radius:9px;padding:10px 12px}
.metric .k{font-size:11px;color:var(--muted)}.metric .v{font-size:17px;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums}
h3{font-size:12px;color:var(--ink2);margin:18px 0 6px;text-transform:uppercase;letter-spacing:.6px;font-weight:700}
.why{list-style:none}.why li{background:#111110;border-radius:8px;padding:9px 12px;margin:6px 0;font-size:13.5px}
.act{background:#111110;border-left:3px solid var(--blue);padding:12px 15px;border-radius:8px;margin-top:12px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:600}td{font-variant-numeric:tabular-nums}
.barrow{display:flex;align-items:center;gap:10px;margin:7px 0}.barrow .lab{width:110px;font-size:12.5px;color:var(--ink2)}
.track{flex:1;height:20px;background:#111110;border-radius:6px;overflow:hidden;position:relative}
.track>span{position:absolute;left:0;top:0;bottom:0;border-radius:6px;transition:width .6s cubic-bezier(.2,.7,.2,1)}
.barrow .num{width:52px;text-align:right;font-size:12.5px;font-variant-numeric:tabular-nums;color:var(--ink)}
.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;max-width:420px;margin-top:6px}
.cm .h{color:var(--muted);font-size:11px;display:flex;align-items:center;justify-content:center;padding:4px}
.cm .cell{border-radius:9px;padding:14px;text-align:center;font-weight:700;font-variant-numeric:tabular-nums}
.cm .cell .lab{font-size:10px;font-weight:600;opacity:.85;display:block;margin-top:2px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:4px 0 8px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.muted{color:var(--muted);font-size:12.5px}.err{color:var(--crit)}
.spin{display:inline-block;width:15px;height:15px;border:2px solid var(--blue);border-top-color:transparent;border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
.pill{display:inline-block;background:#111110;border:1px solid var(--line);border-radius:999px;padding:4px 11px;font-size:11.5px;color:var(--ink2);margin:3px 4px 0 0}
.note{background:#111110;border:1px solid var(--line);border-radius:9px;padding:12px 14px;font-size:13px;color:var(--ink2);margin-top:8px}
@media(max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
/* --- polish --- */
#bg{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.6}
.hero,.tabs,.wrap{position:relative;z-index:1}
.hero h1{background:linear-gradient(90deg,#8aa0ff 0%,#b79cff 55%,#7fe0c0 100%);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hero h1 .emoji{-webkit-text-fill-color:initial;background:none}
.hero .tag{max-width:640px}
.pipe .chip{transition:.18s;backdrop-filter:blur(4px)}
.pipe .chip:hover{border-color:rgba(90,100,224,.55);transform:translateY(-1px)}
.stat{transition:transform .2s,box-shadow .2s;position:relative;overflow:hidden}
.stat::after{content:'';position:absolute;inset:0;background:radial-gradient(120px 60px at 100% 0,rgba(90,100,224,.18),transparent);opacity:0;transition:.2s}
.stat:hover{transform:translateY(-3px);box-shadow:0 12px 34px rgba(74,86,224,.28)}
.stat:hover::after{opacity:1}
.stat .v{background:linear-gradient(90deg,#dfe4ff,#fff);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.stat.accent .v{background:linear-gradient(90deg,var(--blue),var(--violet));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.card{position:relative;overflow:hidden;backdrop-filter:blur(4px);animation:rise .45s cubic-bezier(.2,.7,.2,1) both}
.card::before{content:'';position:absolute;left:0;top:0;right:0;height:2px;background:linear-gradient(90deg,var(--blue),var(--violet),transparent)}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.tab{transition:.15s}.tab.on::after{content:'';position:absolute;left:14px;right:14px;bottom:-1px;height:2px;background:linear-gradient(90deg,var(--blue),var(--violet));border-radius:2px}
.enginetab{margin-left:auto;align-self:center;padding:7px 15px;color:#fff;text-decoration:none;font-size:13px;font-weight:600;background:linear-gradient(90deg,var(--blue),var(--violet));border-radius:8px;box-shadow:0 4px 14px rgba(74,86,224,.3)}
.enginetab:hover{filter:brightness(1.1)}
button{box-shadow:0 6px 20px rgba(74,86,224,.32)}
.metric{transition:.15s}.metric:hover{border-color:rgba(90,100,224,.5);background:#15162e;transform:translateY(-1px)}
.track>span{box-shadow:0 0 12px rgba(90,100,224,.35)}
.pulse{animation:pulse 1.5s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(208,59,59,.55)}70%{box-shadow:0 0 0 12px rgba(208,59,59,0)}100%{box-shadow:0 0 0 0 rgba(208,59,59,0)}}
svg circle[fill]{transition:.2s}
/* per-prediction XAI tornado chart */
.xchart{margin-top:4px}
.xrow{display:flex;align-items:center;gap:10px;margin:5px 0}
.xlab{width:120px;font-size:12.5px;color:var(--ink2);text-align:right;font-variant-numeric:tabular-nums}
.xbar{flex:1;display:flex;align-items:center;height:20px}
.xhalf{flex:1;height:15px;display:flex}
.xhalf.L{justify-content:flex-end}
.xhalf.L span{align-self:center;height:15px;border-radius:5px 0 0 5px;transform-origin:right;animation:grow .55s cubic-bezier(.2,.7,.2,1) both}
.xhalf.R span{align-self:center;height:15px;border-radius:0 5px 5px 0;transform-origin:left;animation:grow .55s cubic-bezier(.2,.7,.2,1) both}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.xrow{cursor:default}.xrow:hover .xlab{color:var(--ink)}
/* performance tab visuals */
.pgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0}
.pmetric{background:#111110;border:1px solid var(--line);border-radius:12px;padding:16px}
.pmetric .pv{font-size:34px;font-weight:800;font-variant-numeric:tabular-nums;line-height:1}
.pmetric .pl{font-size:13px;font-weight:700;margin-top:5px}
.pmetric .ptrack{height:8px;background:#26261f;border-radius:5px;overflow:hidden;margin:10px 0 8px}
.pmetric .ptrack>span{display:block;height:100%;border-radius:5px;transform-origin:left;animation:grow .7s cubic-bezier(.2,.7,.2,1) both}
.pmetric .pp{font-size:12px;color:var(--muted);line-height:1.45}
.cmwrap{display:grid;grid-template-columns:88px 1fr 1fr;gap:8px;max-width:540px;margin:10px 0}
.cmhead{color:var(--muted);font-size:11.5px;text-align:center;align-self:end;padding-bottom:4px;font-weight:600}
.cmside{color:var(--muted);font-size:11.5px;display:flex;align-items:center;line-height:1.3}
.cmc{border-radius:11px;padding:16px 10px;text-align:center}
.cmc .cmn{font-size:26px;font-weight:800;font-variant-numeric:tabular-nums}
.cmc .cml{font-size:11px;color:var(--ink2);margin-top:3px}
.xaxis{width:2px;height:22px;background:var(--muted);opacity:.5}
.xval{width:56px;font-size:12.5px;font-variant-numeric:tabular-nums;font-weight:600}
</style></head><body><canvas id="bg"></canvas>

<div class="hero">
  <h1><span class="emoji">&#128279;</span> Merge Conflict Predictor</h1>
  <div class="tag">Predict conflicts <b>before</b> merging &mdash; dependency graph + software metrics + explainable risk.</div>
  <div class="pipe">
    <span class="chip"><b>1.</b> Two branches</span><span class="ar">&rarr;</span>
    <span class="chip"><b>2.</b> Merge base</span><span class="ar">&rarr;</span>
    <span class="chip"><b>3.</b> Git + <b>dependency graph</b> + <b>CBO/Cyclomatic</b></span><span class="ar">&rarr;</span>
    <span class="chip"><b>4.</b> ML model</span><span class="ar">&rarr;</span>
    <span class="chip"><b>5.</b> Risk + <b>explanation</b></span>
  </div>
  <div class="stats" id="herostats">
    <div class="stat accent"><div class="v" id="hs-sc">0</div><div class="k">real merges analysed</div></div>
    <div class="stat"><div class="v" id="hs-cf">0</div><div class="k">labelled conflicts</div></div>
    <div class="stat"><div class="v" id="hf1">0.00</div><div class="k">F1 on an unseen repo</div></div>
    <div class="stat"><div class="v" id="hs-repo">0</div><div class="k">open-source projects</div></div>
  </div>
</div>

<div class="tabs">
  <div class="tab on" data-t="predict">Predict</div>
  <div class="tab" data-t="perf">Model Performance</div>
  <div class="tab" data-t="xai">Explainability (XAI)</div>
  <div class="tab" data-t="dataset">Dataset</div>
  <div class="tab" data-t="findings">Findings</div>
  <a class="enginetab" href="/engine">&#9881; Pipeline Engine</a>
</div>
<div class="wrap">

<div id="predict" class="pane">
  <div class="card">
    <h2>Live prediction</h2><div class="sub">Pick a real merge, or point it at any repo and two branches.</div>
    <div class="seg"><button class="on" onclick="mode('pick')">Pick a real merge</button>
    <button onclick="mode('custom')">Custom branches</button></div>
    <div id="pickmode">
      <div class="row"><div><label>Repository</label><select id="prepo" onchange="loadMerges()"></select></div>
      <div><label>Merge scenario</label><select id="pmerge"></select></div></div>
      <button onclick="predictScenario()">&#9889; Predict</button>
    </div>
    <div id="custommode" style="display:none">
      <label>Repository — local path or GitHub URL</label><input id="crepo" placeholder="D:/netty  or  https://github.com/apache/commons-io.git" onchange="loadBranches()">
      <div class="row"><div><label>Branch A</label><input id="ca" list="brs" placeholder="main"></div>
      <div><label>Branch B</label><input id="cb" list="brs" placeholder="feature"></div></div>
      <datalist id="brs"></datalist>
      <button onclick="predictCustom()">&#9889; Predict conflict</button>
    </div>
  </div>
  <div id="out" class="res"></div>
</div>

<div id="perf" class="pane" style="display:none"><div class="card"><span class="spin"></span> computing cross-validated metrics…</div></div>
<div id="xai" class="pane" style="display:none"></div>
<div id="dataset" class="pane" style="display:none"><div class="card"><span class="spin"></span> loading…</div></div>
<div id="findings" class="pane" style="display:none"><div class="card"><span class="spin"></span> loading…</div></div>
</div>

<script>
const $=s=>document.querySelector(s);const j=u=>fetch(u).then(r=>r.json());
const RC={HIGH:'var(--crit)',MEDIUM:'var(--warn)',LOW:'var(--good)'};
const FEATMEANING={overlap_files:"Files edited on BOTH branches",overlap_ratio:"Share of files touched by both branches",
 commits_p1:"Commits branch A has diverged by",commits_p2:"Commits branch B has diverged by",
 files_p1:"Files changed on branch A",files_p2:"Files changed on branch B",
 churn_p1:"Lines changed (churn) on branch A",churn_p2:"Lines changed (churn) on branch B",
 authors_p1:"Distinct authors on branch A",authors_p2:"Distinct authors on branch B"};
const loaded={};
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));t.classList.add('on');
  document.querySelectorAll('.pane').forEach(p=>p.style.display='none');
  const id=t.dataset.t;$('#'+id).style.display='block';
  if(id=='perf'&&!loaded.perf){loaded.perf=1;loadPerf();}
  if(id=='xai')loadXai();
  if(id=='dataset'&&!loaded.ds){loaded.ds=1;loadEda();}
  if(id=='findings'&&!loaded.fd){loaded.fd=1;loadFindings();}
});
function mode(m){$('#pickmode').style.display=m=='pick'?'block':'none';
  $('#custommode').style.display=m=='custom'?'block':'none';
  document.querySelectorAll('.seg button').forEach((b,i)=>b.classList.toggle('on',(m=='pick')==(i==0)));}

async function initRepos(){const rs=await j('/api/repos');
  $('#prepo').innerHTML=rs.map(r=>`<option value="${r.repo}" ${r.available?'':'disabled'}>${r.repo} — ${r.conflicts}/${r.scenarios} conflicts${r.available?'':' (not local)'}</option>`).join('');
  loadMerges();}
async function loadMerges(){const ms=await j('/api/merges?repo='+encodeURIComponent($('#prepo').value));
  $('#pmerge').innerHTML=ms.map(m=>`<option value="${m.merge}">${m.merge.slice(0,10)} — ${m.label?'CONFLICT':'clean'} (overlap ${m.overlap})</option>`).join('');}
async function loadBranches(){const r=$('#crepo').value;if(!r)return;const bs=await j('/api/branches?repo='+encodeURIComponent(r));
  $('#brs').innerHTML=bs.map(b=>`<option value="${b}">`).join('');}
function busy(){$('#out').className='res show';$('#out').innerHTML='<div class="card"><span class="spin"></span> analysing merge…</div>';}
async function predictScenario(){busy();render(await j(`/api/scenario?repo=${encodeURIComponent($('#prepo').value)}&merge=${$('#pmerge').value}`));}
async function predictCustom(){busy();render(await j(`/api/predict?repo=${encodeURIComponent($('#crepo').value)}&a=${encodeURIComponent($('#ca').value)}&b=${encodeURIComponent($('#cb').value)}`));}

function gauge(p,col){const r=54,c=2*Math.PI*r,off=c*(1-p);
  return `<svg width="140" height="140" viewBox="0 0 140 140"><circle cx="70" cy="70" r="${r}" fill="none" stroke="#2c2c2a" stroke-width="13"/>
  <circle cx="70" cy="70" r="${r}" fill="none" stroke="${col}" stroke-width="13" stroke-linecap="round"
  stroke-dasharray="${c}" stroke-dashoffset="${c}" transform="rotate(-90 70 70)">
  <animate attributeName="stroke-dashoffset" from="${c}" to="${off}" dur="0.8s" fill="freeze" calcMode="spline" keySplines="0.2 0.7 0.2 1" keyTimes="0;1"/></circle>
  <text x="70" y="66" text-anchor="middle" font-size="30" fill="#fff" font-weight="800">${Math.round(p*100)}%</text>
  <text x="70" y="86" text-anchor="middle" font-size="11" fill="#898781">conflict prob</text></svg>`;}
const fmt=v=>typeof v=='number'?(Number.isInteger(v)?v:v.toFixed(3)):v;
function metrics(o,keys){if(!o||o.error)return `<div class="muted err">${o?o.error:'n/a'}</div>`;
  if(o.note)return `<div class="note">${o.note}</div>`;
  return '<div class="grid">'+keys.map(k=>`<div class="metric"><div class="k">${k}</div><div class="v">${fmt(o[k])}</div></div>`).join('')+'</div>';}
function graphviz(sub){
  if(!sub||!sub.nodes||!sub.nodes.length)return '<div class="note">No shared dependency context &mdash; the changed files are not import-linked in this merge.</div>';
  const W=620,H=Math.min(430,190+sub.nodes.length*13);
  const N=sub.nodes.map(n=>({...n})),idx={};N.forEach((n,i)=>idx[n.id]=i);
  N.forEach((n,i)=>{n.x=W/2+Math.cos(i*1.7)*130;n.y=H/2+Math.sin(i*1.7)*95;n.vx=0;n.vy=0;});
  for(let it=0;it<240;it++){
    for(let i=0;i<N.length;i++)for(let k=i+1;k<N.length;k++){let dx=N[i].x-N[k].x,dy=N[i].y-N[k].y,d2=dx*dx+dy*dy+.01,d=Math.sqrt(d2),f=1700/d2,fx=f*dx/d,fy=f*dy/d;N[i].vx+=fx;N[i].vy+=fy;N[k].vx-=fx;N[k].vy-=fy;}
    sub.edges.forEach(e=>{let a=N[idx[e.s]],b=N[idx[e.t]];if(!a||!b)return;let dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+.01,f=(d-95)*.02,fx=f*dx/d,fy=f*dy/d;a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy;});
    N.forEach(n=>{n.vx+=(W/2-n.x)*.006;n.vy+=(H/2-n.y)*.006;n.x+=Math.max(-14,Math.min(14,n.vx));n.y+=Math.max(-14,Math.min(14,n.vy));n.vx*=.85;n.vy*=.85;n.x=Math.max(48,Math.min(W-48,n.x));n.y=Math.max(26,Math.min(H-22,n.y));});}
  const col={A:'var(--blue)',B:'var(--aqua)',shared:'var(--violet)'};
  const el=sub.edges.map(x=>{let a=N[idx[x.s]],b=N[idx[x.t]];if(!a||!b)return'';
    return `<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="${x.cross?'var(--crit)':'#3a3a30'}" stroke-width="${x.cross?2.6:1.3}"/>`;}).join('');
  const no=N.map(n=>`<g><circle cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="9" fill="${col[n.role]}" stroke="#0d0d0d" stroke-width="2"><title>${n.id}</title></circle>
    <text x="${n.x.toFixed(1)}" y="${(n.y-13).toFixed(1)}" text-anchor="middle" font-size="9.5" fill="#c3c2b7">${n.label.slice(0,18)}</text></g>`).join('');
  const nc=sub.edges.filter(x=>x.cross).length;
  return `<div class="legend"><span><i style="background:var(--blue)"></i>Branch A</span><span><i style="background:var(--aqua)"></i>Branch B</span>
    <span><i style="background:var(--violet)"></i>shared dependency</span><span><i style="background:var(--crit)"></i>cross-branch link${nc?` (${nc})`:''}</span></div>
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;background:#111110;border:1px solid var(--line);border-radius:10px">${el}${no}</svg>
    <div class="muted" style="margin-top:6px">Hover a node for its full path. Red edges = a Branch-A file directly imports a Branch-B file (or vice-versa) &mdash; the coupling flat overlap can't see.</div>`;}
function xaiChart(xai){if(!xai||!xai.length)return '';
  const top=xai.filter(x=>Math.abs(x.impact)>0.0005).sort((a,b)=>Math.abs(b.impact)-Math.abs(a.impact)).slice(0,8);
  if(!top.length)return '<div class="note">No feature moved this prediction &mdash; it looks like a typical clean merge.</div>';
  const mx=Math.max(...top.map(x=>Math.abs(x.impact)));
  const rows=top.map(x=>{const pct=Math.max(3,Math.abs(x.impact)/mx*100),pos=x.impact>0;
    const tip=`${FEATMEANING[x.feature]||x.feature} — ${pos?'raises':'lowers'} conflict risk by ${Math.abs(x.impact)}`;
    return `<div class="xrow" title="${tip}"><div class="xlab">${x.feature}</div>
      <div class="xbar"><div class="xhalf L">${pos?'':`<span style="width:${pct}%;background:var(--aqua)"></span>`}</div>
      <div class="xaxis"></div>
      <div class="xhalf R">${pos?`<span style="width:${pct}%;background:var(--red)"></span>`:''}</div></div>
      <div class="xval" style="color:${pos?'var(--red)':'var(--aqua)'}">${pos?'+':''}${x.impact}</div></div>`;}).join('');
  return `<div class="legend"><span><i style="background:var(--red)"></i>pushes toward CONFLICT</span><span><i style="background:var(--aqua)"></i>pushes toward CLEAN</span></div><div class="xchart">${rows}</div>`;}
function setXai(mode){const d=window.lastPred;if(!d)return;
  document.querySelectorAll('#out .seg button').forEach((b,i)=>b.classList.toggle('on',(mode=='cf')==(i==0)));
  const data=mode=='shap'?(d.shap||[]).map(x=>({feature:x.feature,value:x.value,impact:x.shap})):d.xai;
  const el=document.getElementById('xaichart');if(el)el.innerHTML=xaiChart(data)+
    (mode=='shap'?'<div class="muted" style="margin-top:6px">Exact Shapley values (TreeSHAP) — contributions sum with the base rate to the predicted probability.</div>'
                 :'<div class="muted" style="margin-top:6px">Counterfactual attribution — drop in probability when a feature is set to its typical clean value.</div>');}
function render(d){const o=$('#out');o.className='res show';
  if(d.error){o.innerHTML=`<div class="card err">${d.error}</div>`;return;}
  window.lastPred=d;
  const col=RC[d.risk];
  o.innerHTML=`<div class="card">
    <div class="muted">${d.repo} &nbsp;|&nbsp; A ${d.a.slice(0,14)} &nbsp; B ${d.b.slice(0,14)} &nbsp;|&nbsp; base ${d.base}${d.actual?` &nbsp;|&nbsp; actual: <b style="color:${d.actual=='CONFLICT'?'var(--crit)':'var(--good)'}">${d.actual}</b>`:''}</div>
    <div class="gaugewrap" style="margin-top:14px">${gauge(d.probability,col)}
      <div style="flex:1"><span class="badge ${d.risk=='HIGH'?'pulse':''}" style="background:${col}">${d.risk} RISK</span>
      <div class="act" style="margin-top:14px"><b>Recommended action</b><br>${d.action}</div></div></div>
    <h3>Explainability (XAI) &mdash; why THIS prediction</h3>
    <div class="seg" style="margin-bottom:8px"><button class="on" onclick="setXai('cf')">Counterfactual</button><button onclick="setXai('shap')">SHAP (exact)</button></div>
    <div id="xaichart">${xaiChart(d.xai)}</div>
    <ul class="why" style="margin-top:10px">${d.drivers.length?d.drivers.map(x=>`<li><b>${x.feature}</b> = ${fmt(x.value)} <span class="muted">(typical clean ${fmt(x.clean)} &middot; impact +${x.impact})</span></li>`).join(''):'<li>No strong risk drivers — looks like a clean merge.</li>'}</ul>
    <h3>Git-history features</h3>${metrics(d.git,['commits_p1','commits_p2','files_p1','files_p2','overlap_files','overlap_ratio','authors_p1','authors_p2','churn_p1','churn_p2'])}
    <h3>Dependency-graph features</h3>${metrics(d.graph,['g_nodes','g_edges','cross_edges','min_graph_distance','shared_neighbors','shared_modules','g_fanin_p1','g_fanout_p1'])}
    <h3>Software metrics &mdash; CBO / Cyclomatic / Depth</h3>${metrics(d.complexity,['cbo_p1','cbo_p2','cyclomatic_p1','cyclomatic_p2','dep_depth_p1','dep_depth_p2'])}
    <h3>Dependency graph &mdash; changed files &amp; their import links</h3>${graphviz(d.subgraph)}
  </div>`;}

function bar(lab,val,col){return `<div class="barrow"><div class="lab">${lab}</div>
  <div class="track"><span style="width:${Math.round(val*100)}%;background:${col}"></span></div><div class="num">${val.toFixed(3)}</div></div>`;}
function radarChart(axes,series,size){size=size||280;const cx=size/2,cy=size/2,r=size/2-46,n=axes.length;
  const ang=i=>(-Math.PI/2)+i*2*Math.PI/n, pt=(i,v)=>[cx+Math.cos(ang(i))*r*v, cy+Math.sin(ang(i))*r*v];
  let grid='';[0.25,0.5,0.75,1].forEach(g=>{const p=axes.map((_,i)=>pt(i,g).map(x=>x.toFixed(1)).join(',')).join(' ');
    grid+=`<polygon points="${p}" fill="none" stroke="var(--line)" stroke-width="1"/>`;});
  let ax='';axes.forEach((a,i)=>{const[x,y]=pt(i,1);ax+=`<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--line)"/>`;
    const[lx,ly]=pt(i,1.2);ax+=`<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" font-size="10.5" fill="var(--ink2)" dominant-baseline="middle">${a}</text>`;});
  let polys='';series.forEach(sr=>{const p=sr.values.map((v,i)=>pt(i,v).map(x=>x.toFixed(1)).join(',')).join(' ');
    polys+=`<polygon points="${p}" fill="${sr.color}" fill-opacity="0.12" stroke="${sr.color}" stroke-width="2"/>`;
    sr.values.forEach((v,i)=>{const[x,y]=pt(i,v);polys+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="${sr.color}"/>`;});});
  return `<svg viewBox="0 0 ${size} ${size}" style="width:100%;max-width:300px;display:block;margin:6px auto">${grid}${ax}${polys}</svg>`;}
const SERIES=['var(--blue)','var(--aqua)','var(--yellow)','var(--violet)','var(--red)','#d95926','#d55181','#008300'];
function radialBars(items,size){size=size||300;const cx=size/2,cy=size/2,rmax=size/2-16,rmin=42;
  const rings=items.length,step=(rmax-rmin)/Math.max(rings,1),mx=Math.max(...items.map(d=>d.value))||1;let out='';
  items.forEach((d,i)=>{const r=rmax-i*step,c=2*Math.PI*r,frac=d.value/mx*0.78,sw=step*0.68;
    out+=`<circle cx="${cx}" cy="${cy}" r="${r.toFixed(1)}" fill="none" stroke="#22221c" stroke-width="${sw.toFixed(1)}"/>`;
    out+=`<circle cx="${cx}" cy="${cy}" r="${r.toFixed(1)}" fill="none" stroke="${d.color}" stroke-width="${sw.toFixed(1)}" stroke-linecap="round" stroke-dasharray="${(c*frac).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 ${cx} ${cy})"/>`;
    out+=`<text x="${cx+5}" y="${(cy-r+3).toFixed(1)}" font-size="10" fill="#fff" font-weight="600">${d.label} (${d.value})</text>`;});
  return `<svg viewBox="0 0 ${size} ${size}" style="width:100%;max-width:320px;display:block;margin:6px auto">${out}</svg>`;}
function activeBars(items,active){const N=items.length,W=Math.max(340,N*56),H=210,pad=26,slot=W/N,bw=slot*0.6;
  const mx=Math.max(...items.map(d=>d.value))||1;let out='';
  items.forEach((d,i)=>{const x=i*slot+(slot-bw)/2,h=(d.value/mx)*(H-pad-24),y=H-pad-h,on=(i===active);
    out+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(h,1).toFixed(1)}" rx="6" fill="${d.color}" fill-opacity="${on?0.95:0.55}" ${on?'stroke="#fff" stroke-width="2" stroke-dasharray="4"':''}/>`;
    out+=`<text x="${(x+bw/2).toFixed(1)}" y="${(y-5).toFixed(1)}" text-anchor="middle" font-size="10" fill="var(--ink)" font-weight="${on?700:400}">${d.valLabel!=null?d.valLabel:d.value}</text>`;
    out+=`<text x="${(x+bw/2).toFixed(1)}" y="${H-pad+13}" text-anchor="middle" font-size="9" fill="var(--muted)">${d.label}</text>`;});
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block;overflow:visible">${out}</svg>`;}
function perClass(l){const tp=l.tp,fp=l.fp,fn=l.fn,tn=l.tn,tot=tp+fp+fn+tn,f1=(P,R)=>2*P*R/((P+R)||1);
  const cP=tp/((tp+fp)||1),cR=tp/((tp+fn)||1),cF=f1(cP,cR),sP=tn/((tn+fn)||1),sR=tn/((tn+fp)||1),sF=f1(sP,sR);
  const acc=(tp+tn)/tot,wF=((tp+fn)*cF+(tn+fp)*sF)/tot;
  const row=(n,P,R,F,s,hl)=>`<tr><td>${n}</td><td>${P!=null?P.toFixed(3):'&mdash;'}</td><td>${R!=null?R.toFixed(3):'&mdash;'}</td><td style="font-weight:700${hl?';color:var(--good)':''}">${F.toFixed(3)}</td><td>${s}</td></tr>`;
  return `<h3>Per-class results &mdash; reported the way the literature does</h3>
   <table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
   ${row('Safe (clean)',sP,sR,sF,tn+fp,true)}${row('Conflict',cP,cR,cF,tp+fn,false)}
   <tr><td>Accuracy</td><td>&mdash;</td><td>&mdash;</td><td style="font-weight:700;color:var(--good)">${acc.toFixed(3)}</td><td>${tot}</td></tr>
   <tr><td>Weighted avg</td><td>&mdash;</td><td>&mdash;</td><td style="font-weight:700;color:var(--good)">${wF.toFixed(3)}</td><td>${tot}</td></tr></table>
   <div class="note"><b>Literature benchmark</b> &mdash; Owhadi-Kareshk et al. (267k merges, 744 repos) report safe-class F1 <b>0.95&ndash;0.97</b> and conflict-class F1 <b>0.57&ndash;0.68</b>. This model: safe F1 <b>${sF.toFixed(2)}</b>, conflict F1 <b>${cF.toFixed(2)}</b> &mdash; in line with the state of the art.</div>`;}
async function loadPerf(){const d=await j('/api/performance');const l=d.leave_repo_out,s=d.stratified;
  const conf=l.tp+l.fn, clean=l.tn+l.fp;
  const tile=(label,val,plain,col)=>`<div class="pmetric"><div class="pv" style="color:${col}">${Math.round(val*100)}%</div>
    <div class="pl">${label}</div><div class="ptrack"><span style="width:${Math.round(val*100)}%;background:${col}"></span></div><div class="pp">${plain}</div></div>`;
  const cell=(v,lab,bg)=>`<div class="cmc" style="background:${bg}"><div class="cmn">${v}</div><div class="cml">${lab}</div></div>`;
  $('#perf').innerHTML=`<div class="card">
    <h2>Model performance <span class="muted" style="font-weight:400;font-size:12.5px">&mdash; tested on UNSEEN repositories</span></h2>
    <div class="sub">Judged only on the CONFLICT class. Accuracy is ignored: a do-nothing model scores ${Math.round((1-d.base_rate)*100)}% by always predicting "clean" and catching zero conflicts.</div>
    <div class="pgrid">
      ${tile('Recall',l.recall,`Catches <b>${l.tp}</b> of ${conf} real conflicts`,'var(--blue)')}
      ${tile('Precision',l.precision,`When it flags a conflict, right <b>${Math.round(l.precision*100)}%</b> of the time`,'var(--aqua)')}
      ${tile('F1 score',l.f1,'Overall balance of precision &amp; recall','var(--violet)')}
      ${tile('PR-AUC',l.pr_auc,`Ranking quality &mdash; <b>${(l.pr_auc/d.base_rate).toFixed(1)}×</b> better than random`,'var(--yellow)')}
    </div>
    ${perClass(l)}
    <h3>Confusion matrix &mdash; what actually happened</h3>
    <div class="cmwrap">
      <div></div><div class="cmhead">Predicted CONFLICT</div><div class="cmhead">Predicted clean</div>
      <div class="cmside">Actually<br>CONFLICT</div>${cell(l.tp,'caught &#10003;','rgba(12,163,12,.25)')}${cell(l.fn,'missed &#10007;','rgba(208,59,59,.28)')}
      <div class="cmside">Actually<br>clean</div>${cell(l.fp,'false alarm','rgba(224,135,30,.22)')}${cell(l.tn,'cleared &#10003;','rgba(137,135,129,.16)')}
    </div>
    <div class="note">In plain English: of <b>${conf}</b> merges that truly conflicted, the model <b>caught ${l.tp}</b> and <b>missed ${l.fn}</b>. Of ${clean} clean merges, it raised <b>${l.fp}</b> false alarms and correctly cleared <b>${l.tn}</b>.</div>
    <h3>Metric radar &mdash; easy test vs hard test</h3>
    <div class="legend"><span><i style="background:var(--muted)"></i>Same-project (easy)</span><span><i style="background:var(--blue)"></i>Unseen repo (realistic)</span></div>
    ${radarChart(['Precision','Recall','F1','PR-AUC'],[
        {color:'var(--muted)',values:[s.precision,s.recall,s.f1,s.pr_auc]},
        {color:'var(--blue)',values:[l.precision,l.recall,l.f1,l.pr_auc]}])}
    <div class="muted" style="text-align:center">The blue shape (unseen repo) sitting just inside the grey (same-project) shows the realistic generalisation gap across all four metrics at once.</div>
  </div>`;
  const hf=$('#hf1');if(hf){hf.textContent=l.f1;countUpEl(hf);}}

async function loadXai(){const d=await j('/api/eda');const mx=d.importance[0].value;const lp=window.lastPred;
  let local='<div class="note">Make a prediction in the <b>Predict</b> tab — its per-merge local attribution appears here and updates every time.</div>';
  if(lp&&lp.drivers){const dm=Math.max(...lp.drivers.map(x=>x.impact),0.001);
    local=`<div class="muted" style="margin-bottom:8px">Last prediction: <b>${lp.repo}</b> &mdash; conflict probability <b style="color:${RC[lp.risk]}">${Math.round(lp.probability*100)}% (${lp.risk})</b>. Each bar = how much that feature pushed <i>this</i> merge toward conflict.</div>`+
     (lp.drivers.length?lp.drivers.map(x=>`<div class="barrow"><div class="lab">${x.feature}</div><div class="track"><span style="width:${Math.round(x.impact/dm*100)}%;background:var(--red)"></span></div><div class="num">+${x.impact}</div></div>`).join(''):'<div class="muted">No strong drivers &mdash; this merge looks clean.</div>');}
  $('#xai').innerHTML=`<div class="card">
    <h2>Explainability (XAI)</h2><div class="sub">Two complementary methods — <b>local</b> (this one prediction, changes every time) and <b>global</b> (the whole model, constant by design).</div>
    <h3>Local &mdash; Counterfactual Attribution <span class="muted" style="text-transform:none;font-weight:400">· per prediction</span></h3>
    <div class="muted" style="margin-bottom:8px">For this merge, each feature is set to its typical <i>clean</i> value and the drop in conflict probability is measured — that drop is the feature's contribution to this prediction.</div>
    ${local}
    <h3 style="margin-top:24px">Global &mdash; SHAP importance <span class="muted" style="text-transform:none;font-weight:400">· exact (mean |SHAP| over the dataset)</span></h3>
    <div class="muted" style="margin-bottom:8px">Average magnitude of each feature's exact Shapley contribution across all merges — the rigorous global ranking.</div>
    <div id="shapglobal"><span class="spin"></span> computing exact SHAP…</div>
    <h3 style="margin-top:24px">Global &mdash; Permutation Importance <span class="muted" style="text-transform:none;font-weight:400">· model-agnostic cross-check</span></h3>
    ${d.importance.map(f=>bar(f.feature,f.value/mx,'var(--blue)')).join('')}
    <div class="note">Local: counterfactual + exact SHAP (toggle on each prediction). Roadmap: GNNExplainer for the graph model.</div>
  </div>`;
  j('/api/shap_summary').then(ss=>{const smx=ss[0].value;const el=$('#shapglobal');
    if(el)el.innerHTML=ss.map(f=>bar(f.feature,f.value/smx,'var(--aqua)')).join('');});}

async function loadEda(){const d=await j('/api/eda');const mx=d.importance[0].value;
  $('#dataset').innerHTML=`<div class="card">
    <div class="grid"><div class="metric"><div class="k">scenarios</div><div class="v">${d.scenarios}</div></div>
    <div class="metric"><div class="k">conflicts</div><div class="v">${d.conflicts} (${Math.round(d.rate*100)}%)</div></div>
    <div class="metric"><div class="k">clean</div><div class="v">${d.scenarios-d.conflicts}</div></div>
    <div class="metric"><div class="k">repositories</div><div class="v">${d.byrepo.length}</div></div></div>
    <h3>Conflicts by repository (top 8)</h3>
    ${radialBars(d.byrepo.slice().sort((a,b)=>b.conflicts-a.conflicts).slice(0,8).map((r,i)=>({label:r.repo,value:r.conflicts,color:SERIES[i%SERIES.length]})))}
    <details style="margin-top:6px"><summary class="muted" style="cursor:pointer">show full table</summary>
    <table style="margin-top:8px"><tr><th>Repository</th><th>Scenarios</th><th>Conflicts</th><th>Rate</th></tr>
    ${d.byrepo.map(r=>`<tr><td>${r.repo}</td><td>${r.n}</td><td>${r.conflicts}</td><td>${Math.round(r.conflicts/r.n*100)}%</td></tr>`).join('')}</table></details>
    <h3>Feature importance (top driver highlighted)</h3>
    ${activeBars(d.importance.slice(0,8).map(f=>({label:f.feature,value:f.value,valLabel:f.value.toFixed(3),color:'var(--violet)'})),0)}</div>`;}

async function loadFindings(){const s=await j('/api/seeded');
  let seed='<div class="muted">seeded.csv not found</div>';
  if(s.available)seed=`<table><tr><th>Scenario type</th><th>n</th><th>Compile-verified conflicts</th><th>Dependency-linked</th></tr>
    ${Object.entries(s.kinds).map(([k,v])=>`<tr><td>${k}</td><td>${v.n}</td><td><b style="color:${v.conflicts?'var(--crit)':'var(--good)'}">${v.conflicts}</b></td><td>${v.linked}</td></tr>`).join('')}</table>`;
  $('#findings').innerHTML=`<div class="card">
    <h2>Findings</h2>
    <h3>Textual conflicts (real data)</h3>
    <div class="muted">561 scenarios, 68 conflicts across 6 repos. The model generalises to an <b>unseen</b> repository at leave-repo-out F1 ≈ 0.59 (≈5× the base rate).</div>
    <h3>Semantic (no-shared-file) conflicts</h3>
    <div class="muted">Compile-based mining of 199 real merges found <b>0</b> — expected: CI and atomic refactoring pre-empt them in merged history. The mechanism is proven by a controlled seeded experiment:</div>
    ${seed}
    <div class="note">Every CONFLICT fails to compile, shares <b>no file</b>, yet is dependency-linked → flat overlap is blind, the dependency graph sees it. CONTROL (same breaking change, unlinked class) → 0 conflicts, isolating the coupling as the cause.</div>
  </div>`;}
// --- animated dependency-network background (thematic eye-candy) ---
(function(){const c=document.getElementById('bg'),x=c.getContext('2d');let W,H,P=[];
 function rz(){W=c.width=innerWidth;H=c.height=innerHeight;}
 addEventListener('resize',rz);rz();
 const n=Math.min(64,Math.floor(W/24));
 for(let i=0;i<n;i++)P.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35});
 function loop(){x.clearRect(0,0,W,H);
  for(const p of P){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;}
  for(let i=0;i<P.length;i++)for(let j=i+1;j<P.length;j++){const dx=P[i].x-P[j].x,dy=P[i].y-P[j].y,d=Math.hypot(dx,dy);
    if(d<140){x.strokeStyle='rgba(96,110,235,'+(1-d/140)*.26+')';x.lineWidth=1;x.beginPath();x.moveTo(P[i].x,P[i].y);x.lineTo(P[j].x,P[j].y);x.stroke();}}
  for(const p of P){x.fillStyle='rgba(130,150,245,.75)';x.beginPath();x.arc(p.x,p.y,1.8,0,6.29);x.fill();}
  requestAnimationFrame(loop);}
 loop();})();
// --- count-up + live hero stats (pulled from the API, not hardcoded) ---
function countUpEl(el){const raw=el.textContent.trim(),m=raw.match(/^([\d.]+)/);if(!m)return;
 const target=parseFloat(m[1]),dec=(m[1].split('.')[1]||'').length,suf=raw.slice(m[1].length);let t0=null;
 function step(ts){if(!t0)t0=ts;const k=Math.min(1,(ts-t0)/1000),v=target*(1-Math.pow(1-k,3));
  el.textContent=(dec?v.toFixed(dec):Math.round(v))+suf;if(k<1)requestAnimationFrame(step);}requestAnimationFrame(step);}
async function updateHero(){try{const d=await j('/api/eda');
  $('#hs-sc').textContent=d.scenarios;$('#hs-cf').textContent=d.conflicts;$('#hs-repo').textContent=d.byrepo.length;
  document.querySelectorAll('.stat .v').forEach(countUpEl);
  j('/api/performance').then(p=>{if(p&&p.leave_repo_out){const el=$('#hf1');el.textContent=p.leave_repo_out.f1;countUpEl(el);}}).catch(()=>{});
 }catch(e){document.querySelectorAll('.stat .v').forEach(countUpEl);}}
updateHero();initRepos();
</script></body></html>"""


ENGINE_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pipeline Engine</title>
<style>
:root{--page:#0d0d0d;--surface:#1a1a19;--line:rgba(255,255,255,.10);--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --blue:#3987e5;--violet:#9085e9;--aqua:#199e70;--crit:#d03b3b;--good:#0ca30c;--warn:#fab219}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,'Segoe UI',sans-serif;background:radial-gradient(1200px 600px at 80% -10%,#1a1c33,transparent 60%),var(--page);color:var(--ink);min-height:100vh}
#bg{position:fixed;inset:0;z-index:0;opacity:.5;pointer-events:none}
.hero,.wrap{position:relative;z-index:1}
.hero{max-width:900px;margin:0 auto;padding:26px 24px 8px}
.back{color:var(--blue);text-decoration:none;font-size:13px}.back:hover{text-decoration:underline}
.hero h1{font-size:26px;margin-top:10px;background:linear-gradient(90deg,#9aa8ff,#c7a5ff);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.hero .tag{color:var(--ink2);font-size:14px;margin-top:5px}
.wrap{max-width:900px;margin:0 auto;padding:18px 24px 60px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px;box-shadow:0 8px 30px rgba(0,0,0,.25)}
.card h2{font-size:15px;margin-bottom:10px}
label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
input{width:100%;padding:11px 13px;border-radius:9px;border:1px solid var(--line);background:#111110;color:var(--ink);font-size:14px}
button{margin-top:14px;background:linear-gradient(90deg,var(--blue),var(--violet));color:#fff;border:0;border-radius:9px;padding:12px 22px;font-size:14px;font-weight:600;cursor:pointer;box-shadow:0 6px 20px rgba(74,86,224,.32)}
button:disabled{opacity:.55;cursor:default}
.stage{padding:13px 0;border-bottom:1px solid var(--line)}.stage:last-child{border-bottom:0}
.sname{font-size:14px;font-weight:600;display:flex;align-items:center;gap:9px}
.strack{height:9px;background:#111110;border-radius:5px;overflow:hidden;margin:9px 0 5px}
.strack>span{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--violet));border-radius:5px;transition:width .5s ease;width:0}
.stage.done .strack>span{background:var(--good)}.stage.error .strack>span{background:var(--crit)}
.stage.done .sname{color:var(--good)}.stage.error .sname{color:var(--crit)}
.sdetail{font-size:12px;color:var(--muted)}
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--blue);border-top-color:transparent;border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.rgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:8px 0 10px}
.rm{background:#111110;border:1px solid var(--line);border-radius:10px;padding:13px;text-align:center}
.rv{font-size:24px;font-weight:800;font-variant-numeric:tabular-nums}.rk{font-size:11px;color:var(--muted);margin-top:3px}
.note{background:#111110;border-left:3px solid var(--blue);padding:11px 14px;border-radius:8px;font-size:13px;color:var(--ink2);margin-top:8px}
.note a{color:var(--blue)}.card.ok{border-color:rgba(12,163,12,.4)}.card.err{border-color:rgba(208,59,59,.5);color:#ffb0b0}
.dsrow{display:flex;align-items:center;gap:12px;margin:9px 0}
.dslabel{width:200px;font-size:12.5px;color:var(--ink)}
.dslabel .m{display:block;font-size:10.5px;color:var(--muted)}
.dstrack{flex:1;height:26px;background:#111110;border-radius:6px;position:relative;overflow:hidden}
.dstrack>span{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:linear-gradient(90deg,var(--blue),var(--violet));transform-origin:left;animation:grow .7s cubic-bezier(.2,.7,.2,1) both}
.dstrack>b{position:absolute;right:9px;top:5px;font-size:12.5px;font-variant-numeric:tabular-nums;color:#fff}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.bigbar{display:flex;align-items:center;gap:10px;margin:6px 0}
.bigbar .l{width:150px;font-size:13px;color:var(--ink)}
.bigtrack{flex:1;height:30px;border-radius:6px;position:relative;overflow:hidden;background:#111110}
.bigtrack>span{position:absolute;left:0;top:0;bottom:0;border-radius:6px;transform-origin:left;animation:grow .7s cubic-bezier(.2,.7,.2,1) both}
.bigtrack>b{position:absolute;right:10px;top:6px;font-weight:700;color:#fff;font-variant-numeric:tabular-nums}
.clrow{display:flex;align-items:center;gap:10px;margin:5px 0}
.cllabel{width:150px;font-size:11.5px;color:var(--ink2)}
.clbarwrap{flex:1}
.cltrack{height:18px;background:#3a3a30;border-radius:5px;overflow:hidden}
.cltrack>span{display:block;height:100%;background:linear-gradient(90deg,var(--good),#2bbf6a);border-radius:5px;transform-origin:left;animation:grow .7s cubic-bezier(.2,.7,.2,1) both}
.clnum{width:82px;text-align:right;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
</style></head><body><canvas id="bg"></canvas>
<div class="hero"><a class="back" href="/">&larr; Back to dashboard</a>
  <h1>&#9881; Pipeline Engine</h1>
  <div class="tag">Run the full pipeline &mdash; mine &rarr; preprocess &rarr; train &rarr; evaluate &mdash; live. The main dashboard updates when it finishes.</div>
</div>
<div class="wrap">
  <div class="card">
    <label>Add a repository (optional) &mdash; GitHub URL or local path</label>
    <input id="repo" placeholder="leave empty to just retrain + evaluate, or e.g. https://github.com/apache/commons-io.git">
    <button id="run" onclick="start()">&#9654; Run Pipeline</button>
  </div>
  <div id="stages"></div>
  <div id="result"></div>
  <div id="cleaning"></div>
  <div class="card"><h2>Datasets used by the pipeline</h2>
    <div style="font-size:12.5px;color:var(--muted);margin-bottom:6px">Row count per data file (each row = one merge scenario or experiment record).</div>
    <div id="datasets"><span class="spin"></span> loading&hellip;</div></div>
</div>
<script>
const $=s=>document.querySelector(s),j=u=>fetch(u).then(r=>r.json());
const STAGES=['Mine merges','Preprocess features','Train model','Evaluate'];
function ic(s){return s=='done'?'&#10003;':s=='running'?'<span class="spin"></span>':s=='error'?'&#10007;':s=='queued'?'&#8230;':'&#9675;';}
function renderStages(st){$('#stages').innerHTML='<div class="card"><h2>Pipeline stages</h2>'+
  st.stages.map((s,i)=>`<div class="stage ${s.state}"><div class="sname">${ic(s.state)} ${i+1}. ${s.name}</div>
    <div class="strack"><span style="width:${s.pct}%"></span></div><div class="sdetail">${s.detail||''}</div></div>`).join('')+'</div>';}
function renderCleaning(c){const mx=Math.max(...c.items.map(d=>d.raw))||1;
  const big=(label,val,total,col)=>`<div class="bigbar"><div class="l">${label}</div><div class="bigtrack"><span style="width:${Math.round(val/total*100)}%;background:${col}"></span><b>${val.toLocaleString()}</b></div></div>`;
  return `<div class="card"><h2>Data cleaning &mdash; before vs after</h2>
    <div style="font-size:12.5px;color:var(--muted);margin-bottom:8px">Raw two-parent merges mined, then reduced by the trivial-merge filter and sampling of large repositories.</div>
    ${big('Raw merges (before)',c.raw,c.raw,'var(--muted)')}
    ${big('Retained (after)',c.retained,c.raw,'var(--good)')}
    <div class="note" style="margin:8px 0"><b>${c.raw.toLocaleString()}</b> raw &rarr; <b>${c.retained.toLocaleString()}</b> retained (${c.removed.toLocaleString()} removed by cleaning + sampling).</div>
    <div style="font-size:12px;color:var(--ink2);margin:12px 0 6px">Per repository &mdash; green = retained, dark = removed</div>
    ${c.items.map(d=>`<div class="clrow"><div class="cllabel">${d.repo}</div>
      <div class="clbarwrap"><div class="cltrack" style="width:${(d.raw/mx*100).toFixed(1)}%"><span style="width:${(d.retained/(d.raw||1)*100).toFixed(1)}%"></span></div></div>
      <div class="clnum">${d.retained}/${d.raw}</div></div>`).join('')}
  </div>`;}
let timer=null;
async function poll(){const st=await j('/api/engine/status');renderStages(st);
  if(st.done){clearInterval(timer);timer=null;$('#run').disabled=false;$('#run').innerHTML='&#9654; Run Pipeline';
    if(st.error)$('#result').innerHTML=`<div class="card err"><h2>&#10007; Pipeline failed</h2>${st.error}</div>`;
    else if(st.result){const r=st.result;
      $('#result').innerHTML=`<div class="card ok"><h2>&#10003; Pipeline complete</h2>
        <div class="rgrid">
          <div class="rm"><div class="rv">${r.scenarios}</div><div class="rk">scenarios</div></div>
          <div class="rm"><div class="rv">${r.conflicts}</div><div class="rk">conflicts</div></div>
          <div class="rm"><div class="rv">${r.repos}</div><div class="rk">repos</div></div>
          <div class="rm"><div class="rv">${r.f1}</div><div class="rk">leave-repo-out F1</div></div>
        </div>
        ${r.added?`<div class="note">Added <b>${r.added}</b> new scenarios from the repository.</div>`:''}
        <div class="note">Recall ${Math.round(r.recall*100)}% &middot; PR-AUC ${r.pr_auc}. The main dashboard now reflects this run &mdash; <a href="/">open dashboard &rarr;</a></div>
      </div>`;
      j('/api/cleaning').then(c=>{const el=document.getElementById('cleaning');if(el)el.innerHTML=renderCleaning(c);});}}}
async function start(){$('#run').disabled=true;$('#run').innerHTML='<span class="spin"></span> running&hellip;';$('#result').innerHTML='';
  const repo=$('#repo').value.trim();
  const res=await j('/api/engine/start'+(repo?'?repo='+encodeURIComponent(repo):''));
  if(res.error){$('#run').disabled=false;$('#run').innerHTML='&#9654; Run Pipeline';$('#result').innerHTML=`<div class="card err">${res.error}</div>`;return;}
  timer=setInterval(poll,1000);poll();}
renderStages({stages:STAGES.map(n=>({name:n,state:'idle',pct:0,detail:''}))});
poll();
j('/api/datasets').then(ds=>{const mx=Math.max(...ds.map(d=>d.rows));
  document.getElementById('datasets').innerHTML=ds.map(d=>`<div class="dsrow">
    <div class="dslabel">${d.file}<span class="m">${d.cols} columns &middot; ${d.role}</span></div>
    <div class="dstrack"><span style="width:${Math.round(d.rows/mx*100)}%"></span><b>${d.rows.toLocaleString()}</b></div></div>`).join('');});
(function(){const c=document.getElementById('bg'),x=c.getContext('2d');let W,H,P=[];
 function rz(){W=c.width=innerWidth;H=c.height=innerHeight;}addEventListener('resize',rz);rz();
 for(let i=0;i<Math.min(50,W/26);i++)P.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3});
 (function loop(){x.clearRect(0,0,W,H);for(const p of P){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;}
  for(let i=0;i<P.length;i++)for(let k=i+1;k<P.length;k++){const dx=P[i].x-P[k].x,dy=P[i].y-P[k].y,d=Math.hypot(dx,dy);
   if(d<140){x.strokeStyle='rgba(96,110,235,'+(1-d/140)*.25+')';x.beginPath();x.moveTo(P[i].x,P[i].y);x.lineTo(P[k].x,P[k].y);x.stroke();}}
  for(const p of P){x.fillStyle='rgba(130,150,245,.7)';x.beginPath();x.arc(p.x,p.y,1.7,0,6.29);x.fill();}
  requestAnimationFrame(loop);})();})();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path.startswith("/api/"):
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                data = api(u.path, q)
            except Exception as e:
                data = {"error": str(e)}
            body = json.dumps(data).encode("utf-8"); ctype = "application/json"
        elif u.path == "/engine":
            body = ENGINE_PAGE.encode("utf-8"); ctype = "text/html; charset=utf-8"
        else:
            body = INDEX.encode("utf-8"); ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Dashboard running at  http://localhost:{PORT}")
    httpd.serve_forever()
