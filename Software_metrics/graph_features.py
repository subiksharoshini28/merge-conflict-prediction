"""
Phase 3 - dependency-graph features (THE NOVELTY).

For each merge scenario we build a file-level Java import graph AT THE MERGE BASE
(point-in-time; content read via `git grep <base>` only, never a checkout), then
compute how the two branches' changed files sit in that graph. The payoff features
are the cross-branch interactions that fire even when the two sides share NO file:
    cross_edges, min_graph_distance, shared_neighbors, shared_modules

Reads all.csv (needs repo + merge), re-derives p1/p2/base from the merge commit,
appends graph columns, writes all_graph.csv (label stays last).

Usage:  python graph_features.py [--limit N] [--in all.csv] [--out all_graph.csv]
Repo folders are resolved as D:/<repo-basename> (matches the miner's naming).
"""
import subprocess, csv, sys, os, re
from collections import deque, defaultdict
import networkx as nx

REPO_ROOT = "D:/"
IMPORT_RE = re.compile(r'^\s*(package|import)\s+(static\s+)?([\w.]+)\s*(\.\*)?\s*;')
DIST_CAP = 12          # BFS cutoff / sentinel for "no path"
_graph_cache = {}      # (repo, base) -> (DiGraph, {path: package})


# These clones are blobless (--filter=blob:none): any op needing file content
# lazily fetches blobs over HTTPS, and a stalled fetch can hang forever. Make git
# abort a transfer that drops below 1 KB/s for 20s, then retry the whole call.
_GIT_ROBUST = ["-c", "http.lowSpeedLimit=100", "-c", "http.lowSpeedTime=120"]


def g(repo, *a, timeout=600, retries=4):
    last = ""
    for attempt in range(retries):
        try:
            r = subprocess.run(["git", *_GIT_ROBUST, "-C", repo, *a],
                               capture_output=True, encoding="utf-8", errors="replace",
                               timeout=timeout)
            if r.returncode == 0 or r.stdout:
                return r.stdout
            last = r.stderr
        except subprocess.TimeoutExpired:
            last = "timeout"  # hung fetch -> kill and retry with a fresh connection
    sys.stderr.write(f"  WARN git {a[0]} on {os.path.basename(repo)} failed after "
                     f"{retries} tries ({last[:60]!r})\n")
    return ""


def java_files(repo, rev):
    out = g(repo, "ls-tree", "-r", "--name-only", rev)
    return [p for p in out.splitlines() if p.endswith(".java")]


def build_graph(repo, base):
    """File-level import graph at `base`. Nodes = java file paths that exist at base."""
    key = (repo, base)
    if key in _graph_cache:
        return _graph_cache[key]

    files = java_files(repo, base)
    fileset = set(files)
    # FQCN of a file's top-level class ~= package + filename-stem (Java convention).
    fqcn2path, pkg2paths, path2pkg = {}, defaultdict(list), {}

    # One git grep pulls every package/import line across all java files at this
    # commit (format: base:path:line) - far faster than a git show per file.
    grep = g(repo, "grep", "-I", "-E", r'^\s*(package|import)\s', base, "--", "*.java")
    per_file = defaultdict(lambda: {"pkg": None, "imports": []})
    for line in grep.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, path, content = parts
        if path not in fileset:
            continue
        m = IMPORT_RE.match(content)
        if not m:
            continue
        kind, _static, name, wild = m.group(1), m.group(2), m.group(3), m.group(4)
        if kind == "package":
            per_file[path]["pkg"] = name
        else:
            per_file[path]["imports"].append((name, bool(wild)))

    # Index every file by its declared FQCN and by package.
    for path in files:
        pkg = per_file[path]["pkg"]
        stem = os.path.basename(path)[:-5]  # strip .java
        path2pkg[path] = pkg or ""
        if pkg:
            fqcn2path[pkg + "." + stem] = path
            pkg2paths[pkg].append(path)

    # Edges: file -> file it depends on (intra-repo only; external imports dropped).
    G = nx.DiGraph()
    G.add_nodes_from(files)
    for path in files:
        for name, wild in per_file[path]["imports"]:
            if wild:  # import a.b.*  -> link to every file in package a.b
                for tgt in pkg2paths.get(name, ()):
                    if tgt != path:
                        G.add_edge(path, tgt)
            else:
                tgt = fqcn2path.get(name)
                if tgt is None:  # maybe a static-member import: strip last segment once
                    tgt = fqcn2path.get(name.rsplit(".", 1)[0])
                if tgt and tgt != path:
                    G.add_edge(path, tgt)

    _graph_cache[key] = (G, path2pkg)
    return G, path2pkg


def changed_java(repo, base, tip):
    out = g(repo, "diff", "--name-only", base, tip)
    return [p for p in out.splitlines() if p.endswith(".java")]


def bounded_min_distance(UG, S1, S2, cap):
    """Min undirected hop-distance between set S1 and set S2 (multi-source BFS, cutoff=cap)."""
    if not S1 or not S2:
        return cap
    if S1 & S2:
        return 0
    targets = S2
    dist = {s: 0 for s in S1 if s in UG}
    q = deque(dist)
    while q:
        u = q.popleft()
        d = dist[u]
        if d >= cap:
            continue
        for v in UG.neighbors(u):
            if v in targets:
                return d + 1
            if v not in dist:
                dist[v] = d + 1
                q.append(v)
    return cap


def scenario_feats(repo, base, p1, p2):
    G, path2pkg = build_graph(repo, base)
    UG = G.to_undirected(as_view=True)
    nodes = set(G.nodes)

    c1_all = changed_java(repo, base, p1)
    c2_all = changed_java(repo, base, p2)
    S1 = set(c1_all) & nodes          # changed files that exist at base (graph nodes)
    S2 = set(c2_all) & nodes

    def deg(S):
        fin = sum(G.in_degree(n) for n in S)
        fout = sum(G.out_degree(n) for n in S)
        return fin, fout
    fin1, fout1 = deg(S1)
    fin2, fout2 = deg(S2)

    # cross_edges: dependency edges linking a p1-changed file to a p2-changed file.
    cross = 0
    for u in S1:
        for v in G.successors(u):
            if v in S2:
                cross += 1
        for v in G.predecessors(u):
            if v in S2:
                cross += 1

    # shared_neighbors: nodes adjacent to BOTH sides (graph coupling without a shared file).
    n1 = set()
    for u in S1:
        n1.update(UG.neighbors(u))
    n2 = set()
    for u in S2:
        n2.update(UG.neighbors(u))
    shared_neigh = len((n1 & n2) - S1 - S2)

    min_dist = bounded_min_distance(UG, S1, S2, DIST_CAP)

    # shared_modules: java packages touched by both sides (uses ALL changed java files).
    mods1 = {path2pkg.get(p, os.path.dirname(p)) for p in c1_all}
    mods2 = {path2pkg.get(p, os.path.dirname(p)) for p in c2_all}
    shared_mods = len(mods1 & mods2)

    return dict(
        g_nodes=G.number_of_nodes(), g_edges=G.number_of_edges(),
        g_changed_p1=len(S1), g_changed_p2=len(S2),
        g_fanin_p1=fin1, g_fanout_p1=fout1, g_fanin_p2=fin2, g_fanout_p2=fout2,
        cross_edges=cross, shared_neighbors=shared_neigh,
        min_graph_distance=min_dist, shared_modules=shared_mods,
    )


GRAPH_COLS = ["g_nodes", "g_edges", "g_changed_p1", "g_changed_p2",
              "g_fanin_p1", "g_fanout_p1", "g_fanin_p2", "g_fanout_p2",
              "cross_edges", "shared_neighbors", "min_graph_distance", "shared_modules"]


def main():
    args = sys.argv[1:]
    limit = None; inp = "all.csv"; out = "all_graph.csv"
    i = 0
    while i < len(args):
        if args[i] == "--limit": limit = int(args[i + 1]); i += 2
        elif args[i] == "--in": inp = args[i + 1]; i += 2
        elif args[i] == "--out": out = args[i + 1]; i += 2
        else: i += 1

    with open(inp, newline="") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    if limit: data = data[:limit]
    lab = header.index("label")
    new_header = header[:lab] + GRAPH_COLS + [header[lab]]

    done = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(new_header)
        for r in data:
            repo_name, merge = r[0], r[1]
            repo = REPO_ROOT + repo_name
            ps = g(repo, "rev-list", "--parents", "-n", "1", merge).split()[1:]
            if len(ps) != 2:
                continue
            p1, p2 = ps
            base = g(repo, "merge-base", p1, p2).strip()
            if not base:
                continue
            f = scenario_feats(repo, base, p1, p2)
            w.writerow(r[:lab] + [f[c] for c in GRAPH_COLS] + [r[lab]])
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(data)} scenarios", file=sys.stderr)
    print(f"DONE: wrote {done} rows with {len(GRAPH_COLS)} graph features -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
