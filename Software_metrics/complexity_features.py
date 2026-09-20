"""
Additional software metrics from the concept: CBO, Cyclomatic Complexity, Dependency Depth.
Computed per scenario at the MERGE BASE (point-in-time), for the files each branch changes.

  CBO (Coupling Between Objects) : in+out degree of a changed file in the import graph
                                   (number of classes it is coupled to). Aggregated per side.
  Cyclomatic Complexity          : 1 + count of decision points (if/for/while/case/catch/&&/||/?)
                                   in the base version of each changed file. Summed per side.
  Dependency Depth               : longest outgoing dependency chain from a changed file
                                   (bounded DFS). Max per side.

Reuses the graph builder from graph_features. Writes complexity.csv keyed by (repo, merge)
so it can be joined onto the feature table. Works offline for hydrated repos.

Usage:  python complexity_features.py [--out complexity.csv] REPO [REPO ...]   (REPO = D:/name)
"""
import subprocess, csv, sys, os, re
import graph_features as gf   # reuse g(), build_graph(), changed_java()

DECISION = re.compile(r'(\bif\b|\bfor\b|\bwhile\b|\bcase\b|\bcatch\b|&&|\|\||\?)')
DEPTH_CAP = 20


def cyclomatic(repo, base, path):
    """1 + decision points in the base version of a file (0 if file absent at base)."""
    content = gf.g(repo, "show", f"{base}:{path}")
    if not content:
        return 0
    # strip line/block comments and strings crudely to reduce false hits
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.S)
    content = re.sub(r'"(\\.|[^"\\])*"', '""', content)
    return 1 + len(DECISION.findall(content))


def depth(G, node, cap=DEPTH_CAP):
    """Longest outgoing dependency chain length from node (bounded, cycle-safe)."""
    best = 0
    stack = [(node, 0, frozenset([node]))]
    while stack:
        u, d, seen = stack.pop()
        if d > best:
            best = d
        if d >= cap:
            continue
        for v in G.successors(u):
            if v not in seen:
                stack.append((v, d + 1, seen | {v}))
    return best


def side_metrics(repo, base, tip, G):
    changed = gf.changed_java(repo, base, tip)
    nodes = [f for f in changed if f in G]
    cbo = sum(G.in_degree(f) + G.out_degree(f) for f in nodes)
    cc = sum(cyclomatic(repo, base, f) for f in changed)
    dep = max((depth(G, f) for f in nodes), default=0)
    return cbo, cc, dep


def main():
    args = sys.argv[1:]
    out = "complexity.csv"; repos = []
    i = 0
    while i < len(args):
        if args[i] == "--out": out = args[i + 1]; i += 2
        else: repos.append(args[i]); i += 1

    fh = open(out, "w", newline=""); w = csv.writer(fh)
    w.writerow(["repo", "merge", "cbo_p1", "cbo_p2", "cyclomatic_p1", "cyclomatic_p2",
                "dep_depth_p1", "dep_depth_p2"])
    for repo in repos:
        name = os.path.basename(repo.rstrip("/\\"))
        merges = gf.g(repo, "log", "--merges", "--min-parents=2", "--max-parents=2",
                      "--format=%H").split()
        done = 0
        for m in merges:
            ps = gf.g(repo, "rev-list", "--parents", "-n", "1", m).split()[1:]
            if len(ps) != 2:
                continue
            p1, p2 = ps
            base = gf.g(repo, "merge-base", p1, p2).strip()
            if not base:
                continue
            G, _ = gf.build_graph(repo, base)
            cbo1, cc1, d1 = side_metrics(repo, base, p1, G)
            cbo2, cc2, d2 = side_metrics(repo, base, p2, G)
            w.writerow([name, m, cbo1, cbo2, cc1, cc2, d1, d2])
            done += 1
            if done % 20 == 0:
                print(f"  [{name}] {done} scenarios", file=sys.stderr, flush=True)
        print(f"[{name}] DONE {done}", file=sys.stderr, flush=True)
    fh.close()
    print("COMPLEXITY FEATURES FINISHED", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
