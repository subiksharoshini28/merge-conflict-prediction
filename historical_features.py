"""
Historical / process features (the strongest legitimate predictors).

For each merge scenario we compute features using ONLY information available BEFORE
that merge (strict point-in-time; no leakage). Merges are ordered by commit date
per repository; each feature looks only at prior merges.

  hist_conflict_rate      prior conflict rate in the repo (conflicts/merges so far)
  hist_prior_conflicts    count of prior conflicting merges
  hist_prior_merges       count of prior merges seen
  file_conflict_overlap   how many files changed by THIS merge have appeared in the
                          conflicted-file set of PRIOR conflicting merges (change-to-conflict coupling)
  divergence_days         days between the merge base and the merge commit (branch age)

Uses git for commit dates + tree diffs (no blob content) and the labels/conflict_files
already in all.csv. Repos live in D:\SM_PROJ\repos.

Usage:  python historical_features.py [--in all.csv] [--out historical.csv]
"""
import subprocess, csv, sys, os
from collections import defaultdict

COLS = ["hist_conflict_rate", "hist_prior_conflicts", "hist_prior_merges",
        "file_conflict_overlap", "divergence_days"]


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def cdate(repo, sha):
    out = g(repo, "show", "-s", "--format=%ct", sha)
    try:
        return int(out.splitlines()[0])
    except Exception:
        return 0


def changed(repo, base, tip):
    return set(x for x in g(repo, "diff", "--name-only", base, tip).splitlines() if x)


def main():
    args = sys.argv[1:]; inp = "all.csv"; out = "historical.csv"
    i = 0
    while i < len(args):
        if args[i] == "--in": inp = args[i+1]; i += 2
        elif args[i] == "--out": out = args[i+1]; i += 2
        else: i += 1

    rows = list(csv.reader(open(inp))); h = rows[0]; data = rows[1:]
    ix = {c: h.index(c) for c in ("repo", "merge", "conflict_files", "label")}
    by = defaultdict(list)
    for r in data:
        if r:
            by[r[ix["repo"]]].append(r)

    results = {}
    for repo_name, rlist in by.items():
        repo = os.path.join("repos", repo_name)
        if not os.path.isdir(os.path.join(repo, ".git")):
            print(f"[{repo_name}] MISSING repo -> zero features for {len(rlist)} rows", file=sys.stderr)
            for r in rlist:
                results[(repo_name, r[ix["merge"]])] = {c: 0 for c in COLS}
            continue
        dated = sorted(((cdate(repo, r[ix["merge"]]), r) for r in rlist), key=lambda t: t[0])
        pm = pc = 0
        cf_counts = defaultdict(int)
        for ts, r in dated:
            m = r[ix["merge"]]
            f = {"hist_conflict_rate": round(pc / pm, 4) if pm else 0.0,
                 "hist_prior_conflicts": pc, "hist_prior_merges": pm,
                 "file_conflict_overlap": 0, "divergence_days": 0}
            ps = g(repo, "rev-list", "--parents", "-n", "1", m).split()[1:]
            if len(ps) == 2:
                base = g(repo, "merge-base", ps[0], ps[1])
                if base:
                    cset = changed(repo, base, ps[0]) | changed(repo, base, ps[1])
                    f["file_conflict_overlap"] = sum(cf_counts.get(x, 0) for x in cset)
                    bd = cdate(repo, base)
                    f["divergence_days"] = round((ts - bd) / 86400, 1) if bd and ts else 0
            results[(repo_name, m)] = f
            # update priors AFTER scoring (strictly point-in-time)
            pm += 1
            if int(r[ix["label"]] or 0) == 1:
                pc += 1
                for x in (r[ix["conflict_files"]] or "").split(";"):
                    if x.strip():
                        cf_counts[x.strip()] += 1
        print(f"[{repo_name}] {len(rlist)} scenarios, {pc} conflicts", file=sys.stderr, flush=True)

    w = csv.writer(open(out, "w", newline=""))
    w.writerow(["repo", "merge"] + COLS)
    for r in data:
        f = results.get((r[ix["repo"]], r[ix["merge"]]), {})
        w.writerow([r[ix["repo"]], r[ix["merge"]]] + [f.get(c, 0) for c in COLS])
    print(f"HISTORICAL DONE -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
