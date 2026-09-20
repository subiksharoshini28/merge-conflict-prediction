"""
Recompute divergence_days as a LEAK-FREE, inference-available branch-age feature:
    divergence_days = (newest branch-tip commit date - merge-base date) / 86400

At training time the two branch tips are the merge commit's parents (merge^1, merge^2);
the base is merge-base(parent1, parent2). This is EXACTLY what is computable before the
merge happens (two un-merged branch tips), so training and inference use one definition.

Writes all_enriched.csv = all.csv + divergence_days.
"""
import subprocess, os, sys
import pandas as pd

REPOS = "repos"   # D:/SM_PROJ/repos/<repo>


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def cdate(repo, sha):
    out = g(repo, "show", "-s", "--format=%ct", sha)
    try:
        return int(out.splitlines()[0])
    except Exception:
        return 0


def divergence_days(repo, merge):
    p1, p2 = g(repo, "rev-parse", merge + "^1"), g(repo, "rev-parse", merge + "^2")
    if not p1 or not p2:
        return 0.0
    base = g(repo, "merge-base", p1, p2)
    if not base:
        return 0.0
    tip = max(cdate(repo, p1), cdate(repo, p2))
    b = cdate(repo, base)
    return round(max(0, tip - b) / 86400.0, 3) if b else 0.0


def main():
    df = pd.read_csv("all.csv")
    vals, n = [], len(df)
    for i, row in df.iterrows():
        repo = os.path.join(REPOS, str(row["repo"]))
        vals.append(divergence_days(repo, str(row["merge"])) if os.path.isdir(repo) else 0.0)
        if (i + 1) % 200 == 0:
            sys.stderr.write(f"  {i+1}/{n}\n")
    df["divergence_days"] = vals
    # keep label last if present
    if "label" in df.columns:
        cols = [c for c in df.columns if c != "label"] + ["label"]
        df = df[cols]
    df.to_csv("all_enriched.csv", index=False)
    nz = sum(1 for v in vals if v > 0)
    print(f"wrote all_enriched.csv ({n} rows); divergence_days nonzero on {nz} rows "
          f"(median={pd.Series(vals).median():.1f}d, max={max(vals):.0f}d)")


if __name__ == "__main__":
    main()
