import subprocess, csv, sys, os

def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a],
                          capture_output=True, text=True).stdout.strip()

def feats(repo, base, tip):
    n = g(repo, "rev-list", "--count", base + ".." + tip)
    f = set(g(repo, "diff", "--name-only", base, tip).splitlines())
    a = len(set(g(repo, "log", "--format=%ae", base + ".." + tip).splitlines()))
    ss = g(repo, "diff", "--shortstat", base, tip)
    ins = dels = 0
    for p in ss.split(", "):
        if "insertion" in p: ins = int(p.split()[0])
        elif "deletion" in p: dels = int(p.split()[0])
    return int(n or 0), f, a, ins + dels

def mine(repo, w, name, limit):
    merges = g(repo, "log", "--merges", "--format=%H").splitlines()
    if limit: merges = merges[:limit]
    rows = 0
    for m in merges:
        ps = g(repo, "rev-list", "--parents", "-n", "1", m).split()[1:]
        if len(ps) != 2: continue
        p1, p2 = ps
        base = g(repo, "merge-base", p1, p2)
        if not base: continue
        r = subprocess.run(["git", "-C", repo, "merge-tree", "--write-tree", p1, p2],
                           capture_output=True, text=True)
        if r.returncode not in (0, 1): continue
        label = 1 if r.returncode == 1 else 0
        cf = [l.split("Merge conflict in ", 1)[1] for l in r.stdout.splitlines()
              if "Merge conflict in " in l]
        c1, f1, a1, ch1 = feats(repo, base, p1)
        c2, f2, a2, ch2 = feats(repo, base, p2)
        if c1 == 0 or c2 == 0: continue  # drop trivial merges (one side never diverged)
        ov = f1 & f2; un = f1 | f2
        code_conflict = 1 if any(f.strip().endswith(".java") for f in cf) else 0
        w.writerow([name, m, c1, c2, len(f1), len(f2), len(ov),
                    round(len(ov) / len(un), 4) if un else 0,
                    a1, a2, ch1, ch2, ";".join(cf), code_conflict, label])
        rows += 1
    return rows

def main():
    args = sys.argv[1:]
    limit = None; out = "all.csv"; repos = []
    i = 0
    while i < len(args):
        if args[i] == "--limit": limit = int(args[i + 1]); i += 2
        elif args[i] == "--out": out = args[i + 1]; i += 2
        else: repos.append(args[i]); i += 1
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["repo", "merge", "commits_p1", "commits_p2", "files_p1",
                    "files_p2", "overlap_files", "overlap_ratio", "authors_p1",
                    "authors_p2", "churn_p1", "churn_p2", "conflict_files",
                    "code_conflict", "label"])
        tot = 0
        for repo in repos:
            n = mine(repo, w, os.path.basename(repo.rstrip("/\\")), limit)
            tot += n
            print(repo + ": " + str(n) + " scenarios", file=sys.stderr)
    pos = sum(1 for r in csv.reader(open(out)) if r and r[-1] == "1")
    print("DONE: " + str(tot) + " scenarios, " + str(pos) + " conflicts", file=sys.stderr)

main()
