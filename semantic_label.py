"""
Option 2 - SEMANTIC (build/compile) conflict labelling, offline, javac-only.

A semantic (higher-order, "no-shared-file") conflict is a merge that is TEXTUALLY
clean (git merge-tree succeeds, no markers) yet the MERGED code fails to compile,
while BOTH parents compiled. javac catches the classic case: branch A renames/removes
a symbol, branch B (a different file) still uses it -> merges cleanly -> won't compile.

Only works for dependency-light repos that compile with the JDK alone (commons-lang,
commons-io, commons-collections). netty / log4j2 need Maven and are out of scope here.

For each clean-textual merge:
  p1_ok, p2_ok, merged_ok = compile(p1 tree), compile(p2 tree), compile(merged tree)
  semantic_conflict = 1  if p1_ok and p2_ok and not merged_ok
                    = 0  if p1_ok and p2_ok and merged_ok
                    skip if a parent does not compile (cannot attribute to the merge)

Usage:  python semantic_label.py [--src src/main/java] [--out semantic.csv] REPO [REPO...]
Repos given as D:/<name>. Caches compile results per tree OID.
"""
import subprocess, csv, sys, os, tempfile, tarfile, io, shutil

_cache = {}   # tree_oid -> (compiles: bool, err: str)


def g(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)


def tree_of(repo, ref):
    return g(repo, "rev-parse", ref + "^{tree}").stdout.strip()


def compiles(repo, tree, src, release=None):
    key = (tree, release)
    if key in _cache:
        return _cache[key]
    d = tempfile.mkdtemp()
    try:
        ar = subprocess.run(["git", "-C", repo, "archive", "--format=tar", tree, src],
                            capture_output=True)
        if ar.returncode != 0 or not ar.stdout:
            res = (None, "no-src"); _cache[key] = res; return res
        tarfile.open(fileobj=io.BytesIO(ar.stdout)).extractall(d)
        srcs = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".java")]
        if not srcs:
            res = (None, "no-java"); _cache[key] = res; return res
        lst = os.path.join(d, "srcs.txt"); open(lst, "w", encoding="utf-8").write("\n".join(srcs))
        cmd = ["javac", "-d", os.path.join(d, "out"), "-proc:none", "-nowarn"]
        if release:
            cmd += ["--release", str(release)]
        cmd.append("@" + lst)
        r = subprocess.run(cmd, capture_output=True, text=True)
        # first real error line (javac prefixes errors with the file path)
        err = ""
        for line in r.stderr.splitlines():
            if ": error:" in line:
                err = line.strip(); break
        res = (r.returncode == 0, err)
        _cache[key] = res
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    args = sys.argv[1:]
    src = "src/main/java"; out = "semantic.csv"; release = None; repos = []
    i = 0
    while i < len(args):
        if args[i] == "--src": src = args[i+1]; i += 2
        elif args[i] == "--out": out = args[i+1]; i += 2
        elif args[i] == "--release": release = args[i+1]; i += 2
        else: repos.append(args[i]); i += 1

    fh = open(out, "w", newline=""); w = csv.writer(fh)
    w.writerow(["repo", "merge", "p1_compiles", "p2_compiles", "merged_compiles",
                "status", "semantic_conflict", "error"])
    totals = {}
    for repo in repos:
        name = os.path.basename(repo.rstrip("/\\"))
        merges = g(repo, "log", "--merges", "--min-parents=2", "--max-parents=2",
                   "--format=%H").stdout.split()
        n_sem = n_clean = n_skip = n_textual = 0
        for k, m in enumerate(merges):
            ps = g(repo, "rev-list", "--parents", "-n", "1", m).stdout.split()[1:]
            if len(ps) != 2: continue
            p1, p2 = ps
            mt = subprocess.run(["git", "-C", repo, "merge-tree", "--write-tree", p1, p2],
                                capture_output=True, text=True)
            if mt.returncode not in (0, 1):
                continue
            if mt.returncode == 1:   # textual conflict - not what we study here
                n_textual += 1
                w.writerow([name, m, "", "", "", "textual_conflict", "", ""]); continue
            merged_tree = mt.stdout.strip().splitlines()[0]
            o1, _ = compiles(repo, tree_of(repo, p1), src, release)
            o2, _ = compiles(repo, tree_of(repo, p2), src, release)
            om, em = compiles(repo, merged_tree, src, release)
            if o1 and o2 and om is not None:
                if om:
                    status, lab = "clean", 0; n_clean += 1
                else:
                    status, lab = "SEMANTIC_CONFLICT", 1; n_sem += 1
                w.writerow([name, m, o1, o2, om, status, lab, em])
            else:
                n_skip += 1
                w.writerow([name, m, o1, o2, om, "skip(parent/no-src)", "", em or ""])
            if (k + 1) % 10 == 0:
                print(f"  [{name}] {k+1}/{len(merges)}  sem={n_sem} clean={n_clean} "
                      f"skip={n_skip} textual={n_textual}", file=sys.stderr, flush=True)
        totals[name] = (n_sem, n_clean, n_skip, n_textual)
        print(f"[{name}] DONE  semantic={n_sem}  clean={n_clean}  "
              f"skipped={n_skip}  textual={n_textual}", file=sys.stderr, flush=True)
    fh.close()
    print("SEMANTIC LABELLING FINISHED:", totals, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
