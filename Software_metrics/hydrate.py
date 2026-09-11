"""
Pre-hydrate blobless clones efficiently before graph extraction.

The clones are --filter=blob:none, so reading file content lazily fetches blobs.
Doing that per-merge-base means hundreds of tiny fetches, each paying full HTTPS
round-trip latency (deadly on a slow link). Instead we compute every merge base up
front and issue ONE `git grep` over a CHUNK of bases at a time, so git batches all
their missing blobs into a single fetch. A handful of fetches per repo replaces
hundreds. After this runs, graph_features.py reads everything from local cache.

Usage:  python hydrate.py [all.csv]
"""
import subprocess, csv, sys, os
from collections import defaultdict

REPO_ROOT = "D:/"
CHUNK = 20
# Abort ONLY if a transfer truly stalls (<100 B/s for 2 min). A higher threshold
# was aborting valid-but-slow downloads on this link and thrashing on retries.
ROBUST = ["-c", "http.lowSpeedLimit=100", "-c", "http.lowSpeedTime=120"]


def g(repo, *a, timeout=None):
    return subprocess.run(["git", "-C", repo, *a],
                          capture_output=True, text=True, timeout=timeout).stdout


def bases_for(repo, merges):
    seen = set()
    for m in merges:
        out = g(repo, "rev-list", "--parents", "-n", "1", m).split()
        if len(out) != 3:      # merge + 2 parents
            continue
        b = g(repo, "merge-base", out[1], out[2]).strip()
        if b:
            seen.add(b)
    return sorted(seen)


def objsize_mb(repo):
    try:
        out = subprocess.run(["du", "-sm", repo + "/.git/objects"],
                             capture_output=True, text=True).stdout
        return out.split("\t")[0].strip()
    except Exception:
        return "?"


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else "all.csv"
    rows = list(csv.reader(open(inp)))[1:]
    by_repo = defaultdict(list)
    for r in rows:
        if r:
            by_repo[r[0]].append(r[1])

    for repo_name, merges in by_repo.items():
        repo = REPO_ROOT + repo_name
        bs = bases_for(repo, merges)
        print(f"[{repo_name}] {len(merges)} merges -> {len(bs)} unique bases; "
              f"objects={objsize_mb(repo)}M  hydrating in chunks of {CHUNK} ...",
              flush=True)
        for i in range(0, len(bs), CHUNK):
            chunk = bs[i:i + CHUNK]
            ok = False
            for attempt in range(4):
                try:
                    # One grep over many bases => one batched promisor fetch of all
                    # their java blobs. Output discarded; we only want the fetch.
                    g(repo, *ROBUST, "grep", "-I", "-E", r"^\s*(package|import)\s",
                      *chunk, "--", "*.java", timeout=1800)
                    ok = True
                    break
                except subprocess.TimeoutExpired:
                    print(f"    chunk {i//CHUNK+1} timeout, retry {attempt+1}", flush=True)
            print(f"    chunk {i//CHUNK+1}/{(len(bs)+CHUNK-1)//CHUNK} "
                  f"({'ok' if ok else 'GAVE UP'})  objects={objsize_mb(repo)}M", flush=True)
        print(f"[{repo_name}] done, objects={objsize_mb(repo)}M", flush=True)

    print("HYDRATE FINISHED", flush=True)


if __name__ == "__main__":
    main()
