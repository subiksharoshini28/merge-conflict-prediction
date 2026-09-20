"""
Feature Extraction - Extract git-history features from repositories.

Usage:  python feature_extraction.py [repo1] [repo2] ...
        python feature_extraction.py --all
Output: features_extracted.csv
"""
import os
import sys
import warnings
import subprocess
import csv

warnings.filterwarnings("ignore")

DIVIDER = "=" * 80


def g(repo, *a):
    return subprocess.run(
        ["git", "-C", repo, *a],
        capture_output=True, text=True
    ).stdout.strip()


def extract_branch_features(repo, base, tip):
    commits = int(g(repo, "rev-list", "--count", base + ".." + tip) or 0)
    files = set(g(repo, "diff", "--name-only", base, tip).splitlines())
    authors = len(set(g(repo, "log", "--format=%ae", base + ".." + tip).splitlines()))
    stat = g(repo, "diff", "--shortstat", base, tip)
    ins = dels = 0
    for part in stat.split(", "):
        if "insertion" in part:
            ins = int(part.split()[0])
        elif "deletion" in part:
            dels = int(part.split()[0])
    return commits, files, authors, ins + dels


def extract_merge_features(repo, merge_commit, repo_name):
    parents = g(repo, "rev-list", "--parents", "-n", "1", merge_commit).split()[1:]
    if len(parents) != 2:
        return None
    p1, p2 = parents
    base = g(repo, "merge-base", p1, p2)
    if not base:
        return None

    result = subprocess.run(
        ["git", "-C", repo, "merge-tree", "--write-tree", p1, p2],
        capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        return None
    label = 1 if result.returncode == 1 else 0

    conflict_files = [
        line.split("Merge conflict in ", 1)[1]
        for line in result.stdout.splitlines()
        if "Merge conflict in " in line
    ]
    code_conflict = 1 if any(f.strip().endswith(".java") for f in conflict_files) else 0

    c1, f1, a1, ch1 = extract_branch_features(repo, base, p1)
    c2, f2, a2, ch2 = extract_branch_features(repo, base, p2)

    if c1 == 0 or c2 == 0:
        return None

    overlap = f1 & f2
    union = f1 | f2
    overlap_ratio = round(len(overlap) / len(union), 4) if union else 0

    return {
        "repo": repo_name,
        "merge": merge_commit,
        "commits_p1": c1,
        "commits_p2": c2,
        "files_p1": len(f1),
        "files_p2": len(f2),
        "overlap_files": len(overlap),
        "overlap_ratio": overlap_ratio,
        "authors_p1": a1,
        "authors_p2": a2,
        "churn_p1": ch1,
        "churn_p2": ch2,
        "conflict_files": ";".join(conflict_files),
        "code_conflict": code_conflict,
        "label": label
    }


def main():
    print(DIVIDER)
    print("  FEATURE EXTRACTION - Git History Features")
    print(DIVIDER)

    args = sys.argv[1:]
    if "--all" in args:
        repos_dir = "repos"
        if os.path.exists(repos_dir):
            repos = [
                os.path.join(repos_dir, d)
                for d in os.listdir(repos_dir)
                if os.path.isdir(os.path.join(repos_dir, d))
            ]
        else:
            print("  Error: repos/ directory not found")
            return
    else:
        repos = [r for r in args if r.startswith("repos/")]

    if not repos:
        print("\n  Usage:")
        print("    python feature_extraction.py repos/commons-lang repos/commons-io")
        print("    python feature_extraction.py --all")
        return

    print(f"\n  Repos to mine: {len(repos)}")
    for r in repos:
        print(f"    - {r}")

    output_file = "features_extracted.csv"
    fieldnames = [
        "repo", "merge", "commits_p1", "commits_p2", "files_p1", "files_p2",
        "overlap_files", "overlap_ratio", "authors_p1", "authors_p2",
        "churn_p1", "churn_p2", "conflict_files", "code_conflict", "label"
    ]

    total_scenarios = 0
    total_conflicts = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repo_path in repos:
            repo_name = os.path.basename(repo_path.rstrip("/\\"))
            print(f"\n  [{repo_name}] Mining merges...")

            merges = g(repo_path, "log", "--merges", "--format=%H").splitlines()
            print(f"    Found {len(merges)} merge commits")

            repo_scenarios = 0
            repo_conflicts = 0

            for merge_commit in merges:
                features = extract_merge_features(repo_path, merge_commit, repo_name)
                if features:
                    writer.writerow(features)
                    repo_scenarios += 1
                    if features["label"] == 1:
                        repo_conflicts += 1

            total_scenarios += repo_scenarios
            total_conflicts += repo_conflicts
            print(f"    Extracted: {repo_scenarios} scenarios, {repo_conflicts} conflicts")

    print(f"\n{'-' * 80}")
    print("  EXTRACTION COMPLETE")
    print(f"{'-' * 80}")
    print(f"  Output file      : {output_file}")
    print(f"  Total scenarios  : {total_scenarios}")
    print(f"  Total conflicts  : {total_conflicts}")
    if total_scenarios > 0:
        print(f"  Conflict rate    : {total_conflicts/total_scenarios*100:.1f}%")
    print(f"  Features per row : 10")
    print(DIVIDER)


if __name__ == "__main__":
    main()
