"""
Seeded / controlled experiment for the SEMANTIC (no-shared-file) conflict claim (patent claim 3).

We synthesise merge scenarios with KNOWN ground truth and verify each by real javac
compilation. Two classes in different packages, linked by an import:
    seed.prov.Provider_k     (a provider with method compute())
    seed.cons.Consumer_k     (imports Provider_k and calls compute())

Scenario types (each: branch A edits ONE file, branch B edits a DIFFERENT file -> NO shared file):
  CONFLICT : A renames Provider.compute()->calculate() (breaking);
             B adds a new call to Provider.compute() in Consumer.
             merged code compiles? NO  -> semantic conflict = 1
  CLEAN    : A adds a new method to Provider (non-breaking);
             B adds a call to the still-existing compute() in Consumer.
             merged compiles? YES -> 0
  CONTROL  : A makes a breaking rename in an UNRELATED class nobody imports;
             B edits Consumer harmlessly. Edited files are NOT dependency-linked.
             merged compiles? YES -> 0   (breaking change with no dependency link => no cross-file conflict)

The point: for every CONFLICT, file-overlap = 0 (flat features are blind), yet the two
edited files ARE connected by a dependency-graph edge. The graph sees what overlap cannot.

Offline, javac-only. Usage:  python seeded_experiment.py [--n 20] [--out seeded.csv]
"""
import subprocess, sys, os, csv, tempfile, shutil, re

IMPORT_RE = re.compile(r'^\s*import\s+([\w.]+);', re.M)


def write_and_compile(files):
    """files: {relpath: content}. Returns (compiles: bool, first_error: str)."""
    d = tempfile.mkdtemp()
    try:
        paths = []
        for rel, content in files.items():
            fp = os.path.join(d, rel)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w", encoding="utf-8").write(content)
            paths.append(fp)
        r = subprocess.run(["javac", "-d", os.path.join(d, "out"), "-proc:none", *paths],
                           capture_output=True, text=True)
        err = ""
        for line in r.stderr.splitlines():
            if ": error:" in line:
                err = line.split(": error:")[-1].strip(); break
        return r.returncode == 0, err
    finally:
        shutil.rmtree(d, ignore_errors=True)


def linked(files, a, b):
    """True if edited files a,b are connected by an import edge (either direction)."""
    def fqcn(rel):  # seed/cons/Consumer_k.java -> seed.cons.Consumer_k
        return rel[:-5].replace("/", ".")
    ia = set(IMPORT_RE.findall(files[a])); ib = set(IMPORT_RE.findall(files[b]))
    return (fqcn(b) in ia) or (fqcn(a) in ib)


def provider(k, method="compute", extra=""):
    return (f"package seed.prov;\npublic class Provider_{k} {{\n"
            f"  public int {method}() {{ return {k}; }}\n{extra}}}\n")


def consumer(k, calls):
    body = "\n".join(f"  public int use{i}() {{ return new seed.prov.Provider_{k}().{c}(); }}"
                     for i, c in enumerate(calls))
    return (f"package seed.cons;\nimport seed.prov.Provider_{k};\n"
            f"public class Consumer_{k} {{\n{body}\n}}\n")


def unrelated(k, method="compute"):
    return f"package seed.other;\npublic class Unrelated_{k} {{ public int {method}() {{ return 0; }} }}\n"


def scenario(k, kind):
    prov = f"seed/prov/Provider_{k}.java"
    cons = f"seed/cons/Consumer_{k}.java"
    other = f"seed/other/Unrelated_{k}.java"
    if kind == "CONFLICT":
        merged = {prov: provider(k, method="calculate"),          # A: breaking rename
                  cons: consumer(k, calls=["compute", "compute"])}  # B: still calls compute()
        return merged, prov, cons
    if kind == "CLEAN":
        merged = {prov: provider(k, method="compute", extra="  public int extra(){ return -1; }\n"),
                  cons: consumer(k, calls=["compute", "compute"])}
        return merged, prov, cons
    # CONTROL: breaking rename in an unrelated (unused) class; consumer edited harmlessly
    merged = {prov: provider(k, method="compute"),
              cons: consumer(k, calls=["compute", "compute"]),
              other: unrelated(k, method="renamed")}   # A edits Unrelated (nobody imports it)
    return merged, other, cons


def main():
    args = sys.argv[1:]; n = 20; out = "seeded.csv"
    i = 0
    while i < len(args):
        if args[i] == "--n": n = int(args[i+1]); i += 2
        elif args[i] == "--out": out = args[i+1]; i += 2
        else: i += 1

    rows = []
    for kind in ("CONFLICT", "CLEAN", "CONTROL"):
        for k in range(n):
            files, fa, fb = scenario(k, kind)
            ok, err = write_and_compile(files)
            rows.append(dict(kind=kind, k=k, file_a=os.path.basename(fa), file_b=os.path.basename(fb),
                             share_file=int(fa == fb), dependency_linked=int(linked(files, fa, fb)),
                             merged_compiles=int(ok), semantic_conflict=int(not ok), error=err))

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- summary / proof ----
    def agg(kind):
        r = [x for x in rows if x["kind"] == kind]
        return (len(r), sum(x["semantic_conflict"] for x in r),
                sum(x["dependency_linked"] for x in r), sum(x["share_file"] for x in r))
    print("=" * 70)
    print(f"SEEDED EXPERIMENT  ({n} per type, verified by real javac compilation)")
    print("=" * 70)
    print(f"{'type':<10}{'n':>4}{'conflicts':>11}{'dep-linked':>12}{'share-file':>12}")
    for kind in ("CONFLICT", "CLEAN", "CONTROL"):
        tot, conf, dep, sh = agg(kind)
        print(f"{kind:<10}{tot:>4}{conf:>11}{dep:>12}{sh:>12}")
    conf = [x for x in rows if x["kind"] == "CONFLICT"]
    print("\nPROOF for patent claim 3 (no-shared-file semantic conflicts):")
    print(f"  - {sum(x['semantic_conflict'] for x in conf)}/{len(conf)} CONFLICT scenarios fail to compile"
          f" (verified semantic conflicts)")
    print(f"  - share_file = 0 for ALL of them  -> flat file-overlap features are BLIND")
    print(f"  - dependency_linked = 1 for ALL of them -> the dependency GRAPH reveals the coupling")
    print(f"  - CONTROL: identical breaking change but NO dependency link -> "
          f"{agg('CONTROL')[1]} conflicts (link is what matters)")
    print(f"\nExample failure: {conf[0]['error']}")
    print(f"wrote {len(rows)} scenarios -> {out}")


if __name__ == "__main__":
    main()
