"""
Exploratory Data Analysis of the mined conflict dataset.

Answers, with real numbers: how big is it, how imbalanced, how does each repo
contribute, which features separate conflicts from clean merges, and - the key
motivating statistic - how many conflicts share NO file (the case flat overlap
features are blind to and the dependency graph is meant to catch).

Offline. Usage:  python eda.py [all_graph.csv | all.csv]
"""
import sys, os
import numpy as np
import pandas as pd

GIT = ["commits_p1", "commits_p2", "files_p1", "files_p2", "overlap_files",
       "overlap_ratio", "authors_p1", "authors_p2", "churn_p1", "churn_p2"]
GRAPH = ["g_nodes", "g_edges", "g_changed_p1", "g_changed_p2", "g_fanin_p1",
         "g_fanout_p1", "g_fanin_p2", "g_fanout_p2", "cross_edges",
         "shared_neighbors", "min_graph_distance", "shared_modules"]


def bar(x, xmax, width=24):
    return "#" * max(0, int(round(x / xmax * width))) if xmax else ""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "all_graph.csv" if os.path.exists("all_graph.csv") else "all.csv")
    df = pd.read_csv(path)
    feats = GIT + [c for c in GRAPH if c in df.columns]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    n, pos = len(df), int(df["label"].sum())

    print("=" * 68)
    print(f"DATASET EDA  |  {path}")
    print("=" * 68)
    print(f"scenarios (rows) ....... {n}")
    print(f"conflicts (label=1) .... {pos}  ({pos/n:.1%})")
    print(f"clean     (label=0) .... {n-pos}  ({(n-pos)/n:.1%})")
    print(f"imbalance ratio ........ 1 conflict : {(n-pos)/pos:.1f} clean")
    print(f"feature columns ........ {len(feats)} ({'git+graph' if len(feats)>len(GIT) else 'git only'})")

    # ---- per repository ----
    print("\n### Contribution & conflict rate by repository ###")
    print(f"{'repo':<24}{'scenarios':>10}{'conflicts':>11}{'rate':>8}")
    g = df.groupby("repo")["label"].agg(["count", "sum"]).sort_values("count", ascending=False)
    for repo, r in g.iterrows():
        cnt, s = int(r["count"]), int(r["sum"])
        print(f"{repo:<24}{cnt:>10}{s:>11}{s/cnt:>7.1%}   {bar(s, g['sum'].max())}")

    # ---- the motivating statistic ----
    print("\n### THE KEY STATISTIC: conflicts that share NO common file ###")
    conf = df[df["label"] == 1]
    no_shared = int((conf["overlap_files"] == 0).sum())
    print(f"of {pos} conflicts, {no_shared} ({no_shared/pos:.0%}) modify NO file in common.")
    print("Flat overlap features cannot see these; they are the target of the")
    print("dependency-graph interaction features (cross_edges, min_graph_distance, ...).")

    # ---- feature separation clean vs conflict ----
    print("\n### Feature signal: median value, clean vs conflict (+ correlation w/ label) ###")
    print(f"{'feature':<18}{'median clean':>13}{'median confl':>13}{'corr(label)':>13}")
    rows = []
    for f in feats:
        mc = df.loc[df.label == 0, f].median()
        mk = df.loc[df.label == 1, f].median()
        c = np.corrcoef(df[f], df["label"])[0, 1]
        rows.append((f, mc, mk, 0.0 if np.isnan(c) else c))
    for f, mc, mk, c in sorted(rows, key=lambda t: abs(t[3]), reverse=True):
        print(f"{f:<18}{mc:>13.4g}{mk:>13.4g}{c:>13.3f}")

    print("\nReading: features are ranked by |correlation with the conflict label|.")
    print("These rank-order matches the model's permutation importances (see explain.py).")


if __name__ == "__main__":
    main()
