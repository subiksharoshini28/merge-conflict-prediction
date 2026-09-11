# Merge Conflict Predictor

Predict whether two git branches will produce a **merge conflict — before merging** —
with a conflict probability, a risk score (LOW / MEDIUM / HIGH), an explanation of *why*,
and a recommended action. Built on git-history features plus a code **dependency graph**
and software metrics (CBO, Cyclomatic Complexity, Dependency Depth), with explainable output.

> Inventors: Subiksha Roshini Durai Murugan, Tamizhiniyan K.

---

## 1. Requirements

- **Python 3.10+** with the packages in `requirements.txt`
- **git** on the PATH
- **JDK (javac)** — only for the semantic-conflict tools (`semantic_label.py`, `seeded_experiment.py`)

```bash
pip install -r requirements.txt
```

## 2. Quick start — run the framework

```bash
# 1) Train the model once (creates model.pkl from the mined dataset all.csv)
python conflict_predictor.py train

# 2) Predict for two real branches of a repo  <-- the product
python conflict_predictor.py predict <path-to-repo> <branchA> <branchB>

# 3) Demo on a real merge from the dataset (no branches needed)
python conflict_predictor.py demo
```

Example output:

```
 MERGE CONFLICT PREDICTION
 repo: netty | branch A: main | branch B: feature | merge base: c84a6c9697be
 Conflict probability : 0.96
 Risk score           : HIGH  [##########]
 Why (top drivers):
   - overlap_files = 5 (typical clean 0)
   - churn_p1 = 584964 (typical clean 27)
 Recommended action:
   -> Both branches edit the same file(s) - coordinate or merge the smaller branch first.
```

## 3. Rebuild the dataset from scratch (optional)

```bash
# clone some Java repos as blobless clones (small on disk)
git clone --filter=blob:none https://github.com/apache/commons-lang.git D:/commons-lang
# ... more repos ...

# mine every 2-parent merge -> labelled dataset all.csv (label = did it conflict?)
python mine_conflicts.py D:/commons-lang D:/netty D:/commons-io ...
```

## 4. Analysis & research tools

| Script | What it does |
|---|---|
| `conflict_predictor.py` | **Framework entry point** — train / predict / demo |
| `mine_conflicts.py` | Mine merges, label via `git merge-tree` → `all.csv` |
| `graph_features.py` | Build the file dependency graph, extract graph features |
| `complexity_features.py` | CBO, Cyclomatic Complexity, Dependency Depth |
| `train_baseline.py` | Model + metrics (precision/recall/F1/PR-AUC, leave-repo-out) |
| `explain.py` | Feature importance + per-merge "why" + risk banding |
| `eda.py` | Dataset statistics |
| `semantic_label.py` | Compile-based semantic (no-shared-file) conflict labelling (javac) |
| `seeded_experiment.py` | Controlled experiment proving no-shared-file conflicts (patent claim 3) |
| `train_graph.py` / `train_semantic.py` | Feature-set comparisons |

## 5. Key results (see `Dataset_and_Findings`)

- **Textual conflicts** — 561 scenarios / 68 conflicts across 6 repos; the model generalizes
  to an unseen repository at **leave-repo-out F1 ≈ 0.59** (PR-AUC ≈ 0.60 vs 0.12 base rate).
- **Semantic conflicts** — a controlled seeded experiment proves conflicts that share **no file**
  are real and detectable via the dependency graph where flat overlap features are blind.
  Real merged history contains ~0 (CI survivorship + atomic refactoring) — a documented finding.

## 6. Data files

| File | Contents |
|---|---|
| `all.csv` | Main labelled textual-conflict dataset (561 rows) |
| `graph_semantic.csv`, `complexity.csv` | Feature tables |
| `semantic.csv` | Compile-based semantic labels |
| `seeded.csv` | Controlled experiment results |
| `model.pkl` | Trained model (created by `train`) |

## Notes

- All features are computed **at the merge base** (point-in-time); merge-outcome fields are never
  used as inputs (no leakage).
- Graph/complexity analysis targets **Java**.
