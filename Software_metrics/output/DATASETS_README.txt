MERGE CONFLICT PREDICTOR — DATASETS
====================================

datasets/  — the datasets that matter
------------------------------------------------------------
all.csv              THE MAIN DATASET. 1495 merge scenarios, 149 conflicts, 18 repos.
                     One row per real two-parent merge. Columns:
                       repo, merge, commits_p1/p2, files_p1/p2, overlap_files,
                       overlap_ratio, authors_p1/p2, churn_p1/p2, conflict_files,
                       code_conflict, label   (label = 1 conflict, 0 clean)
                     This is what the model trains on and the dashboard shows.

complexity.csv       Software metrics per scenario: CBO, Cyclomatic Complexity,
                     Dependency Depth (for a subset of repos).
graph_semantic.csv   Dependency-graph features (cross_edges, min_graph_distance,
                     shared_neighbors, fan-in/out, ...) for a subset of repos.
semantic.csv         Compile-based SEMANTIC conflict labels (javac). 0 real semantic
                     conflicts across 199 merges — a documented finding.
seeded.csv           Controlled seeded experiment (60 rows) proving no-shared-file
                     semantic conflicts are real & detectable by the dependency graph.

backups/  — dataset versions as it grew
------------------------------------------------------------
all_v1_backup.csv    561 scenarios / 68 conflicts / 6 repos   (leave-repo-out F1 0.59)
all_v2_backup.csv    891 / 94 / 10                              (F1 0.64)
all_v3_backup.csv    1041 / 112 / 14                            (F1 0.60)
(current all.csv)    1495 / 149 / 18                            (F1 0.62)

intermediate/  — mining/build artifacts (safe to ignore or delete)
------------------------------------------------------------
part_a/part_b/new*/mina/new_repos   raw mining outputs, later merged into all.csv
all_graph*/all_hydrated*/all_3repos  Phase-3 graph-feature experiments

NOTE: the LIVE framework reads all.csv and model.pkl from D:\SM_PROJ (the parent
folder), NOT from here. These are organized COPIES for reference/submission.
Do not delete D:\SM_PROJ\all.csv or D:\SM_PROJ\model.pkl — the app needs them.
