#!/usr/bin/env bash
# Resume + complete Phase 3 when the connection is good.
# Run from D:/SM_PROJ :   bash finish_phase3.sh
#
# It is RESUMABLE: blobs already downloaded into each repo's .git persist, so
# hydrate.py only fetches what is still missing. Safe to re-run after any drop.
set -e
cd "$(dirname "$0")"

echo "=== STEP 1/3  hydrate blobless clones (network-bound; resumable) ==="
python hydrate.py all.csv

echo "=== STEP 2/3  extract dependency-graph features (local, fast) ==="
python graph_features.py --in all.csv --out all_graph.csv

echo "=== STEP 3/3  GIT vs GRAPH vs ALL comparison (the headline number) ==="
python train_graph.py all_graph.csv

echo ""
echo "DONE. If this completed, drop the leave-repo-out ALL-vs-GIT F1 / PR-AUC"
echo "into IDF section 8 (the remaining [EDIT])."
