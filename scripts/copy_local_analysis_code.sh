#!/usr/bin/env bash
set -euo pipefail
src="$HOME/projects/stellar"
dst="$(cd "$(dirname "$0")/.." && pwd)/scripts/reproduction"
mkdir -p "$dst"
for file in run_stellar_preprocessed.py evaluate_masked_graph_250.py \
            analyze_neighborhood_effect.py analyze_homophily_by_class.py \
            analyze_multiseed_classes.py analyze_stellar.py; do
  if [[ -f "$src/$file" ]]; then
    cp "$src/$file" "$dst/$file"
    echo "Included locally authored code: $file"
  else
    echo "Missing, skipped: $src/$file"
  fi
done
cat <<EOF

Please review scripts/reproduction for local paths, usernames, and private data
before committing them to a public GitHub repository.
Do NOT copy the original CODEX pickle, model weights, predictions or raw lab files.
EOF
