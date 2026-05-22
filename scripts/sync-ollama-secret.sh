#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OLLAMA_API_KEY:-}" ]; then
  echo "Set OLLAMA_API_KEY in your shell, then re-run." >&2
  echo "  export OLLAMA_API_KEY='...'" >&2
  exit 1
fi

REPOS=(
  monte-carlo
  stock-sentiment-analysis
  ethereum-blocks
  energy-market-visualization
  raft-consensus
)

for repo in "${REPOS[@]}"; do
  echo "Setting OLLAMA_API_KEY on pdj555/${repo}..."
  gh secret set OLLAMA_API_KEY -R "pdj555/${repo}" --body "$OLLAMA_API_KEY"
done

echo "Done."
