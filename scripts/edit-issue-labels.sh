#!/usr/bin/env bash
#
# edit-issue-labels.sh — apply a comma-separated list of labels to an issue.
#
# Usage:
#   scripts/edit-issue-labels.sh <issue_number> <label1,label2,...>
#
# Used by the Claude triage workflow (.github/workflows/claude-triage.yml)
# via the /profit-triage slash command. Creates any missing labels on demand
# so triage never fails because a new severity/category was introduced.
#
# Requires: gh CLI authenticated via GITHUB_TOKEN.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <issue_number> <label1,label2,...>" >&2
  exit 2
fi

issue_number="$1"
labels_csv="$2"

if [[ -z "${labels_csv}" ]]; then
  echo "no labels provided; nothing to do" >&2
  exit 0
fi

# Split CSV into array, trimming whitespace around each label.
IFS=',' read -r -a labels <<< "${labels_csv}"

# Ensure every label exists before applying. `gh label create` is idempotent
# with --force, so repeated runs are safe.
for raw_label in "${labels[@]}"; do
  label="$(echo -n "${raw_label}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "${label}" ]] && continue

  if ! gh label list --limit 200 --json name --jq '.[].name' | grep -Fxq "${label}"; then
    # Pick a sensible default color by prefix; gh will ignore dupes.
    case "${label}" in
      p0-*)            color="b60205" ;;
      p1-*)            color="d93f0b" ;;
      p2-*)            color="fbca04" ;;
      p3-*)            color="c2e0c6" ;;
      noise)           color="cccccc" ;;
      bug)             color="d73a4a" ;;
      perf)            color="5319e7" ;;
      correctness)     color="b60205" ;;
      risk)            color="e99695" ;;
      cli)             color="0e8a16" ;;
      sdk)             color="1d76db" ;;
      docs)            color="0075ca" ;;
      question)        color="d876e3" ;;
      revenue-blocker) color="000000" ;;
      *)               color="ededed" ;;
    esac
    gh label create "${label}" --color "${color}" --force >/dev/null
  fi
done

# Apply all labels in one call. --add-label can be passed multiple times.
add_args=()
for raw_label in "${labels[@]}"; do
  label="$(echo -n "${raw_label}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "${label}" ]] && continue
  add_args+=(--add-label "${label}")
done

gh issue edit "${issue_number}" "${add_args[@]}"

echo "applied labels to #${issue_number}: ${labels_csv}"
