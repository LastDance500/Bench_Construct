#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK_DIR="${REPO_ROOT}/.r5_generation.lock"

QUESTIONS_PER_ONTOLOGY="${QUESTIONS_PER_ONTOLOGY:-100}"
R5_REASONER_TIMEOUT_SECONDS="${R5_REASONER_TIMEOUT_SECONDS:-300}"
R5_FILE_TIMEOUT_SECONDS="${R5_FILE_TIMEOUT_SECONDS:-600}"
JAVA_MEMORY="${JAVA_MEMORY:-2g}"
export JAVA_MEMORY

cleanup() {
  rm -rf "$LOCK_DIR"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "R5 appears to be running already: $LOCK_DIR"
  echo "If this is stale, remove it after confirming no R5/java process is running."
  exit 1
fi
trap cleanup EXIT

mkdir -p "$REPO_ROOT/logs"
cd "$SCRIPT_DIR/2_5_description_logic_reasoning"

echo "Running R5 safely: QUESTIONS_PER_ONTOLOGY=$QUESTIONS_PER_ONTOLOGY, R5_REASONER_TIMEOUT_SECONDS=$R5_REASONER_TIMEOUT_SECONDS, R5_FILE_TIMEOUT_SECONDS=$R5_FILE_TIMEOUT_SECONDS, JAVA_MEMORY=$JAVA_MEMORY"

exec nice -n 10 python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_5" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --reasoner-timeout-seconds "$R5_REASONER_TIMEOUT_SECONDS" \
  --file-timeout-seconds "$R5_FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info
