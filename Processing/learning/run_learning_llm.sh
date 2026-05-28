#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAPHRASE_PROVIDER="${PARAPHRASE_PROVIDER:-deepseek}"
PARAPHRASE_MODEL="${DEEPSEEK_MODEL:-${OPENAI_MODEL:-deepseek-chat}}"
QUESTIONS_PER_ONTOLOGY="${QUESTIONS_PER_ONTOLOGY:-30}"
PARAPHRASE_WORKERS="${PARAPHRASE_WORKERS:-4}"

if [[ "$PARAPHRASE_PROVIDER" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required when PARAPHRASE_PROVIDER=deepseek." >&2
  exit 1
fi

if [[ "$PARAPHRASE_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required when PARAPHRASE_PROVIDER=openai." >&2
  exit 1
fi

# 3_1 / L1: ontology term extraction from text with LLM surface paraphrasing.
# Gold labels and IRIs are still derived from source ontology entities.
cd "$SCRIPT_DIR/3_1_term_extraction_from_text"
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_1" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --no-imports \
  --strict-ambiguity-check \
  --paraphrase-provider "$PARAPHRASE_PROVIDER" \
  --paraphrase-model "$PARAPHRASE_MODEL" \
  --paraphrase-retries 2 \
  --paraphrase-workers "$PARAPHRASE_WORKERS" \
  --paraphrase-failure-policy clean_fallback \
  --no-warnings \
  --log info
