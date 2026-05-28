#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUESTIONS_PER_ONTOLOGY="${QUESTIONS_PER_ONTOLOGY:-200}"
FILE_TIMEOUT_SECONDS="${FILE_TIMEOUT_SECONDS:-1800}"
U1_MAX_CANDIDATES="${U1_MAX_CANDIDATES:-3000}"
MAX_FILE_MB="${MAX_FILE_MB:-0}"

cd "$SCRIPT_DIR/1_1_class2definition"

# 1_1: class -> definition
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_1" \
  --concept-source ontology \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --max-candidates "$U1_MAX_CANDIDATES" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --max-file-mb "$MAX_FILE_MB" \
  --no-imports \
  --no-warnings \
  --log info

cd "$SCRIPT_DIR/1_2_class2class"

# 1_2: class -> class (relations)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_2" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info

cd "$SCRIPT_DIR/1_3_property_semantics"
# 1_3 / U3: property semantics -> explicit domain/range
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_3" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info

cd "$SCRIPT_DIR/1_4_instance2class"
# 1_4: instance -> class
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_4" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info

cd "$SCRIPT_DIR/1_5_instance_description"
# 1_5 / U5: instance -> description (instances with explicit descriptions)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_5" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info
