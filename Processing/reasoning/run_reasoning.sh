#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUESTIONS_PER_ONTOLOGY="${QUESTIONS_PER_ONTOLOGY:-200}"
FILE_TIMEOUT_SECONDS="${FILE_TIMEOUT_SECONDS:-1800}"
RUN_R5="${RUN_R5:-0}"

cd "$SCRIPT_DIR/2_1_relation_reasoning"
# 2_1: class relation reasoning (implicit triples)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_1" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info

cd "$SCRIPT_DIR/2_2_property_constraint_reasoning"
# 2_2 / R2: property constraint reasoning (domain/range inheritance, chains, existential)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_2" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info

cd "$SCRIPT_DIR/2_3_instance_reasoning"
# 2_3: instance reasoning (instance -> inferred classes)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_3" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --file-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
  --no-imports \
  --no-warnings \
  --log info
cd "$SCRIPT_DIR/2_4_swrl_rule_reasoning"
# 2_4: SWRL-like rule reasoning (from subclass axioms)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_4" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --no-imports \
  --no-warnings \
  --log info

if [[ "$RUN_R5" == "1" ]]; then
  cd "$SCRIPT_DIR/2_5_description_logic_reasoning"
  # 2_5 / R5: bounded description logic satisfiability puzzles.
  # R5 launches a Java reasoner, so keep it opt-in for shared servers.
  python task_generate.py \
    --input "../../../data" \
    --output "../../../bench/bench_2_5" \
    --concept-scope native \
    --max-questions "$QUESTIONS_PER_ONTOLOGY" \
    --reasoner-timeout-seconds "$FILE_TIMEOUT_SECONDS" \
    --no-imports \
    --no-warnings \
    --log info
else
  echo "Skipping R5 by default. Run Processing/reasoning/run_r5_safe.sh for the reasoner-heavy task."
fi
