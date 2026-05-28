#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUESTIONS_PER_ONTOLOGY="${QUESTIONS_PER_ONTOLOGY:-200}"

# 3_2 / L2: class definition generation
cd "$SCRIPT_DIR/3_2_definition_generation"
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_2" \
  --concept-scope native \
  --max-questions "$QUESTIONS_PER_ONTOLOGY" \
  --no-imports \
  --no-warnings \
  --log info

# 3_3 / L3: class hierarchy construction
cd "$SCRIPT_DIR/3_3_hierarchy_construction"
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_3" \
  --concept-scope native \
  --log info

# 3_4 / L4: property relation construction
cd "$SCRIPT_DIR/3_4_property_relation_construction"
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_4" \
  --concept-scope native \
  --no-imports \
  --no-warnings \
  --log info

# 3_5 / L5: constraint construction
cd "$SCRIPT_DIR/3_5_constraint_construction"
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_5" \
  --concept-scope native \
  --no-imports \
  --no-warnings \
  --log info
