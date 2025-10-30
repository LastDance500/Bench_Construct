#!/usr/bin/env bash
set -euo pipefail

# 3_1: class definition generation (open-ended)
cd 3_1_definition_generation
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_1" \
  --concept-scope native \
  --max-questions 100 \
  --no-imports \
  --no-warnings \
  --log info

# 3_2: hierarchy construction (classes + properties)
cd ../3_2_hierarchy_construction
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_2" \
  --concept-scope native \
  --log info

# 3_3: property-only construction (no subclassOf)
cd ../3_3_hierarchy_constructioin_with_property
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_3" \
  --concept-scope native \
  --log info

# 3_4: property constraints (domain, range, functional)
cd ../3_4_hierarchy_constructioin_with_constriant
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_3_4" \
  --concept-scope native \
  --log info

# 3_5: ontology alignment (OAEI-style alignments)
cd ../3_5_ontology_alignment
python task_generate.py \
  --owl-dir "../../../data/alignment/conference/ontologies" \
  --rdf-dir "../../../data/alignment/conference/alignments" \
  --output-csv "../../../bench/bench_3_5/conference_alignment.csv" \
  --task-label 3_5 \
  --domain Conference \
  --log info


