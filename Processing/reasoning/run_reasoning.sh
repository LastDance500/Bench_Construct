#!/usr/bin/env bash
set -euo pipefail

cd 2_1_relation_reasoning
# 2_1: class relation reasoning (implicit triples)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_1" \
  --concept-scope native \
  --no-imports \
  --no-warnings \
  --log info

cd ../2_2_property_inheritance
# 2_2: property inheritance (range, chains, existential)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_2" \
  --concept-scope native \
  --max-questions 0 \
  --no-imports \
  --no-warnings \
  --log info

cd ../2_3_instance_reasoning
# 2_3: instance reasoning (instance -> inferred classes)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_3" \
  --concept-scope native \
  --max-questions 0 \
  --no-imports \
  --no-warnings \
  --log info



cd ../2_4_swrl_rule_reasoning
# 2_4: SWRL-like rule reasoning (from subclass axioms)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_4" \
  --concept-scope native \
  --max-questions 500 \
  --no-imports \
  --no-warnings \
  --log info

cd ../2_5_complex_logic_reasoning
# 2_5: complex logic satisfiability puzzles
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_2_5" \
  --concept-scope native \
  --no-warnings \
  --log info

