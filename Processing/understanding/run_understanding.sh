#!/usr/bin/env bash

cd 1_1_class2definition

# 1_1: class -> definition
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_1" \
  --concept-source ontology \
  --concept-scope native \
  --no-imports \
  --log info \
  --no-warnings

cd ../1_2_class2class

# 1_2: class -> class (relations)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_2" \
  --concept-scope native \
  --no-imports \
  --log info \
  --no-warnings

cd ../1_3_property2domain
# 1_3: property -> domain/range
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_3" \
  --concept-scope native \
  --max-questions 0 \
  --no-imports \
  --log info \
  --no-warnings

cd ../1_4_instance2class
# 1_4: instance -> class
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_4" \
  --concept-scope native \
  --no-imports \
  --log info \
  --no-warnings

cd ../1_5_instance2defintion
# 1_5: instance -> definition (instances with explicit definitions)
python task_generate.py \
  --input "../../../data" \
  --output "../../../bench/bench_1_5" \
  --concept-scope native \
  --no-imports \
  --log info \
  --no-warnings

