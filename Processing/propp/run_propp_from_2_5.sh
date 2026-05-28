#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROPP_INPUT="$ROOT_DIR/data/Arts_Media_Entertainment/Propp/root-ontology.owl"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_LEVEL="${LOG_LEVEL:-warning}"
NO_IMPORTS="${NO_IMPORTS:-1}"
SWRL_MAX_QUESTIONS="${SWRL_MAX_QUESTIONS:-100}"
CLEAN_OLD="${CLEAN_OLD:-0}"
SYNC_BENCH_PROPP="${SYNC_BENCH_PROPP:-1}"
BUILD_SEED="${BUILD_SEED:-42}"
MCQ_ONLY="${MCQ_ONLY:-0}"
# Comma-separated caps, e.g. "2_5=500,3_1=500,propp_v1=500"
TASK_CAPS="${TASK_CAPS:-1_1=1000,1_2=1000,1_3=300,1_4=1000,1_5=300,2_1=300,2_2=1000,2_3=1000,2_4=500,2_5=500,3_1=500,3_2=500,3_3=300,3_4=300,3_5=300,propp_v1=500,propp_v2=300}"

if [[ ! -f "$PROPP_INPUT" ]]; then
  echo "Missing ontology: $PROPP_INPUT" >&2
  exit 1
fi

NO_IMPORTS_ARGS=()
if [[ "$NO_IMPORTS" == "1" ]]; then
  NO_IMPORTS_ARGS+=(--no-imports)
fi

SYNC_BENCH_PROPP_ARGS=()
if [[ "$CLEAN_OLD" == "1" ]]; then
  SYNC_BENCH_PROPP_ARGS+=(--clean)
fi

run_py() {
  "$PYTHON_BIN" "$@"
}

run_step() {
  local title="$1"
  shift
  echo "==> $title"
  run_py "$@"
}

clean_dir() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f \( -name '*.json' -o -name '*.csv' \) -delete
  fi
}

task_output_dir() {
  local task="$1"
  echo "$ROOT_DIR/bench/bench_${task}/Arts_Media_Entertainment/Propp"
}

task_generated_dir() {
  local task="$1"
  echo "$(task_output_dir "$task")/root-ontology"
}

flatten_task_output() {
  local task="$1"
  local source_dir
  local target_dir
  source_dir="$(task_generated_dir "$task")"
  target_dir="$(task_output_dir "$task")"
  if [[ ! -d "$source_dir" ]]; then
    return
  fi
  find "$source_dir" -maxdepth 1 -type f \( -name '*.json' -o -name '*.csv' \) -exec mv {} "$target_dir"/ \;
  rmdir "$source_dir" 2>/dev/null || true
}

if [[ "$CLEAN_OLD" == "1" ]]; then
  echo "==> Cleaning previous Propp outputs from 2_5 onward"
  clean_dir "$(task_output_dir 2_5)"
  clean_dir "$(task_output_dir 3_1)"
  clean_dir "$(task_output_dir 3_2)"
  clean_dir "$(task_output_dir 3_3)"
  clean_dir "$(task_output_dir 3_4)"
  clean_dir "$(task_output_dir 3_5)"
  clean_dir "$ROOT_DIR/bench/bench_propp_special/Arts_Media_Entertainment/Propp"
  clean_dir "$ROOT_DIR/bench_propp"
  rm -f "$ROOT_DIR/final_bench/propp/propp_all_tasks_with_special.csv"
fi

mkdir -p "$ROOT_DIR/bench/bench_propp_special/Arts_Media_Entertainment/Propp"
mkdir -p "$ROOT_DIR/final_bench/propp"

run_step "2_5 description logic reasoning" \
  "$ROOT_DIR/Processing/reasoning/2_5_description_logic_reasoning/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 2_5)" \
  --log "$LOG_LEVEL"
flatten_task_output "2_5"

run_step "3_1 term extraction" \
  "$ROOT_DIR/Processing/learning/3_1_term_extraction_from_text/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 3_1)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --max-questions 100 \
  --log "$LOG_LEVEL"
flatten_task_output "3_1"

run_step "3_2 definition generation" \
  "$ROOT_DIR/Processing/learning/3_2_definition_generation/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 3_2)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --max-questions 100 \
  --task-label 3_2 \
  --log "$LOG_LEVEL"
flatten_task_output "3_2"

run_step "3_3 hierarchy construction" \
  "$ROOT_DIR/Processing/learning/3_3_hierarchy_construction/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 3_3)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "3_3"

run_step "3_4 property construction" \
  "$ROOT_DIR/Processing/learning/3_4_property_relation_construction/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 3_4)" \
  --log "$LOG_LEVEL"
flatten_task_output "3_4"

run_step "3_5 constraint construction" \
  "$ROOT_DIR/Processing/learning/3_5_constraint_construction/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 3_5)" \
  --log "$LOG_LEVEL"
flatten_task_output "3_5"

run_step "propp_v1 / propp_v2 special tasks" \
  "$ROOT_DIR/Processing/propp/generate_special_tasks.py" \
  --input "$PROPP_INPUT" \
  --output "$ROOT_DIR/bench/bench_propp_special/Arts_Media_Entertainment/Propp" \
  --log "$LOG_LEVEL"

if [[ "$SYNC_BENCH_PROPP" == "1" ]]; then
  run_step "sync bench_propp mirror" \
    "$ROOT_DIR/Processing/propp/sync_bench_propp.py" \
    --output "$ROOT_DIR/bench_propp" \
    "${SYNC_BENCH_PROPP_ARGS[@]}"
fi

BUILD_ARGS=(--seed "$BUILD_SEED")
if [[ -n "$TASK_CAPS" ]]; then
  IFS=',' read -r -a TASK_CAP_LIST <<< "$TASK_CAPS"
  for cap in "${TASK_CAP_LIST[@]}"; do
    trimmed="${cap// /}"
    if [[ -n "$trimmed" ]]; then
      BUILD_ARGS+=(--task-cap "$trimmed")
    fi
  done
fi
if [[ "$MCQ_ONLY" == "1" ]]; then
  BUILD_ARGS+=(--mcq-only)
fi

run_step "build final Propp dataset" "$ROOT_DIR/final_bench/propp/build_propp_dataset.py" "${BUILD_ARGS[@]}"

echo "Done."
echo "Final CSV: $ROOT_DIR/final_bench/propp/propp_all_tasks_with_special.csv"
