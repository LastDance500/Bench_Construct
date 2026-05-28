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
MCQ_ONLY="${MCQ_ONLY:-1}"
RUN_DEEPSEEK_FINAL="${RUN_DEEPSEEK_FINAL:-1}"
DEEPSEEK_WORKERS="${DEEPSEEK_WORKERS:-2}"
DEEPSEEK_RETRIES="${DEEPSEEK_RETRIES:-3}"
DEEPSEEK_JUDGE_ALL="${DEEPSEEK_JUDGE_ALL:-0}"
# Comma-separated caps, e.g. "1_1=300,1_2=850,2_2=200".
# 2_5 is disabled for Propp by the final quality filters because the generated
# satisfiability expressions are too prone to domain/range/cardinality errors.
TASK_CAPS="${TASK_CAPS:-1_1=300,1_2=900,1_3=0,1_4=950,1_5=0,2_1=150,2_2=0,2_3=950,2_4=800,2_5=0,3_1=250,3_2=250,3_3=300,3_4=250,3_5=0,propp_v1=500,propp_v1_reverse=500,propp_v2=250,propp_v2_reverse=250}"

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

PROPP_BENCH_DIR="$ROOT_DIR/bench"
PROPP_POST_DIR="$ROOT_DIR/bench/bench_propp_special/Arts_Media_Entertainment/Propp"

task_output_dir() {
  local task="$1"
  echo "$PROPP_BENCH_DIR/bench_${task}/Arts_Media_Entertainment/Propp"
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

clean_dir() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f \( -name '*.json' -o -name '*.csv' \) -delete
  fi
}

if [[ "$CLEAN_OLD" == "1" ]]; then
  echo "==> Cleaning previous Propp outputs"
  clean_dir "$(task_output_dir 1_1)"
  clean_dir "$(task_output_dir 1_2)"
  clean_dir "$(task_output_dir 1_3)"
  clean_dir "$(task_output_dir 1_4)"
  clean_dir "$(task_output_dir 1_5)"
  clean_dir "$(task_output_dir 2_1)"
  clean_dir "$(task_output_dir 2_2)"
  clean_dir "$(task_output_dir 2_3)"
  clean_dir "$(task_output_dir 2_4)"
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

run_step "1_1 class2definition" \
  "$ROOT_DIR/Processing/understanding/1_1_class2definition/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 1_1)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "1_1"

run_step "1_2 class2class" \
  "$ROOT_DIR/Processing/understanding/1_2_class2class/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 1_2)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "1_2"

run_step "1_3 property semantics" \
  "$ROOT_DIR/Processing/understanding/1_3_property_semantics/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 1_3)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "1_3"

run_step "1_4 instance2class" \
  "$ROOT_DIR/Processing/understanding/1_4_instance2class/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 1_4)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "1_4"

run_step "1_5 instance2definition" \
  "$ROOT_DIR/Processing/understanding/1_5_instance_description/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 1_5)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "1_5"

run_step "2_1 relation reasoning" \
  "$ROOT_DIR/Processing/reasoning/2_1_relation_reasoning/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 2_1)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "2_1"

run_step "2_2 property constraint reasoning" \
  "$ROOT_DIR/Processing/reasoning/2_2_property_constraint_reasoning/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 2_2)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "2_2"

run_step "2_3 instance reasoning" \
  "$ROOT_DIR/Processing/reasoning/2_3_instance_reasoning/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 2_3)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --log "$LOG_LEVEL"
flatten_task_output "2_3"

run_step "2_4 swrl reasoning" \
  "$ROOT_DIR/Processing/reasoning/2_4_swrl_rule_reasoning/task_generate.py" \
  --input "$PROPP_INPUT" \
  --output "$(task_output_dir 2_4)" \
  "${NO_IMPORTS_ARGS[@]}" \
  --max-questions "$SWRL_MAX_QUESTIONS" \
  --log "$LOG_LEVEL"
flatten_task_output "2_4"

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

FINAL_INPUT="$ROOT_DIR/final_bench/propp/propp_all_tasks_with_special.csv"
FINAL_OUTPUT="$ROOT_DIR/final_bench/propp/propp.csv"
FINAL_REPORT="$ROOT_DIR/final_bench/propp/propp.final.report.csv"

if [[ "$RUN_DEEPSEEK_FINAL" == "1" ]]; then
  DEEPSEEK_ARGS=(
    --input "$FINAL_INPUT"
    --output "$FINAL_OUTPUT"
    --report "$FINAL_REPORT"
    --workers "$DEEPSEEK_WORKERS"
    --retries "$DEEPSEEK_RETRIES"
  )
  if [[ "$DEEPSEEK_JUDGE_ALL" == "1" ]]; then
    DEEPSEEK_ARGS+=(--judge-all)
  fi
  run_step "DeepSeek final repair and judge" \
    "$ROOT_DIR/final_bench/propp/run_perfect_pipeline.py" \
    "${DEEPSEEK_ARGS[@]}"
else
  cp "$FINAL_INPUT" "$FINAL_OUTPUT"
fi

run_step "final Propp quality audit" \
  "$ROOT_DIR/final_bench/propp/audit_propp_quality.py" \
  "$FINAL_OUTPUT" \
  --fail-on-issues

"$PYTHON_BIN" - "$FINAL_OUTPUT" "$ROOT_DIR/final_bench/propp/propp.dedup.csv" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
seen = set()
deduped = []
for row in rows:
    key = (row.get("question", ""), row.get("options", ""), row.get("answer", ""))
    if key in seen:
        continue
    seen.add(key)
    deduped.append(row)
with target.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
    writer.writeheader()
    writer.writerows(deduped)
print(f"Deduped CSV: {target} ({len(deduped)} rows)")
PY

echo "Done."
echo "Final CSV: $FINAL_OUTPUT"
echo "Deduped CSV: $ROOT_DIR/final_bench/propp/propp.dedup.csv"
