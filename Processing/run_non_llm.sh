#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/understanding/run_understanding.sh"
bash "$SCRIPT_DIR/reasoning/run_reasoning.sh"
bash "$SCRIPT_DIR/learning/run_learning_non_llm.sh"
