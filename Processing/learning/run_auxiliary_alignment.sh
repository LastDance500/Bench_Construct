#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/auxiliary_alignment"

python task_generate.py \
  --owl-dir "../../../data/alignment/conference/ontologies" \
  --rdf-dir "../../../data/alignment/conference/alignments" \
  --output-csv "../../../bench/bench_aux_alignment/conference_alignment.csv" \
  --task-label A1 \
  --domain Conference \
  --log info

python task_generate_freizeit.py \
  --owl-dir "../../../data/alignment/freizeit/ontologies" \
  --rdf-dir "../../../data/alignment/freizeit/alignments" \
  --output-csv "../../../bench/bench_aux_alignment/freizeit_alignment.csv" \
  --task-label A1 \
  --domain freizeit \
  --concept-scope native \
  --no-warnings \
  --log info

python task_generate_lebensmittel.py \
  --owl-dir "../../../data/alignment/lebensmittel/ontologies" \
  --rdf-dir "../../../data/alignment/lebensmittel/alignments" \
  --output-csv "../../../bench/bench_aux_alignment/lebensmittel_alignment.csv" \
  --task-label A1 \
  --domain lebensmittel \
  --concept-scope native \
  --no-warnings \
  --log info

python task_generate2.py \
  --owl-dir "../../../data/alignment/multifarm/ontologies" \
  --rdf-dir "../../../data/alignment/multifarm/alignments" \
  --output-csv "../../../bench/bench_aux_alignment/multifarm_alignment.csv" \
  --task-label A1 \
  --domain multifarm \
  --concept-scope native \
  --no-warnings \
  --log info
