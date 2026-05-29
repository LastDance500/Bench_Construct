# Bench_Construct

Bench_Construct contains the reproducible construction pipeline for OntoURL.
The companion benchmark and inference repository is:
https://github.com/LastDance500/OntoURL

This repository is updated for OntoURL v1.1. The release contains 36,159
benchmark instances from 43 formal ontology resources across 8 domains and 15
tasks. The released `ontology` column uses 38 normalized ontology labels across
the 15 task splits because several Science resources are grouped under broader
labels.

## What This Repository Provides

- Ontology parsers and shared utilities in `Processing/common/`.
- Task generators for Understanding, Reasoning, and Learning.
- A lightweight L1 ontology term extraction generator that derives gold
  classes/properties from source ontology entities and optionally uses DeepSeek
  only for surface paraphrasing of the generated text.
- The final OntoURL v1.1 CSV export in `bench/v1.1/`.
- A small source ontology sample under `data/` for local smoke tests.

The full benchmark dataset is released at:
https://huggingface.co/datasets/XiaoZhang98/OntoURL

## Task Suite

| ID | Task | Format | Metric | Instances |
| --- | --- | --- | --- | ---: |
| U1 | Class Definition Understanding | MCQ | Accuracy | 3,000 |
| U2 | Class Relation Understanding | MCQ | Accuracy | 3,000 |
| U3 | Property Semantics Understanding | MCQ | Accuracy | 2,862 |
| U4 | Instance Class Understanding | MCQ | Accuracy | 3,116 |
| U5 | Instance Description Understanding | MCQ | Accuracy | 1,776 |
| R1 | Inferred Class Relation Reasoning | MCQ | Accuracy | 2,968 |
| R2 | Property Constraint Reasoning | MCQ | Accuracy | 2,814 |
| R3 | Inferred Instance Class Reasoning | MCQ | Accuracy | 2,415 |
| R4 | SWRL-based Rule Reasoning | MCQ | Accuracy | 3,000 |
| R5 | Description Logic Reasoning | True/False | Accuracy | 2,535 |
| L1 | Ontology Term Extraction | Generation | Entity-F1 | 1,799 |
| L2 | Class Definition Generation | Generation | BERTScore F1 | 3,000 |
| L3 | Class Hierarchy Construction | Generation | Triple-F1 | 2,997 |
| L4 | Property Relation Construction | Generation | Triple-F1 | 264 |
| L5 | Constraint Construction | Generation | Triple-F1 | 613 |

## Repository Layout

```text
Processing/common/           shared ontology loading, labeling, stats, and IO
Processing/understanding/    U1-U5 generators
Processing/reasoning/        R1-R5 generators
Processing/learning/         L1-L5 generators and auxiliary alignment code
Processing/tests/            smoke test for the construction scripts
bench/v1.1/                  final OntoURL v1.1 CSV splits and metadata
data/                        small source ontology sample for local tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test

Run the construction smoke test before regenerating a full task suite:

```bash
python Processing/tests/smoke_test.py
```

## Regenerating Tasks

The scripts expect source ontology files under `data/` and write generated task
files under `bench/`.

Understanding tasks:

```bash
cd Processing
bash understanding/run_understanding.sh
```

Reasoning tasks:

```bash
cd Processing
bash reasoning/run_reasoning.sh
```

Learning tasks L2-L5 are deterministic and do not call an LLM:

```bash
cd Processing
bash learning/run_learning_non_llm.sh
```

Learning task L1 uses deterministic ontology-fragment verbalization plus
optional DeepSeek surface paraphrasing. The gold labels and IRIs are always
derived from ontology entities, not from the paraphraser.

```bash
cd Processing
export DEEPSEEK_API_KEY=...
bash learning/run_learning_llm.sh
```

To avoid paraphrasing and generate template-only L1 text, call the generator
with `--paraphrase-provider none`.

## L1 Construction Summary

For each ontology, L1 samples a local fragment around a focal class. The
fragment can include direct parents, children, and connected properties. The
generator verbalizes the fragment into a short text and records the source
classes/properties as the gold answer. Validation rejects samples when required
labels are missing, labels are ambiguous, text is too short/long, or generated
text contains obvious paraphrasing artifacts.

DeepSeek, when enabled, rewrites only the surface form of the paragraph. It is
not used to generate gold labels, ontology entities, or task answers.

## Citation

```bibtex
@article{zhang2025ontourl,
  title={OntoURL: A Benchmark for Evaluating Large Language Models on Symbolic Ontological Understanding, Reasoning and Learning},
  author={Zhang, Xiao and Lai, Huiyuan and Meng, Qianru and Bos, Johan},
  journal={arXiv preprint arXiv:2505.11031},
  year={2025}
}
```

## License

Code is released under the MIT License. Generated benchmark data are released
under CC BY 4.0 where permitted by source ontology licenses; ontology-specific
license metadata are provided in the dataset card.
