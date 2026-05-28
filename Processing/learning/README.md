# Learning Task Pipeline

The main Learning suite is single-ontology construction and now has five tasks:

- `3_1` / `L1`: Ontology Term Extraction from Text
- `3_2` / `L2`: Class Definition Generation
- `3_3` / `L3`: Class Hierarchy Construction
- `3_4` / `L4`: Property Relation Construction
- `3_5` / `L5`: Constraint Construction

Run `run_learning_non_llm.sh` to generate non-LLM Learning tasks `L2` through `L5`.
Run `run_learning_llm.sh` to generate `L1` with DeepSeek surface paraphrasing.
By default it uses `PARAPHRASE_PROVIDER=deepseek`, `DEEPSEEK_MODEL=deepseek-chat`, `QUESTIONS_PER_ONTOLOGY=30`, and `PARAPHRASE_WORKERS=4`.
Run `run_learning.sh` only when you explicitly want the deterministic full Learning suite in one pass.
Each generator writes ontology-local JSON for traceability and CSV rows for the existing benchmark combine/export path.

For DeepSeek, set `DEEPSEEK_API_KEY` on the server before running `run_learning_llm.sh`.
Optional overrides are `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`.

Ontology alignment is no longer part of the main 15-task benchmark. It is kept as auxiliary task `A1` and can be generated with `run_auxiliary_alignment.sh` into `bench/bench_aux_alignment`.
