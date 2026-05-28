#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ONTOLOGY = REPO_ROOT / "data" / "Health & Medicine" / "Alzheimer's Disease Ontology" / "ado.owl"


def run_command(args):
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def find_single_json(root: Path) -> Path:
    files = sorted(root.rglob("*.json"))
    if not files:
        raise AssertionError(f"No JSON output found under {root}")
    return files[0]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_mcq_schema(items):
    if not isinstance(items, list) or not items:
        raise AssertionError("Expected a non-empty list of MCQ items")
    sample = items[0]
    required = {"prompt", "options", "correct_answer", "meta"}
    if not required.issubset(sample):
        raise AssertionError(f"Missing MCQ keys: {required - set(sample)}")


def assert_learning_schema(items):
    if not isinstance(items, list) or not items:
        raise AssertionError("Expected a non-empty list of learning tasks")
    sample = items[0]
    required = {"task_description", "classes", "triples"}
    if not required.issubset(sample):
        raise AssertionError(f"Missing learning keys: {required - set(sample)}")


def main():
    if not SAMPLE_ONTOLOGY.exists():
        raise SystemExit(f"Missing ontology for smoke test: {SAMPLE_ONTOLOGY}")

    with tempfile.TemporaryDirectory(prefix="ontobench_smoke_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        understanding_out = tmp_root / "u14"
        run_command(
            [
                sys.executable,
                "Processing/understanding/1_1_class2definition/task_generate.py",
                "--input",
                str(SAMPLE_ONTOLOGY),
                "--output",
                str(understanding_out),
                "--concept-scope",
                "native",
                "--no-imports",
                "--no-warnings",
                "--log",
                "warning",
            ]
        )
        assert_mcq_schema(load_json(find_single_json(understanding_out)))

        reasoning_out = tmp_root / "r23"
        run_command(
            [
                sys.executable,
                "Processing/reasoning/2_1_relation_reasoning/task_generate.py",
                "--input",
                str(SAMPLE_ONTOLOGY),
                "--output",
                str(reasoning_out),
                "--concept-scope",
                "native",
                "--no-imports",
                "--no-warnings",
                "--log",
                "warning",
            ]
        )
        assert_mcq_schema(load_json(find_single_json(reasoning_out)))

        learning_out = tmp_root / "l32"
        run_command(
            [
                sys.executable,
                "Processing/learning/3_3_hierarchy_construction/task_generate.py",
                "--input",
                str(SAMPLE_ONTOLOGY),
                "--output",
                str(learning_out),
                "--concept-scope",
                "native",
                "--no-imports",
                "--no-warnings",
                "--log",
                "warning",
            ]
        )
        assert_learning_schema(load_json(find_single_json(learning_out)))

    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
