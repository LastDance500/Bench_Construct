import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TASK_SOURCES = {
    "1_1": ROOT / "bench" / "bench_1_1" / "Arts_Media_Entertainment" / "Propp",
    "1_2": ROOT / "bench" / "bench_1_2" / "Arts_Media_Entertainment" / "Propp",
    "1_3": ROOT / "bench" / "bench_1_3" / "Arts_Media_Entertainment" / "Propp",
    "1_4": ROOT / "bench" / "bench_1_4" / "Arts_Media_Entertainment" / "Propp",
    "1_5": ROOT / "bench" / "bench_1_5" / "Arts_Media_Entertainment" / "Propp",
    "2_1": ROOT / "bench" / "bench_2_1" / "Arts_Media_Entertainment" / "Propp",
    "2_2": ROOT / "bench" / "bench_2_2" / "Arts_Media_Entertainment" / "Propp",
    "2_3": ROOT / "bench" / "bench_2_3" / "Arts_Media_Entertainment" / "Propp",
    "2_4": ROOT / "bench" / "bench_2_4" / "Arts_Media_Entertainment" / "Propp",
    "2_5": ROOT / "bench" / "bench_2_5" / "Arts_Media_Entertainment" / "Propp",
    "3_1": ROOT / "bench" / "bench_3_1" / "Arts_Media_Entertainment" / "Propp",
    "3_2": ROOT / "bench" / "bench_3_2" / "Arts_Media_Entertainment" / "Propp",
    "3_3": ROOT / "bench" / "bench_3_3" / "Arts_Media_Entertainment" / "Propp",
    "3_4": ROOT / "bench" / "bench_3_4" / "Arts_Media_Entertainment" / "Propp",
    "3_5": ROOT / "bench" / "bench_3_5" / "Arts_Media_Entertainment" / "Propp",
    "special": ROOT / "bench" / "bench_propp_special" / "Arts_Media_Entertainment" / "Propp",
}


def normalize_name(name: str) -> str:
    normalized = name.replace(".owl.json", ".json").replace(".owl.csv", ".csv")
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    return normalized


def sync_task(task_name: str, source_dir: Path, target_root: Path) -> int:
    if not source_dir.exists():
        return 0
    target_dir = target_root / task_name / "propp"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".csv"}:
            continue
        target_path = target_dir / normalize_name(path.name)
        shutil.copy2(path, target_path)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync standard Propp outputs into a normalized bench_propp mirror.")
    parser.add_argument("--output", default=str(ROOT / "bench_propp"), help="Output mirror root.")
    parser.add_argument("--clean", action="store_true", help="Remove existing mirrored JSON/CSV files before syncing.")
    args = parser.parse_args()

    output_root = Path(args.output)
    if args.clean and output_root.exists():
        for path in output_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
                path.unlink()

    total = 0
    for task_name, source_dir in TASK_SOURCES.items():
        total += sync_task(task_name, source_dir, output_root)

    print(f"Synced {total} files into {output_root}")


if __name__ == "__main__":
    main()
