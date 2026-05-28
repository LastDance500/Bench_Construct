import json
import logging
import os
import signal
import warnings
from contextlib import contextmanager
from pathlib import Path
from collections import OrderedDict
from typing import Iterable, Optional

from owlready2 import set_log_level


DEFAULT_EXTENSIONS = (".owl", ".rdf", ".rdfs", ".ttl")


class FileProcessingTimeout(RuntimeError):
    pass


@contextmanager
def file_timeout(seconds: Optional[int]):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(_signum, _frame):
        raise FileProcessingTimeout(f"timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def slugify_for_windows(name: str) -> str:
    safe = []
    prev_us = False
    for ch in name:
        if ch.isalnum() or ch in ("-", "."):
            safe.append(ch)
            prev_us = False
        else:
            if not prev_us:
                safe.append("_")
            prev_us = True
    result = "".join(safe).strip("_")
    return result or "unnamed"


def configure_logging(level_name: str, log_file: Optional[str] = None, fmt: str = "%(asctime)s %(levelname)s: %(message)s") -> int:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, "w", "utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return level


def suppress_library_noise(enabled: bool) -> None:
    if not enabled:
        return
    try:
        set_log_level(0)
    except Exception:
        pass
    warnings.filterwarnings("ignore")
    for name in ("owlready2", "rdflib"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass


def resolve_onto_paths(raw_paths: Optional[Iterable[str]]) -> Optional[list[Path]]:
    if not raw_paths:
        return None
    return [Path(path) for path in raw_paths]


def discover_ontology_files(
    input_path: Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    exclude_dir_names: Iterable[str] = ("alignment",),
) -> tuple[list[Path], Path]:
    normalized_extensions = tuple(ext.lower() for ext in extensions)
    excluded = {name.lower() for name in exclude_dir_names}
    files: list[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in normalized_extensions:
        return [input_path], input_path.parent

    input_root = input_path
    for root, dirnames, filenames in os.walk(str(input_path)):
        dirnames[:] = [dirname for dirname in dirnames if dirname.lower() not in excluded]
        for filename in filenames:
            if filename.lower().endswith(normalized_extensions):
                files.append(Path(root) / filename)
    return files, input_root


def build_mirrored_output_dir(file_path: Path, input_root: Path, output_root: Path, sanitize_parts: bool = True) -> tuple[Path, str]:
    try:
        relative = file_path.relative_to(input_root)
    except Exception:
        relative = Path(file_path.name)

    relative_parts = list(Path(relative).parts)
    stem = Path(relative_parts[-1]).stem if relative_parts else file_path.stem
    if sanitize_parts:
        safe_parts = [slugify_for_windows(part) for part in relative_parts[:-1]]
        safe_stem = slugify_for_windows(stem)
    else:
        safe_parts = relative_parts[:-1]
        safe_stem = stem

    output_dir = output_root.joinpath(*safe_parts, safe_stem)
    return output_dir, safe_stem


def save_json(data, path: Path, description: str = "items", skip_empty: bool = False) -> bool:
    if skip_empty and not data:
        logging.info("No %s generated for %s, skipping save", description, path)
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)
    try:
        size = len(data)
    except TypeError:
        size = "unknown number of"
    logging.info("Saved %s %s to %s", size, description, path)
    return True


def empty_marker_path(out_dir: Path, task_prefix: str, safe_stem: str) -> Path:
    return out_dir / f"no_{task_prefix}_{safe_stem}.json"


def save_empty_marker(
    path: Path,
    *,
    source_file: Path,
    reason: str,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "source_file": str(source_file),
        "reason": reason,
        "generated_questions": 0,
    }
    if extra:
        payload.update(extra)
    save_json(payload, path, description="empty ontology marker")


def should_skip_existing(output_path: Path, empty_path: Optional[Path] = None, *extra_paths: Path) -> bool:
    if output_path.exists() and all(path.exists() for path in extra_paths):
        logging.info("Skip existing: %s", output_path)
        return True
    if empty_path is not None and empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return True
    return False


def question_subject_key(question: dict) -> str:
    meta = question.get("meta", {}) if isinstance(question, dict) else {}
    for field in ("subject_iri", "class_context_iri", "subject_label", "class_context_label"):
        value = meta.get(field)
        if value:
            return f"{field}:{value}"
    return f"prompt:{question.get('prompt', '') if isinstance(question, dict) else str(question)}"


def limit_questions_by_subject(
    questions: Iterable[dict],
    max_questions: Optional[int],
) -> list[dict]:
    """Round-robin questions by subject so caps preserve entity diversity."""
    buckets: OrderedDict[str, list[dict]] = OrderedDict()
    for question in questions:
        buckets.setdefault(question_subject_key(question), []).append(question)

    if not max_questions or max_questions <= 0:
        max_questions = sum(len(bucket) for bucket in buckets.values())

    selected: list[dict] = []
    depth = 0
    while len(selected) < max_questions:
        progressed = False
        for bucket in buckets.values():
            if depth < len(bucket):
                selected.append(bucket[depth])
                progressed = True
                if len(selected) >= max_questions:
                    break
        if not progressed:
            break
        depth += 1
    return selected
