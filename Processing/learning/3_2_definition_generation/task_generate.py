import json
import csv
import os
import random
import logging
import argparse
import warnings
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import World, owl, ThingClass, onto_path, set_log_level


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    BaseOntologyLoader,
    build_mirrored_output_dir,
    class_depth,
    class_stats,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    get_definition as shared_get_definition,
    get_label as shared_get_label,
    global_class_metrics,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    selection_weight,
    siblings as class_siblings,
    slugify_for_windows,
    suppress_library_noise,
)

# Global caches for definitions and labels
definition_cache = {}
label_cache = {}

def get_definition(entity):
    try:
        key = str(entity.iri)
        if key in definition_cache:
            return definition_cache[key]
        definition = shared_get_definition(entity)
        definition_cache[key] = definition
        return definition
    except Exception as e:
        logging.warning(f"Error retrieving definition for {getattr(entity, 'name', 'unknown')}: {e}")
        return "No definition provided."

def get_label(entity):
    try:
        key = str(entity.iri)
        if key in label_cache:
            return label_cache[key]
        result = shared_get_label(entity)
        label_cache[key] = result
        return result
    except Exception as e:
        logging.warning(f"Error retrieving label for {getattr(entity, 'name', 'unknown')}: {e}")
        return str(entity.name)


def normalize_definition_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").replace("-", " ")).strip().lower()


def is_low_quality_definition(text: str) -> bool:
    normalized = normalize_definition_text(text)
    if (
        not normalized
        or normalized in {"none", "no definition", "no definition provided.", "the concept", "the concept."}
        or normalized.startswith("of ")
        or normalized.startswith("note ")
        or normalized.startswith("note:")
        or normalized.startswith("_:")
        or "http://" in normalized
        or "https://" in normalized
        or "improperly formatted iri" in normalized
        or "error" in normalized
        or "current axiom" in normalized
        or "will have to be" in normalized
        or "what about" in normalized
        or "added in the ontology" in normalized
        or "play with inconsistencies" in normalized
    ):
        return True
    return len(normalized.split()) < 4 or len(normalized) < 24


def compute_depth(entity, memo=None, visiting=None):
    return class_depth(entity, memo=memo, visiting=visiting)

def get_siblings(entity):
    return class_siblings(entity)

def compute_global_metrics(classes):
    return global_class_metrics(classes)

def compute_selection_weight(entity, global_metrics):
    return selection_weight(entity, global_metrics)

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
    s = "".join(safe).strip("_")
    return s or "unnamed"


class OntologyLoader(BaseOntologyLoader):
    def load(self):
        logging.info(f"Loading ontology: {self.file_path}")
        return super().load()

    def preload_entities(self):
        """
        Preload entity attributes to improve performance.
        """
        if not self.onto:
            return
        for cls in self.onto.classes():
            try:
                _ = getattr(cls, "IAO_0000115", None)
                _ = getattr(cls, "definition", None)
                _ = getattr(cls, "comment", None)
                _ = getattr(cls, "label", None)
            except Exception as e:
                logging.warning(f"Error preloading attributes for {getattr(cls, 'name', 'unknown')}: {e}")

    def get_all_classes_with_definition(self):
        """
        Get all classes with valid definitions (excluding owl.Thing).
        """
        if not self.onto:
            return []
        return [
            cls
            for cls in self.onto.classes()
            if cls != owl.Thing and not is_low_quality_definition(get_definition(cls))
        ]

class DefinitionQuestionGenerator:
    """
    Generate open-ended definition questions for ontology classes.
    """
    def __init__(self, classes, global_metrics, max_questions=100, mask_concept: bool = True):
        self.classes = classes
        self.global_metrics = global_metrics
        self.max_questions = max_questions
        self.inferred = False  # Placeholder; extend if inference is implemented
        self.mask_concept = mask_concept

    def _collect_aliases(self, entity: ThingClass) -> List[str]:
        aliases: set = set()
        for lab in (getattr(entity, 'label', []) or []) + (getattr(entity, 'prefLabel', []) or []):
            try:
                aliases.add(str(lab))
            except Exception:
                pass
        try:
            iri_s = str(entity.iri)
            local = iri_s.split('#')[-1] if '#' in iri_s else iri_s.rsplit('/', 1)[-1]
            aliases.add(local)
        except Exception:
            pass
        for prop_name in ("altLabel", "alternativeLabel", "hasExactSynonym", "hasRelatedSynonym", "hasBroadSynonym", "hasNarrowSynonym"):
            vals = getattr(entity, prop_name, []) or []
            for v in vals:
                try:
                    aliases.add(str(v))
                except Exception:
                    pass
        variants = set(a.replace('_', ' ').replace('-', ' ') for a in aliases)
        aliases |= variants
        return [a for a in aliases if len(a.strip()) >= 2]

    def _mask_definition_text(self, text: str, entity: ThingClass) -> str:
        if not self.mask_concept or not text:
            return text
        masked = text
        for alias in sorted(self._collect_aliases(entity), key=len, reverse=True):
            escaped = re.escape(alias)
            pattern = re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")
            masked = pattern.sub("the concept", masked)
        return masked

    def generate_question_for_target(self, target):
        """
        Generate a question for a given target class, including meta information.
        """
        try:
            target_label = get_label(target)
            target_def = self._mask_definition_text(get_definition(target), target)
            if is_low_quality_definition(target_def):
                return None
            stats = class_stats(target)
            depth = stats.depth
            sibling_count = stats.sibling_count
            subclass_count = stats.subclass_count
            parent_count = stats.parent_count

            # Get parent for relation and object_iri (assuming is_a relationship)
            parents = [p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing]
            relation = "is_a" if parents else None
            object_iri = str(parents[0].iri) if parents else None

            return {
                "prompt": f"Please provide the definition of the concept '{target_label}'.",
                "definition": target_def,
                "meta": {
                    "subject_iri": str(target.iri),
                    "subject_label": target_label,
                    "subject_kind": "class",
                    "relation": "class_definition_open",
                    "object_iri": object_iri,
                    "object_label": None,
                    "object_kind": None,
                    "class_context_iri": str(target.iri),
                    "class_context_label": target_label,
                    "depth": depth,
                    "sibling_count": sibling_count,
                    "subclass_count": subclass_count,
                    "parent_count": parent_count,
                    "inferred": self.inferred
                }
            }
        except Exception as e:
            logging.warning(f"Error generating question for {getattr(target, 'name', 'unknown')}: {e}")
            return None

    def generate_all_questions(self):
        """
        Generate questions by sampling classes based on their weights.
        Returns a list of question dictionaries.
        """
        if not self.classes:
            return []

        # Compute weights for all classes
        weights = [compute_selection_weight(cls, self.global_metrics) for cls in self.classes]
        total_weight = sum(weights)
        if total_weight <= 0:
            weights = [1.0 / len(self.classes)] * len(self.classes)  # Uniform weights if all are zero

        # Weighted sampling without replacement to ensure uniqueness of targets
        pool = list(zip(self.classes, weights))
        # Filter out zero-weight entries to avoid degenerate loops
        pool = [(c, w) for (c, w) in pool if w > 0]
        if not pool:
            pool = list(zip(self.classes, [1.0] * len(self.classes)))

        num_questions = min(self.max_questions, len({str(c.iri) for c, _ in pool}))
        selected_by_iri = set()
        selected_classes = []

        while len(selected_classes) < num_questions and pool:
            classes_in_pool, weights_in_pool = zip(*pool)
            chosen = random.choices(classes_in_pool, weights=weights_in_pool, k=1)[0]
            iri = str(chosen.iri)
            if iri not in selected_by_iri:
                selected_by_iri.add(iri)
                selected_classes.append(chosen)
            # Remove chosen from pool (all entries with same IRI just in case)
            pool = [(c, w) for (c, w) in pool if str(c.iri) != iri]

        questions = []
        for target in selected_classes:
            question = self.generate_question_for_target(target)
            if question:
                questions.append(question)

        return questions

def save_questions(questions, save_path):
    save_json(questions, Path(save_path), description="questions")


def write_csv(questions, csv_path: Path, task_label: str, domain: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "definition", "task_label", "iri", "label", "depth", "domain"])
        writer.writeheader()
        for item in questions:
            meta = item.get("meta", {}) or {}
            writer.writerow(
                {
                    "question": item.get("prompt", ""),
                    "definition": item.get("definition", ""),
                    "task_label": task_label,
                    "iri": meta.get("subject_iri", ""),
                    "label": meta.get("subject_label", ""),
                    "depth": meta.get("depth", ""),
                    "domain": domain,
                }
            )

def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    concept_scope: str,
    max_questions: int,
    task_label: str,
) -> None:
    """Process a single OWL file, generate questions, and save them."""
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f"class_definitions_{safe_stem}.json"
    csv_path = out_dir / f"class_definitions_{safe_stem}.csv"
    empty_path = empty_marker_path(out_dir, "class_definitions", safe_stem)
    if save_path.exists() and csv_path.exists():
        logging.info(f"Skip existing: {save_path}")
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return
    loader = OntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    onto = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    loader.preload_entities()
    classes = loader.get_all_classes_with_definition()
    # concept scope filter
    if concept_scope != 'all':
        def is_native(c):
            return getattr(getattr(c, 'namespace', None), 'ontology', None) is onto
        def is_imported(c):
            o = getattr(getattr(c, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            classes = [c for c in classes if is_native(c)]
        else:
            classes = [c for c in classes if is_imported(c)]
    logging.info(f"Found {len(classes)} classes with definitions in {file_path}")
    if not classes:
        logging.info(f"No defined classes in {file_path}")
        save_empty_marker(empty_path, source_file=file_path, reason="no_classes_with_usable_definitions")
        return
    global_metrics = compute_global_metrics(classes)
    gen = DefinitionQuestionGenerator(classes, global_metrics, max_questions=max_questions, mask_concept=False)
    questions = gen.generate_all_questions()
    logging.info(f"Generated {len(questions)} questions for {file_path}")
    if questions:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_questions(questions, str(save_path))
        try:
            domain = "/".join(file_path.relative_to(input_root).parts[:-1]) or file_path.parent.name
        except Exception:
            domain = file_path.parent.name
        write_csv(questions, csv_path, task_label=task_label, domain=domain)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_definition_generation_questions",
            extra={"classes_with_definitions": len(classes)},
        )
    # Clear caches
    definition_cache.clear()
    label_cache.clear()

def main():
    parser = argparse.ArgumentParser(description='Generate open-ended class definition questions from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=100, help='Max questions per ontology.')
    parser.add_argument('--task-label', type=str, default='3_2', help='Task label written to generated CSV rows.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of classes.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()
    configure_logging(args.log, "process_3_1.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl','.rdf','.rdfs','.ttl')
    files, input_root = discover_ontology_files(input_path, exts)

    logging.info(f"Found {len(files)} files.")
    onto_paths = resolve_onto_paths(args.onto_path)
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                load_imports=not args.no_imports,
                onto_paths=onto_paths,
                concept_scope=args.concept_scope,
                max_questions=args.max_questions,
                task_label=args.task_label,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == "__main__":
    main()
