import json
import os
import random
import logging
import argparse
import warnings
import sys
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from owlready2 import DataPropertyClass, ObjectPropertyClass, World, ThingClass, owl, onto_path, set_log_level


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    class_depth,
    class_stats,
    configure_world_paths,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    get_label,
    limit_questions_by_subject,
    load_ontology,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)


# Config and caches
MIN_DISTRACTORS = 2
MAX_DISTRACTOR_CANDIDATES = 200
PROPERTY_STOP_TOKENS = {
    "has", "have", "is", "are", "of", "by", "for", "with", "related",
    "associated", "part", "contains", "contain", "about", "to", "from",
    "in", "on", "at", "the", "a", "an",
}


def normalize_label(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text or ''))
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def label_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_label(text)) if len(token) >= 3}


def leak_tokens(text: str) -> set[str]:
    return {token for token in label_tokens(text) if token not in PROPERTY_STOP_TOKENS}


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = label_tokens(left)
    right_tokens = label_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

def compute_depth(entity, memo=None):
    return class_depth(entity, memo=memo)

def get_siblings(entity):
    return class_stats(entity).sibling_count

def get_ancestors(entity, memo=None):
    """Recursively collect all super-classes."""
    if memo is None:
        memo = {}
    if entity in memo:
        return memo[entity]
    ancestors = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass):
            ancestors.add(p)
            ancestors |= get_ancestors(p, memo)
    memo[entity] = ancestors
    return ancestors

"""Property domain/range extraction and question generation"""


def render_range_label(value) -> str:
    if value is None:
        return ""
    if value is str:
        return "string"
    if value is int:
        return "integer"
    if value is float:
        return "float"
    if value is bool:
        return "boolean"
    return get_label(value)


def property_answer_leak_score(prop, answer) -> float:
    prop_tokens = leak_tokens(get_label(prop))
    answer_tokens = leak_tokens(render_range_label(answer))
    if not answer_tokens:
        return 0.0
    return len(prop_tokens & answer_tokens) / len(answer_tokens)


def is_usable_range_value(value) -> bool:
    label = render_range_label(value)
    return bool(label) and normalize_label(label) not in {"none", "null", "nan"}


def leak_flag(score: float) -> str:
    if score >= 0.5:
        return "high"
    if score > 0:
        return "medium"
    return "none"


def detect_leak(prop, relation: str, answer) -> tuple[bool, list[str], float, str]:
    prop_label = get_label(prop)
    leak_score = property_answer_leak_score(prop, answer)
    flag = leak_flag(leak_score)
    leak_types = []
    if leak_score > 0:
        leak_types.append("property_label_overlap")
    if relation == "range" and not isinstance(answer, ThingClass):
        prop_tokens = leak_tokens(prop_label)
        datatype = render_range_label(answer)
        datatype_cues = {
            "integer": {"age", "count", "number", "num", "year", "size", "height", "width", "length", "amount", "quantity"},
            "float": {"rate", "ratio", "score", "value", "weight", "temperature", "latitude", "longitude", "percent"},
            "boolean": {"is", "has", "can", "flag", "enabled", "valid", "active"},
            "string": {"name", "label", "comment", "description", "text", "title", "identifier", "id"},
        }
        if prop_tokens & datatype_cues.get(datatype, set()):
            leak_types.append("datatype_cue")
            if flag == "none":
                flag = "medium"
    return flag == "high" or "datatype_cue" in leak_types, leak_types, round(leak_score, 4), flag


def is_valid_semantic_property(prop) -> bool:
    return isinstance(prop, (ObjectPropertyClass, DataPropertyClass)) and bool(get_label(prop))


def property_kind(prop) -> str:
    return "object_property" if isinstance(prop, ObjectPropertyClass) else "data_property"


def is_generic_class(cls) -> bool:
    if not isinstance(cls, ThingClass):
        return False
    label = normalize_label(get_label(cls)).strip(".")
    return cls == owl.Thing or label in {"thing", "entity", "object", "resource", "class", "concept"}


def extract_property_semantics_info(onto):
    """Extract explicit property domain/range information.

    Returns individual question targets instead of domain/range cross-products so
    properties with only a domain or only a range can still produce valid MCQs.
    """
    targets = []
    prop_info = {}
    object_property_names = {prop.name for prop in onto.object_properties()}
    data_property_names = {prop.name for prop in onto.data_properties()}
    mixed_names = object_property_names & data_property_names
    properties = [
        prop
        for prop in list(onto.object_properties()) + list(onto.data_properties())
        if is_valid_semantic_property(prop) and prop.name not in mixed_names
    ]
    logging.info("Found %d semantic properties in ontology", len(properties))
    for prop in properties:
        domains, ranges = set(), set()
        for domain in getattr(prop, "domain", []) or []:
            if isinstance(domain, ThingClass) and not is_generic_class(domain):
                domains.add(domain)
        for range_value in getattr(prop, "range", []) or []:
            if isinstance(prop, ObjectPropertyClass):
                if isinstance(range_value, ThingClass) and not is_generic_class(range_value):
                    ranges.add(range_value)
            else:
                if is_usable_range_value(range_value):
                    ranges.add(range_value)
        if not domains and not ranges:
            continue
        prop_info[prop] = {"domains": domains, "ranges": ranges, "kind": property_kind(prop)}
        for domain in domains:
            targets.append((prop, "domain", domain))
        for range_value in ranges:
            targets.append((prop, "range", range_value))
    logging.info("Generated %d property semantic targets", len(targets))
    return targets, prop_info

"""Question generation"""

class PropertyDomainRangeQuestionGenerator:
    def __init__(self, targets, prop_info, all_classes):
        self.targets = targets
        self.prop_info = prop_info
        self.all_classes = [c for c in all_classes if isinstance(c, ThingClass) and not is_generic_class(c)]
        self.all_properties = list(prop_info.keys())
        self._stats_cache = {}
        datatype_pool = [str, int, float, bool] + sorted(
            {
                r
                for info in prop_info.values()
                for r in info.get("ranges", set())
                if not isinstance(r, ThingClass)
            },
            key=lambda item: render_range_label(item),
        )
        self.datatype_pool = []
        seen_datatypes = set()
        for item in datatype_pool:
            key = render_range_label(item)
            if key not in seen_datatypes:
                seen_datatypes.add(key)
                self.datatype_pool.append(item)

    def _candidate_pool(self, relation, correct_answer):
        if relation == "domain" or isinstance(correct_answer, ThingClass):
            return self.all_classes
        return self.datatype_pool

    def _candidate_score(self, candidate, correct_answer, prop, relation):
        if not isinstance(correct_answer, ThingClass) or not isinstance(candidate, ThingClass):
            return 0
        score = 0
        try:
            correct_parents = {p for p in getattr(correct_answer, "is_a", []) if isinstance(p, ThingClass)}
            candidate_parents = {p for p in getattr(candidate, "is_a", []) if isinstance(p, ThingClass)}
            if correct_parents & candidate_parents:
                score += 5
            if correct_answer in candidate_parents or candidate in correct_parents:
                score += 2
        except Exception:
            pass
        prop_overlap_correct = lexical_overlap(get_label(prop), render_range_label(correct_answer))
        prop_overlap_candidate = lexical_overlap(get_label(prop), render_range_label(candidate))
        score -= abs(prop_overlap_candidate - prop_overlap_correct) * 4
        score += lexical_overlap(render_range_label(correct_answer), render_range_label(candidate)) * 3
        correct_leak = property_answer_leak_score(prop, correct_answer)
        candidate_leak = property_answer_leak_score(prop, candidate)
        score -= abs(candidate_leak - correct_leak) * 6
        if correct_leak >= 0.5 and candidate_leak >= 0.5:
            score += 8
        return score

    def _cached_stats(self, cls):
        if cls not in self._stats_cache:
            self._stats_cache[cls] = class_stats(cls)
        return self._stats_cache[cls]

    def _structural_candidates(self, correct_answer):
        if not isinstance(correct_answer, ThingClass):
            return []
        structural = []
        seen = {correct_answer}
        for parent in getattr(correct_answer, "is_a", []) or []:
            if not isinstance(parent, ThingClass):
                continue
            try:
                siblings = list(parent.subclasses())
            except Exception:
                siblings = []
            random.shuffle(siblings)
            for sibling in siblings[:100]:
                if isinstance(sibling, ThingClass) and sibling not in seen and not is_generic_class(sibling):
                    structural.append(sibling)
                    seen.add(sibling)
        try:
            children = list(correct_answer.subclasses())
            random.shuffle(children)
            for child in children[:100]:
                if isinstance(child, ThingClass) and child not in seen and not is_generic_class(child):
                    structural.append(child)
                    seen.add(child)
        except Exception:
            pass
        return structural

    def _is_leak_balanced(self, prop, correct_answer, distractors) -> bool:
        correct_score = property_answer_leak_score(prop, correct_answer)
        if correct_score < 0.5:
            return True
        correct_parents = set()
        if isinstance(correct_answer, ThingClass):
            correct_parents = {p for p in getattr(correct_answer, "is_a", []) if isinstance(p, ThingClass)}
        for distractor in distractors:
            if property_answer_leak_score(prop, distractor) >= 0.5:
                return True
            if lexical_overlap(render_range_label(correct_answer), render_range_label(distractor)) >= 0.25:
                return True
            if isinstance(distractor, ThingClass):
                distractor_parents = {p for p in getattr(distractor, "is_a", []) if isinstance(p, ThingClass)}
                if correct_parents and correct_parents & distractor_parents:
                    return True
        return False

    def _select_distractors(self, prop, relation, correct_answer, true_values, num_choices):
        candidates = [
            c for c in self._candidate_pool(relation, correct_answer)
            if c != correct_answer and c not in true_values
        ]
        if not candidates:
            return []
        if isinstance(correct_answer, ThingClass) and len(candidates) > MAX_DISTRACTOR_CANDIDATES:
            structural = self._structural_candidates(correct_answer)
            structural_set = set(structural)
            remaining = [candidate for candidate in candidates if candidate not in structural_set]
            random.shuffle(remaining)
            candidates = structural + remaining[: max(0, MAX_DISTRACTOR_CANDIDATES - len(structural))]
        elif len(candidates) > MAX_DISTRACTOR_CANDIDATES:
            random.shuffle(candidates)
            candidates = candidates[:MAX_DISTRACTOR_CANDIDATES]
        if isinstance(correct_answer, ThingClass):
            random.shuffle(candidates)
            candidates = sorted(
                candidates,
                key=lambda c: self._candidate_score(c, correct_answer, prop, relation),
                reverse=True,
            )
        else:
            candidates = sorted(candidates, key=render_range_label)
        distractors = []
        seen_labels = {render_range_label(correct_answer)}
        for candidate in candidates:
            label = render_range_label(candidate)
            if label in seen_labels:
                continue
            distractors.append(candidate)
            seen_labels.add(label)
            if len(distractors) >= num_choices - 1:
                break
        return distractors

    def generate_one(self, prop, relation, correct_answer, num_choices=4):
        try:
            # Metadata
            stats = class_stats(correct_answer) if isinstance(correct_answer, ThingClass) else None

            true_values = self.prop_info[prop]["domains"] if relation == "domain" else self.prop_info[prop]["ranges"]
            logging.debug("Generating %s question for property %s", relation, get_label(prop))
            distractors = self._select_distractors(prop, relation, correct_answer, true_values, num_choices)

            # Ensure a minimum number of distractors
            if len(distractors) < num_choices - 1:
                logging.warning(f"Insufficient distractors ({len(distractors)}) for {get_label(prop)}, skipping question")
                return None
            if not self._is_leak_balanced(prop, correct_answer, distractors):
                logging.info(
                    "Skipping high-leak U3 question without balanced distractor: %s -> %s",
                    get_label(prop),
                    render_range_label(correct_answer),
                )
                return None

            options = [correct_answer] + distractors[:num_choices - 1]
            if len({render_range_label(option) for option in options}) != num_choices:
                logging.warning("Duplicate or invalid option labels for %s, skipping question", get_label(prop))
                return None
            random.shuffle(options)
            answer_leak_risk, leak_types, leak_score, leak_flag_value = detect_leak(prop, relation, correct_answer)

            letters = ['A', 'B', 'C', 'D'][:num_choices]
            opts = []
            correct = None
            for i, choice in enumerate(options):
                label = render_range_label(choice)
                opts.append({'option_letter': letters[i], 'label': label})
                if choice == correct_answer:
                    correct = letters[i]

            prompt = (
                f"Which of the following is a valid {relation} for the "
                f"{'object property' if isinstance(prop, ObjectPropertyClass) else 'data property'} "
                f"'{get_label(prop)}'?"
            )

            # Safe handling of range_type IRI and label
            object_iri = str(getattr(correct_answer, 'iri', str(correct_answer))) if correct_answer else 'N/A'
            object_label = render_range_label(correct_answer)
            is_class_answer = isinstance(correct_answer, ThingClass)

            return {
                'prompt': prompt,
                'options': opts,
                'correct_answer': correct,
                'meta': {
                    'subject_iri': str(prop.iri),
                    'subject_label': get_label(prop),
                    'subject_kind': 'property',
                    'property_kind': property_kind(prop),
                    'relation': 'property_domain' if relation == 'domain' else 'property_range',
                    'constraint_kind': 'property_domain' if relation == 'domain' else 'property_range',
                    'source_axiom': 'rdfs:domain' if relation == 'domain' else 'rdfs:range',
                    'extraction_backend': 'owlready2_native',
                    'answer_leak_risk': answer_leak_risk,
                    'leak_type': leak_types,
                    'leak_score': leak_score,
                    'leak_flag': leak_flag_value,
                    'leak_balanced_distractors': self._is_leak_balanced(prop, correct_answer, distractors),
                    'distractor_strategy': 'type_matched_structural_lexical',
                    'object_iri': object_iri,
                    'object_label': object_label,
                    'object_kind': 'class' if is_class_answer else 'datatype',
                    'class_context_iri': str(correct_answer.iri) if is_class_answer else None,
                    'class_context_label': get_label(correct_answer) if is_class_answer else None,
                    'depth': stats.depth if stats else None,
                    'sibling_count': stats.sibling_count if stats else None,
                    'subclass_count': stats.subclass_count if stats else None,
                    'parent_count': stats.parent_count if stats else None,
                }
            }
        except Exception as e:
            logging.error(f"Failed to generate question for property {prop}: {e}")
            return None

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.targets)
        target_limit = max_q * 20 if max_q else None
        for idx, (prop, relation, value) in enumerate(self.targets, start=1):
            if target_limit and idx > target_limit:
                logging.info("Reached U3 target scan limit: %s", target_limit)
                break
            if idx % 200 == 0:
                logging.info("U3 progress: processed %d/%d targets, kept %d questions", idx, len(self.targets), len(questions))
            q = self.generate_one(prop, relation, value)
            if q:
                questions.append(q)
                if max_q and len(questions) >= max_q * 4:
                    break
        questions = limit_questions_by_subject(questions, max_q)
        logging.info(f"Generated {len(questions)} questions for this ontology")
        return questions

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


def save_questions(questions, save_path: Path):
    save_json(questions, save_path, description="questions", skip_empty=True)


def save_composition_summary(questions, save_path: Path):
    relation_counts = {}
    leak_counts = {}
    leak_flag_counts = {}
    leak_risk_count = 0
    leak_scores = []
    unbalanced_leak_count = 0
    for question in questions:
        meta = question.get("meta", {})
        relation = meta.get("constraint_kind", meta.get("relation", "unknown"))
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        leak_flag_value = meta.get("leak_flag", "none")
        leak_flag_counts[leak_flag_value] = leak_flag_counts.get(leak_flag_value, 0) + 1
        leak_scores.append(float(meta.get("leak_score", 0.0) or 0.0))
        if meta.get("answer_leak_risk") and not meta.get("leak_balanced_distractors", True):
            unbalanced_leak_count += 1
        if meta.get("answer_leak_risk"):
            leak_risk_count += 1
            for leak_type in meta.get("leak_type", []) or ["unknown"]:
                leak_counts[leak_type] = leak_counts.get(leak_type, 0) + 1
    total = len(questions)
    summary = {
        "total": total,
        "constraint_kind_counts": relation_counts,
        "constraint_kind_percent": {
            key: round(value / total, 4) if total else 0.0
            for key, value in relation_counts.items()
        },
        "answer_leak_risk_count": leak_risk_count,
        "leak_flag_counts": leak_flag_counts,
        "leak_type_counts": leak_counts,
        "avg_leak_score": round(sum(leak_scores) / total, 4) if total else 0.0,
        "max_leak_score": round(max(leak_scores), 4) if leak_scores else 0.0,
        "unbalanced_leak_count": unbalanced_leak_count,
    }
    save_json(summary, save_path, description="U3 composition summary")


def process_owl_file(file_path: Path, input_root: Path, output_root: Path, max_questions: Optional[int], load_imports: bool, onto_paths: Optional[List[Path]], suppress_warnings: bool, concept_scope: str) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    out_file = out_dir / f"property_semantics_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "property_semantics", safe_stem)
    if out_file.exists():
        logging.info("Skip existing: %s", out_file)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    world = World()
    configure_world_paths(world, onto_paths)
    onto = load_ontology(world, file_path, load_imports=load_imports)
    if onto is None:
        return

    all_classes = list(onto.classes())
    targets, prop_info = extract_property_semantics_info(onto)
    # Filter subject (property) origin by scope
    if concept_scope != 'all':
        def is_native_prop(p) -> bool:
            return getattr(getattr(p, 'namespace', None), 'ontology', None) is onto
        def is_imported_prop(p) -> bool:
            o = getattr(getattr(p, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        allowed = set()
        for prop in prop_info.keys():
            if concept_scope == 'native' and is_native_prop(prop):
                allowed.add(prop)
            if concept_scope == 'imported' and is_imported_prop(prop):
                allowed.add(prop)
        targets = [t for t in targets if t[0] in allowed]
    gen = PropertyDomainRangeQuestionGenerator(targets, prop_info, all_classes)
    questions = gen.generate_all(max_questions)

    if questions:
        save_questions(questions, out_file)
        save_composition_summary(questions, out_dir / f"property_semantics_{safe_stem}_summary.json")
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_property_semantics_questions",
            extra={"targets": len(targets), "properties": len(prop_info)},
        )
    world.close()


def main():
    parser = argparse.ArgumentParser(description='Generate property domain/range MCQs from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory with Windows-safe mirrored structure.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--concept-scope', type=str, choices=['all', 'native', 'imported'], default='all', help='Filter by origin of properties: all/native/imported.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    configure_logging(args.log, "process_1_3.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')
    files, input_root = discover_ontology_files(input_path, exts)
    max_q = None if args.max_questions == 0 else args.max_questions
    logging.info(f"Found {len(files)} ontology files")
    onto_paths = resolve_onto_paths(args.onto_path)
    for fp in files:
        try:
            with file_timeout(args.file_timeout_seconds):
                process_owl_file(
                    file_path=fp,
                    input_root=input_root,
                    output_root=output_root,
                    max_questions=max_q,
                    load_imports=not args.no_imports,
                    onto_paths=onto_paths,
                    suppress_warnings=args.no_warnings,
                    concept_scope=args.concept_scope,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"Processing {fp} failed: {e}")


if __name__ == '__main__':
    main()
