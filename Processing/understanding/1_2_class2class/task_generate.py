import os
import json
import random
import logging
import argparse
import warnings
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from contextlib import ExitStack, redirect_stdout, redirect_stderr

from rdflib import URIRef, RDF
from owlready2 import World, ThingClass, Restriction, owl, onto_path, set_log_level
from collections import defaultdict


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    BaseOntologyLoader,
    build_mirrored_output_dir,
    class_stats,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    get_label,
    limit_questions_by_subject,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)


# Caches
label_cache: Dict[str, str] = {}
MAX_DISTRACTOR_CANDIDATES = 1000


class _NullWriter:
    def write(self, _):
        return 0
    def flush(self):
        return None


def silence_stdio(enabled: bool):
    if not enabled:
        class _Noop:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return _Noop()
    null = _NullWriter()
    stack = ExitStack()
    stack.enter_context(redirect_stdout(null))
    stack.enter_context(redirect_stderr(null))
    return stack


def cached_label(entity) -> str:
    key = str(entity.iri)
    if key not in label_cache:
        label_cache[key] = get_label(entity)
    return label_cache[key]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").replace("-", " ")).strip().lower()


def label_tokens(text: str) -> set:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_text(text)) if len(token) >= 3}


def labels_too_similar(left: str, right: str) -> bool:
    left_tokens = label_tokens(left)
    right_tokens = label_tokens(right)
    if not left_tokens or not right_tokens:
        return normalize_text(left) == normalize_text(right)
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) > 0.72


def is_generic_top_class(entity, metadata: Dict) -> bool:
    label = cached_label(entity).strip()
    if not label or label == "Unnamed":
        return True
    normalized = normalize_text(label).strip(".")
    if normalized in {"thing", "motif", "function", "person", "type", "class", "entity"}:
        return True
    alpha = re.sub(r"[^A-Za-z]", "", label)
    return bool(alpha) and len(alpha) >= 4 and alpha.isupper() and metadata.get(entity, {}).get("depth", 0) <= 2


def humanize_relation(rel_name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', rel_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.replace('_', ' ').lower()


def make_prompt(subj_label: str, rel_name: str) -> str:
    templates = {
        'subclassOf': f"Which of the following classes is the superclass of '{subj_label}'?",
        'equivalentTo': f"Which of the following classes is equivalent to '{subj_label}'?",
        'disjointWith': f"Which of the following classes is disjoint with '{subj_label}'?",
        'complementOf': f"Which of the following classes is the complement of '{subj_label}'?",
        'unionOf': f"Which of the following classes is a member of the union defining '{subj_label}'?",
        'intersectionOf': f"Which of the following classes is a member of the intersection defining '{subj_label}'?",
    }
    return templates.get(rel_name, f"Which of the following classes {humanize_relation(rel_name)} '{subj_label}'?")


class RelationQuestionGenerator:
    def __init__(self, triples: Iterable[Tuple[ThingClass, str, ThingClass]], all_classes: Iterable[ThingClass], metadata: Dict):
        self.triples = list(triples)
        self.all_classes = list(all_classes)
        self.meta = metadata
        self.disjoint_sets = defaultdict(set)
        for c in self.all_classes:
            for dis in getattr(c, 'disjoint_with', []):
                if isinstance(dis, ThingClass):
                    self.disjoint_sets[c].add(dis)

    def _get_distractors(self, obj: ThingClass, k: int) -> List[ThingClass]:
        if obj not in self.meta:
            return []
        obj_anc = self.meta[obj]['ancestors']
        distractors: List[ThingClass] = []
        candidates = list(self.disjoint_sets[obj])
        random.shuffle(candidates)
        for c in candidates:
            if c is not obj and not is_generic_top_class(c, self.meta) and len(distractors) < k:
                distractors.append(c)
        if len(distractors) < k:
            obj_depth = self.meta[obj]['depth']
            remaining = [
                c for c in self.all_classes
                if c is not obj
                and c not in obj_anc
                and c not in distractors
                and not is_generic_top_class(c, self.meta)
                and abs(self.meta.get(c, {}).get('depth', obj_depth) - obj_depth) <= 1
                and not labels_too_similar(cached_label(c), cached_label(obj))
            ]
            random.shuffle(remaining)
            remaining = remaining[:MAX_DISTRACTOR_CANDIDATES]
            remaining.sort(key=lambda c: (abs(self.meta.get(c, {}).get('depth', obj_depth) - obj_depth), cached_label(c)))
            distractors.extend(remaining[:k - len(distractors)])
        return distractors[:k]

    def generate_all(self, max_q: int) -> List[Dict]:
        questions = []
        letters = ['A', 'B', 'C', 'D']
        triples = list(self.triples)
        random.shuffle(triples)
        buffer_limit = max_q * 4 if max_q else None
        for subj, rel, obj in triples:
            if subj not in self.meta or obj not in self.meta:
                continue
            if rel == 'subclassOf' and is_generic_top_class(obj, self.meta):
                continue
            if is_generic_top_class(subj, self.meta) or is_generic_top_class(obj, self.meta):
                continue
            stats = class_stats(subj, self.all_classes)
            distractors = self._get_distractors(obj, 3)
            if len(distractors) < 3:
                continue
            options = [obj] + distractors
            random.shuffle(options)
            opts = []
            correct = None
            for i, choice in enumerate(options):
                opts.append({'option_letter': letters[i], 'label': cached_label(choice)})
                if choice is obj:
                    correct = letters[i]
            labels = [o['label'] for o in opts]
            if len({normalize_text(label) for label in labels}) != len(labels):
                continue
            prompt = make_prompt(cached_label(subj), rel)
            questions.append({
                'prompt': prompt,
                'options': opts,
                'correct_answer': correct,
                'meta': {
                    'subject_iri': str(subj.iri),
                    'subject_label': cached_label(subj),
                    'subject_kind': 'class',
                    'relation': rel,
                    'object_iri': str(obj.iri),
                    'object_label': cached_label(obj),
                    'object_kind': 'class',
                    'class_context_iri': str(subj.iri),
                    'class_context_label': cached_label(subj),
                    'depth': stats.depth,
                    'sibling_count': stats.sibling_count,
                    'subclass_count': stats.subclass_count,
                    'parent_count': stats.parent_count,
                }
            })
            if buffer_limit and len(questions) >= buffer_limit:
                break
        return limit_questions_by_subject(questions, max_q)


def compute_ancestors(parent_map: Dict[ThingClass, List[ThingClass]], all_classes: Iterable[ThingClass]) -> Dict[ThingClass, set]:
    ancestors_map: Dict[ThingClass, set] = {c: set() for c in all_classes}

    def get_ancestors(c: ThingClass):
        if c not in ancestors_map:
            return set()
        if ancestors_map[c]:
            return ancestors_map[c]
        anc = set()
        for p in parent_map[c]:
            if p in ancestors_map:
                anc.add(p)
                anc.update(get_ancestors(p))
        ancestors_map[c] = anc
        return anc

    for c in all_classes:
        get_ancestors(c)
    return ancestors_map


def extract_and_prepare(onto) -> Tuple[List[ThingClass], List[Tuple[ThingClass, str, ThingClass]], Dict]:
    all_classes = list(onto.classes())
    parent_map: Dict[ThingClass, List[ThingClass]] = {}
    for c in all_classes:
        try:
            parent_map[c] = [p for p in c.is_a if isinstance(p, ThingClass) and p != owl.Thing]
        except Exception:
            parent_map[c] = []

    child_map = defaultdict(list)
    for c, parents in parent_map.items():
        for p in parents:
            child_map[p].append(c)

    ancestors_map = compute_ancestors(parent_map, all_classes)

    metadata: Dict[ThingClass, Dict] = {}
    for c in all_classes:
        if parent_map[c]:
            try:
                depth = max(metadata[p]['depth'] for p in parent_map[c]) + 1
            except Exception:
                depth = 0
        else:
            depth = 0
        siblings = sum(len(child_map[p]) - 1 for p in parent_map[c])
        metadata[c] = {
            'depth': depth,
            'siblings': siblings,
            'subclasses': len(child_map.get(c, [])),
            'parents': len(parent_map[c]),
            'ancestors': ancestors_map[c],
        }

    relations = {'subclassOf', 'equivalentTo', 'disjointWith', 'complementOf', 'unionOf', 'intersectionOf'}
    triples: List[Tuple[ThingClass, str, ThingClass]] = []

    for c in all_classes:
        try:
            for p in parent_map[c]:
                if p != owl.Thing:
                    triples.append((c, 'subclassOf', p))
            for eq in getattr(c, 'equivalent_to', []):
                if isinstance(eq, ThingClass):
                    triples.append((c, 'equivalentTo', eq))
            for dis in getattr(c, 'disjoint_with', []):
                if isinstance(dis, ThingClass):
                    triples.append((c, 'disjointWith', dis))
            for r in c.is_a:
                if isinstance(r, Restriction):
                    name = r.property.python_name
                    if name in relations:
                        val = getattr(r, 'value', None) or getattr(r, 'some_values_from', None) or getattr(r, 'all_values_from', None)
                        if isinstance(val, ThingClass):
                            triples.append((c, name, val))
        except Exception:
            pass

    try:
        for a, b in onto.disjoint_classes():
            As = a if isinstance(a, (list, tuple, set)) else [a]
            Bs = b if isinstance(b, (list, tuple, set)) else [b]
            for x in As:
                for y in Bs:
                    if isinstance(x, ThingClass) and isinstance(y, ThingClass):
                        triples.append((x, 'disjointWith', y))
    except Exception:
        pass

    graph = onto.world.as_rdflib_graph()
    for c in all_classes:
        try:
            subj = URIRef(c.iri)
            for pred, obj in graph.predicate_objects(subj):
                local = str(pred).rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                if local in ('complementOf', 'unionOf', 'intersectionOf'):
                    node = obj
                    while node and node != RDF.nil:
                        first = graph.value(node, RDF.first)
                        ent = onto.world._entities.get(str(first))
                        if isinstance(ent, ThingClass):
                            triples.append((c, local, ent))
                        node = graph.value(node, RDF.rest)
                        if node == obj:
                            break
        except Exception:
            pass

    for prop in onto.object_properties():
        try:
            name = prop.python_name
            if name in relations:
                for c in all_classes:
                    for o in prop[c]:
                        if isinstance(o, ThingClass):
                            triples.append((c, name, o))
        except Exception:
            pass

    triples = list(set(triples))
    return all_classes, triples, metadata


class OntologyLoader(BaseOntologyLoader):
    pass


def save_questions(questions: List[Dict], save_path: Path) -> None:
    save_json(questions, save_path, description="questions")


def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    max_questions: int,
    load_imports: bool,
    concept_scope: str,
    onto_paths: Optional[List[Path]],
    suppress_io: bool,
) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    out_file = out_dir / f"relations_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "relations", safe_stem)
    if out_file.exists():
        logging.info("Skip existing: %s", out_file)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    loader = OntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    with silence_stdio(suppress_io):
        onto = loader.load()
    if not onto:
        logging.error(f"Failed to load ontology {file_path}")
        return

    with silence_stdio(suppress_io):
        all_classes, triples, metadata = extract_and_prepare(onto)
        # Filter by concept scope (subject side): all, native, imported
        if concept_scope != 'all':
            def is_native(c: ThingClass) -> bool:
                return getattr(getattr(c, 'namespace', None), 'ontology', None) is onto
            def is_imported(c: ThingClass) -> bool:
                o = getattr(getattr(c, 'namespace', None), 'ontology', None)
                return (o is not None) and (o is not onto)
            if concept_scope == 'native':
                allowed = {c for c in all_classes if is_native(c)}
            else:
                allowed = {c for c in all_classes if is_imported(c)}
            triples = [t for t in triples if t[0] in allowed]
        gen = RelationQuestionGenerator(triples, all_classes, metadata)
        questions = gen.generate_all(max_questions)

    logging.info(f"Generated {len(questions)} questions from {file_path.name}.")
    if questions:
        save_questions(questions, out_file)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_class_relation_questions",
            extra={"classes": len(all_classes), "triples": len(triples)},
        )


def main():
    parser = argparse.ArgumentParser(description='Generate class-to-class relation MCQs from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory for Windows-safe mirrored folders and JSON.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=30000, help='Maximum questions per ontology.')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--concept-scope', type=str, choices=['all', 'native', 'imported'], default='all', help='Filter by origin of subject classes: all/native/imported.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library stdout/stderr noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    configure_logging(args.log, "process.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')

    files, input_root = discover_ontology_files(input_path, exts)

    logging.info(f"Found {len(files)} files to process.")
    for fp in files:
        try:
            with file_timeout(args.file_timeout_seconds):
                process_owl_file(
                    file_path=fp,
                    input_root=input_root,
                    output_root=output_root,
                    max_questions=args.max_questions,
                    load_imports=not args.no_imports,
                    concept_scope=args.concept_scope,
                    onto_paths=resolve_onto_paths(args.onto_path),
                    suppress_io=args.no_warnings,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")


if __name__ == '__main__':
    main()
