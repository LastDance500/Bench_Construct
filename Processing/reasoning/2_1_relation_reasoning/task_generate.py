import json
import os
import random
import logging
import re
import argparse
import warnings
import signal
import sys
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from owlready2 import World, ThingClass, owl, Restriction, sync_reasoner, onto_path, set_log_level


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    class_stats,
    configure_world_paths,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    load_ontology,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)

# Defaults
MAX_CLASSES = 1000
MAX_TRIPLES = 10000
REASONING_TIMEOUT = 300
MAX_FULL_REASONER_CLASSES = 300

# Caches
label_cache: Dict[str, str] = {}
depth_cache: Dict[ThingClass, int] = {}
ancestors_cache: Dict[ThingClass, set] = defaultdict(set)

# ---------- 获取标签 ----------
def get_label(entity):
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else entity.name
    label_cache[key] = label
    return label

def precompute_class_metadata(onto, classes):
    depths = {}
    ancestors_cache.clear()
    targets = set(classes)
    for cls in targets:
        ancestors = {a for a in cls.ancestors() if isinstance(a, ThingClass)}
        ancestors_cache[cls] = ancestors | {cls}
        depths[cls] = len([a for a in ancestors if a != owl.Thing])
    if owl.Thing not in depths:
        depths[owl.Thing] = 0
        ancestors_cache[owl.Thing] = {owl.Thing}
    return depths, ancestors_cache

def get_siblings(entity, classes):
    sibs = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass) and p != owl.Thing:
            sibs.update(c for c in p.subclasses() if c in classes)
    sibs.discard(entity)
    return sibs

def extract_explicit_class_triples(onto, relations, classes):
    triples = set()
    for cls in classes:
        if 'subclassOf' in relations:
            for parent in cls.is_a:
                if isinstance(parent, ThingClass) and parent != owl.Thing and parent in classes:
                    triples.add((cls, 'subclassOf', parent))
        if 'equivalentTo' in relations:
            for eq in cls.equivalent_to:
                if isinstance(eq, ThingClass) and eq in classes:
                    triples.add((cls, 'equivalentTo', eq))
        for restriction in cls.is_a:
            if isinstance(restriction, Restriction):
                prop = restriction.property
                name = getattr(prop, 'python_name', None)
                if not name or name not in relations:
                    continue
                filler = getattr(restriction, 'value', None) or \
                         getattr(restriction, 'some_values_from', None) or \
                         getattr(restriction, 'all_values_from', None)
                if isinstance(filler, ThingClass) and filler in classes:
                    triples.add((cls, name, filler))
    return triples

def humanize_relation(rel_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', rel_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.replace('_', ' ').lower()


def normalize_label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").replace("_", " ").replace("-", " ").lower()).strip()


def make_prompt(subj_label, rel_name, inferred=False):
    if rel_name == 'subclassOf':
        question = f"Which of the following classes is the superclass of '{subj_label}'?"
    elif rel_name == 'equivalentTo':
        question = f"Which of the following classes is equivalent to '{subj_label}'?"
    else:
        human = humanize_relation(rel_name)
        question = f"'{subj_label}' has which '{human}' relation to the following classes?"
    if inferred:
        return "After reasoning, " + question[:1].lower() + question[1:]
    return question

# ---------- 题目生成 ----------
class RelationQuestionGenerator:
    def __init__(self, triples, all_classes, inferred=False):
        self.triples = triples
        self.all_classes = all_classes
        self.inferred = inferred
        self.by_subject = defaultdict(list)
        for subj, rel, obj in triples:
            self.by_subject[subj].append((subj, rel, obj))

    def _select_triple(self, subj):
        return random.choice(self.by_subject[subj])

    def _get_distractors(self, subj, rel, obj, num_choices):
        # 对于 subclassOf，排除所有真正的超类；对于 equivalentTo，排除所有等价类；
        # 其他关系，也排除所有真实填充值
        if rel == 'subclassOf':
            true_supers = {
                anc for anc in subj.ancestors()
                if isinstance(anc, ThingClass) and anc in self.all_classes and anc not in (subj, owl.Thing)
            }
            disallowed = true_supers
        elif rel == 'equivalentTo':
            disallowed = set(subj.equivalent_to)
        else:
            source_triples = getattr(self, 'all_true_triples', self.triples)
            disallowed = {f for (s, r, f) in source_triples if s == subj and r == rel}

        # 随机候选
        candidates = random.sample(self.all_classes, min(100, len(self.all_classes)))
        distractors = []
        used_labels = {normalize_label_text(get_label(subj)), normalize_label_text(get_label(obj))}
        for c in candidates:
            if c in {subj, obj}:
                continue
            label_key = normalize_label_text(get_label(c))
            if not label_key or label_key in used_labels:
                continue
            if c in disallowed:
                continue
            distractors.append(c)
            used_labels.add(label_key)
            if len(distractors) >= num_choices - 1:
                break

        # 如果不足，再从剩下的里补
        if len(distractors) < num_choices - 1:
            extra = list(self.all_classes)
            random.shuffle(extra)
            for c in extra:
                if c in distractors or c in disallowed or c in {subj, obj}:
                    continue
                label_key = normalize_label_text(get_label(c))
                if not label_key or label_key in used_labels:
                    continue
                distractors.append(c)
                used_labels.add(label_key)
                if len(distractors) >= num_choices - 1:
                    break

        return distractors[:num_choices - 1]

    def generate_one(self, subj, rel, obj, num_choices=4):
        stats = class_stats(subj, self.all_classes)
        depth = stats.depth
        sibling_count = stats.sibling_count
        subclass_count = stats.subclass_count
        parent_count = stats.parent_count

        distractors = self._get_distractors(subj, rel, obj, num_choices)
        if len(distractors) < num_choices - 1:
            return None
        options = [obj] + distractors
        if len({normalize_label_text(get_label(option)) for option in options}) != num_choices:
            return None
        random.shuffle(options)

        letters = ['A', 'B', 'C', 'D']
        opts = []
        correct = None
        for i, choice in enumerate(options):
            label = get_label(choice)
            opts.append({'option_letter': letters[i], 'label': label})
            if choice == obj:
                correct = letters[i]

        prompt = make_prompt(get_label(subj), rel, self.inferred)
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(subj.iri),
                'subject_label': get_label(subj),
                'subject_kind': 'class',
                'relation': rel,
                'object_iri': str(obj.iri),
                'object_label': get_label(obj),
                'object_kind': 'class',
                'class_context_iri': str(subj.iri),
                'class_context_label': get_label(subj),
                'depth': depth,
                'sibling_count': sibling_count,
                'subclass_count': subclass_count,
                'parent_count': parent_count,
                'inferred': self.inferred
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        subjects = list(self.by_subject.keys())
        random.shuffle(subjects)
        for subj in subjects:
            subj, rel, obj = self._select_triple(subj)
            try:
                q = self.generate_one(subj, rel, obj)
                if q:
                    questions.append(q)
                if max_q and len(questions) >= max_q:
                    break
            except Exception as e:
                logging.warning(f"Error generating question: {e}")
        return questions

# ---------- 保存 ----------
def save_questions(questions, save_path: Path):
    save_json(questions, save_path, description="questions")

# ---------- 主流程 ----------
def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    max_questions: int,
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    no_warnings: bool,
    concept_scope: str,
) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f"relations_inferred_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "relations_inferred", safe_stem)
    if save_path.exists():
        logging.info("Skip existing: %s", save_path)
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
    original_class_count = len(all_classes)
    if original_class_count > MAX_CLASSES:
        all_classes = random.sample(all_classes, MAX_CLASSES)

    global depth_cache
    depth_cache, _ = precompute_class_metadata(onto, set(all_classes))

    relations = ['subclassOf', 'equivalentTo'] + [prop.python_name for prop in onto.object_properties()]
    explicit = extract_explicit_class_triples(onto, relations, all_classes)

    num_classes = len(all_classes)
    inferred = set()

    def _reason():
        try:
            with onto:
                sync_reasoner(infer_property_values=True)
        except Exception:
            return False
        return True

    if original_class_count <= MAX_FULL_REASONER_CLASSES:
        if no_warnings:
            null = type('NW', (), {'write': lambda self, x: 0, 'flush': lambda self: None})()
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(null))
                stack.enter_context(redirect_stderr(null))
                _reason()
        else:
            _reason()
        for cls in all_classes:
            for anc in cls.ancestors():
                if isinstance(anc, ThingClass) and anc != cls and anc != owl.Thing and anc in all_classes:
                    inferred.add((cls, 'subclassOf', anc))
            for eq in cls.equivalent_to:
                if isinstance(eq, ThingClass) and eq in all_classes:
                    inferred.add((cls, 'equivalentTo', eq))
    else:
        logging.info(
            "%s - Skip full reasoner for %d classes; using hierarchy closure only",
            file_path,
            original_class_count,
        )
        for cls in all_classes:
            for anc in cls.ancestors():
                if isinstance(anc, ThingClass) and anc != cls and anc != owl.Thing and anc in all_classes:
                    inferred.add((cls, 'subclassOf', anc))
            for eq in cls.equivalent_to:
                if isinstance(eq, ThingClass) and eq in all_classes:
                    inferred.add((cls, 'equivalentTo', eq))

    for cls in all_classes:
        for restriction in cls.is_a:
            if isinstance(restriction, Restriction):
                prop = restriction.property
                name = getattr(prop, 'python_name', None)
                if not name or name not in relations:
                    continue
                filler = getattr(restriction, 'value', None) or \
                         getattr(restriction, 'some_values_from', None) or \
                         getattr(restriction, 'all_values_from', None)
                if isinstance(filler, ThingClass) and filler in all_classes:
                    inferred.add((cls, name, filler))

    implicit_all = list(inferred - explicit)
    if not implicit_all:
        logging.info(f"{file_path} - No implicit triples, skipping")
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_implicit_class_relation_triples",
            extra={"classes": len(all_classes), "explicit_triples": len(explicit)},
        )
        return

    # concept-scope on subject side
    if concept_scope != 'all':
        def is_native(c):
            return getattr(getattr(c, 'namespace', None), 'ontology', None) is onto
        def is_imported(c):
            o = getattr(getattr(c, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            implicit_all = [t for t in implicit_all if is_native(t[0])]
        else:
            implicit_all = [t for t in implicit_all if is_imported(t[0])]
        if not implicit_all:
            logging.info(f"{file_path} - No triples after concept-scope filtering")
            save_empty_marker(
                empty_path,
                source_file=file_path,
                reason="no_implicit_class_relation_triples_after_scope_filter",
                extra={"classes": len(all_classes), "explicit_triples": len(explicit)},
            )
            return

    implicit = random.sample(implicit_all, min(MAX_TRIPLES, len(implicit_all)))

    gen = RelationQuestionGenerator(implicit, all_classes, inferred=True)
    try:
        gen.all_true_triples = explicit | set(implicit)
    except Exception:
        gen.all_true_triples = set(implicit)
    questions = gen.generate_all(max_questions)

    if questions:
        save_questions(questions, save_path)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_inferred_relation_questions",
            extra={"implicit_triples": len(implicit)},
        )
    label_cache.clear(); depth_cache.clear(); ancestors_cache.clear()

def main():
    parser = argparse.ArgumentParser(description='Generate reasoning-based class relation MCQs (implicit triples).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=30000, help='Max questions per ontology.')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of subject classes: all/native/imported.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library stdout/stderr noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    configure_logging(args.log, "process_2_1.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')
    files, input_root = discover_ontology_files(input_path, exts)
    logging.info(f"Found {len(files)} files to process")
    onto_paths = resolve_onto_paths(args.onto_path)
    for fp in files:
        try:
            with file_timeout(args.file_timeout_seconds):
                process_owl_file(
                    file_path=fp,
                    input_root=input_root,
                    output_root=output_root,
                    max_questions=args.max_questions,
                    load_imports=not args.no_imports,
                    onto_paths=onto_paths,
                    no_warnings=args.no_warnings,
                    concept_scope=args.concept_scope,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()
