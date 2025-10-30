import os
import json
import random
import logging
import argparse
import warnings
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
from contextlib import ExitStack, redirect_stdout, redirect_stderr

from rdflib import URIRef, RDF
from owlready2 import World, ThingClass, Restriction, owl, onto_path, set_log_level
from collections import defaultdict
from itertools import islice


# Caches
label_cache: Dict[str, str] = {}


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


def get_label(entity) -> str:
    key = str(entity.iri)
    if key not in label_cache:
        labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
        label_cache[key] = labs[0] if labs else entity.name
    return label_cache[key]


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
        obj_anc = self.meta[obj]['ancestors']
        distractors: List[ThingClass] = []
        candidates = list(self.disjoint_sets[obj])
        random.shuffle(candidates)
        for c in candidates:
            if c is not obj and len(distractors) < k:
                distractors.append(c)
        if len(distractors) < k:
            remaining = [c for c in self.all_classes if c is not obj and c not in obj_anc and c not in distractors]
            random.shuffle(remaining)
            distractors.extend(remaining[:k - len(distractors)])
        return distractors[:k]

    def generate_all(self, max_q: int) -> List[Dict]:
        questions = []
        letters = ['A', 'B', 'C', 'D']
        for subj, rel, obj in islice(self.triples, max_q):
            m = self.meta[subj]
            distractors = self._get_distractors(obj, 3)
            options = [obj] + distractors
            random.shuffle(options)
            opts = []
            correct = None
            for i, choice in enumerate(options):
                opts.append({'option_letter': letters[i], 'label': get_label(choice)})
                if choice is obj:
                    correct = letters[i]
            prompt = make_prompt(get_label(subj), rel)
            questions.append({
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
                    'depth': m['depth'],
                    'sibling_count': m['siblings'],
                    'subclass_count': m['subclasses'],
                    'parent_count': m['parents'],
                }
            })
            if len(questions) >= max_q:
                break
        return questions


def compute_ancestors(parent_map: Dict[ThingClass, List[ThingClass]], all_classes: Iterable[ThingClass]) -> Dict[ThingClass, set]:
    ancestors_map: Dict[ThingClass, set] = {c: set() for c in all_classes}

    def get_ancestors(c: ThingClass):
        if ancestors_map[c]:
            return ancestors_map[c]
        anc = set()
        for p in parent_map[c]:
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


class OntologyLoader:
    def __init__(self, file_path: Path, load_imports: bool = True, onto_paths: Optional[List[Path]] = None):
        self.file_path = Path(file_path)
        self.world = World()
        self.onto = None
        self.load_imports = load_imports
        if onto_paths:
            for p in onto_paths:
                try:
                    pp = str(Path(p).resolve())
                    if pp not in self.world._ontology_path:
                        self.world._ontology_path.append(pp)
                    if pp not in onto_path:
                        onto_path.append(pp)
                except Exception:
                    pass

    def load(self):
        iri = f"file://{self.file_path.resolve()}"
        onto = self.world.get_ontology(iri)
        try:
            if self.load_imports:
                onto.load()
            else:
                onto.load(only_local=True)
        except Exception as e:
            if self.load_imports:
                logging.warning(f"Failed loading ontology with imports; retrying local-only. File: {self.file_path} ({e})")
                try:
                    onto.load(only_local=True)
                except Exception as e2:
                    logging.error(f"Failed loading ontology local-only: {self.file_path} ({e2})")
                    return None
            else:
                logging.error(f"Failed loading ontology local-only: {self.file_path} ({e})")
                return None
        self.onto = onto
        return onto


def save_questions(questions: List[Dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


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
    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    out_file = out_dir / f"relations_{safe_stem}.json"

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


def main():
    parser = argparse.ArgumentParser(description='Generate class-to-class relation MCQs from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory for Windows-safe mirrored folders and JSON.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=30000, help='Maximum questions per ontology.')
    parser.add_argument('--concept-scope', type=str, choices=['all', 'native', 'imported'], default='all', help='Filter by origin of subject classes: all/native/imported.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library stdout/stderr noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('process.log', 'w', 'utf-8'),
        ],
    )

    if args.no_warnings:
        try:
            set_log_level(0)
        except Exception:
            pass
        warnings.filterwarnings('ignore')
        for name in ('owlready2', 'rdflib'):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass

    random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')

    files: List[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in exts:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, filenames in os.walk(str(input_path)):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    files.append(Path(root) / fname)

    logging.info(f"Found {len(files)} files to process.")
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                max_questions=args.max_questions,
                load_imports=not args.no_imports,
                concept_scope=args.concept_scope,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                suppress_io=args.no_warnings,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")


if __name__ == '__main__':
    main()