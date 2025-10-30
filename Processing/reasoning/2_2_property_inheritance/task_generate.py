import json
import os
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import World, ThingClass, ObjectPropertyClass, Restriction, owl, sync_reasoner, onto_path, set_log_level
from collections import OrderedDict
from functools import lru_cache

# Defaults
NUM_CHOICES = 4
MAX_CACHE_SIZE = 10000
MAX_CHAINS = 1000

# ---------- 缓存 ----------
label_cache = OrderedDict()

# ---------- label helper ----------
def get_label(entity):
    key = str(getattr(entity, 'iri', str(entity)))
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else getattr(entity, 'name', str(entity))
    label_cache[key] = label
    if len(label_cache) > MAX_CACHE_SIZE:
        label_cache.popitem(last=False)
    return label

# ---------- class depth ----------
@lru_cache(maxsize=10000)
def compute_depth(entity):
    if not isinstance(entity, ThingClass):
        return float('inf')
    queue = [(entity, 0)]
    visited = {entity}
    while queue:
        current, dist = queue[0]
        queue = queue[1:]
        if current == owl.Thing:
            return dist
        for parent in (current.is_a or []):
            if not isinstance(parent, ThingClass):
                continue
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, dist + 1))
    return float('inf')

# ---------- extract property domain/range/subclass tuples ----------
def extract_property_info(onto):
    triples = []
    for prop in onto.object_properties():
        if not isinstance(prop, ObjectPropertyClass):
            continue
        domains = [d for d in (getattr(prop, 'domain', []) or []) if isinstance(d, ThingClass)]
        ranges = [r for r in (getattr(prop, 'range', []) or []) if isinstance(r, ThingClass)]
        if not domains or not ranges:
            continue
        for domain in domains:
            for range_cls in ranges:
                subclasses = [sc for sc in domain.subclasses() if isinstance(sc, ThingClass) and sc != domain]
                for subclass in subclasses:
                    triples.append((prop, domain, range_cls, subclass))
    logging.info(f"Extracted {len(triples)} property-domain-range-subclass triples")
    return triples

# ---------- extract property chains ----------
def extract_property_chain_info(onto):
    chains = []
    props = list(onto.object_properties())
    for p1 in props:
        if not isinstance(p1, ObjectPropertyClass):
            continue
        d1_list = [d for d in (getattr(p1, 'domain', []) or []) if isinstance(d, ThingClass)]
        r1_list = [r for r in (getattr(p1, 'range', []) or []) if isinstance(r, ThingClass)]
        for p2 in props:
            if not isinstance(p2, ObjectPropertyClass):
                continue
            d2_list = [d for d in (getattr(p2, 'domain', []) or []) if isinstance(d, ThingClass)]
            r2_list = [r for r in (getattr(p2, 'range', []) or []) if isinstance(r, ThingClass)]
            for d1 in d1_list:
                for r1 in r1_list:
                    for d2 in d2_list:
                        if r1 != d2:
                            continue
                        for r2 in r2_list:
                            subclasses = [sc for sc in d1.subclasses() if isinstance(sc, ThingClass) and sc != d1]
                            for subclass in subclasses:
                                chains.append((p1, p2, d1, r1, r2, subclass))
                                if len(chains) >= MAX_CHAINS:
                                    logging.info(f"Reached max chains limit: {MAX_CHAINS}")
                                    return chains
    logging.info(f"Extracted {len(chains)} property chains")
    return chains

# ---------- extract existential restrictions ----------
def extract_existential_info(onto):
    existentials = []
    for cls in onto.classes():
        constraints = (getattr(cls, 'equivalent_to', []) or []) + (getattr(cls, 'is_a', []) or [])
        for constraint in constraints:
            if not isinstance(constraint, Restriction):
                continue
            prop = getattr(constraint, 'property', None)
            if not isinstance(prop, ObjectPropertyClass):
                continue
            filler = None
            # someValuesFrom
            if getattr(constraint, 'type', None) == owl.some:
                filler = getattr(constraint, 'some_values_from', None)
            # hasValue
            elif getattr(constraint, 'type', None) == owl.value:
                filler = getattr(constraint, 'value', None)
            if isinstance(filler, ThingClass):
                existentials.append((cls, prop, filler))
    logging.info(f"Extracted {len(existentials)} existential restrictions")
    return existentials

# ---------- distractors ----------
def get_distractors(correct, all_classes, exclude=None):
    candidates = [c for c in all_classes if c != correct and isinstance(c, ThingClass)]
    if exclude:
        candidates = [c for c in candidates if c not in exclude]
    candidates = sorted(candidates, key=lambda c: abs(compute_depth(c) - compute_depth(correct)))
    return random.sample(candidates[:10], min(NUM_CHOICES - 1, len(candidates)))

# ---------- question: range ----------
class RangeQuestionGenerator:
    def __init__(self, triples, all_classes):
        self.triples = triples
        self.all_classes = all_classes

    def generate_one(self, prop, domain, range_cls, subclass):
        # Exclude: domain and its subclasses; true ranges for (prop, subclass)
        exclude = set([domain] + list(domain.subclasses()))
        true_ranges = {r for (p, d, r, sc) in self.triples if p == prop and sc == subclass and isinstance(r, ThingClass)}
        exclude.update(true_ranges)
        distractors = get_distractors(range_cls, self.all_classes, exclude=list(exclude))
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [range_cls] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == range_cls)
        prompt = (f"Given that the property '{get_label(prop)}' has a domain of '{get_label(domain)}', "
                  f"and '{get_label(subclass)}' is a subclass of '{get_label(domain)}', "
                  f"what is the range of '{get_label(prop)}' when applied to '{get_label(subclass)}'?")
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(prop.iri),
                'subject_label': get_label(prop),
                'subject_kind': 'property',
                'relation': 'property_range_inheritance',
                'object_iri': str(range_cls.iri),
                'object_label': get_label(range_cls),
                'object_kind': 'class',
                'class_context_iri': str(subclass.iri),
                'class_context_label': get_label(subclass),
                'depth': None,
                'sibling_count': None,
                'subclass_count': None,
                'parent_count': None,
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.triples)
        for prop, domain, range_cls, subclass in self.triples[:max_q or len(self.triples)]:
            q = self.generate_one(prop, domain, range_cls, subclass)
            if q:
                questions.append(q)
        return questions

# ---------- question: property chain ----------
class ChainQuestionGenerator:
    def __init__(self, chains, all_classes):
        self.chains = chains
        self.all_classes = all_classes

    def generate_one(self, p1, p2, d1, mid, final, subclass):
        # Exclude true finals for the same (p1,p2,d1,subclass)
        true_finals = {fn for (pp1, pp2, dd1, mm, fn, sc) in self.chains if pp1 == p1 and pp2 == p2 and dd1 == d1 and sc == subclass and isinstance(fn, ThingClass)}
        distractors = get_distractors(final, self.all_classes, exclude=list(true_finals))
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [final] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == final)
        prompt = (f"Given the property chain '{get_label(p1)} ∘ {get_label(p2)}': "
                  f"{get_label(p1)}: '{get_label(d1)}' → '{get_label(mid)}', "
                  f"{get_label(p2)}: '{get_label(mid)}' → '{get_label(final)}', "
                  f"what class does the chain point to when applied to '{get_label(subclass)}' (a subclass of '{get_label(d1)}')?")
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(p1.iri),
                'subject_label': get_label(p1),
                'subject_kind': 'property',
                'relation': 'property_chain',
                'object_iri': str(final.iri),
                'object_label': get_label(final),
                'object_kind': 'class',
                'class_context_iri': str(subclass.iri),
                'class_context_label': get_label(subclass),
                'depth': None,
                'sibling_count': None,
                'subclass_count': None,
                'parent_count': None,
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.chains)
        for p1, p2, d1, mid, final, subclass in self.chains[:max_q or len(self.chains)]:
            q = self.generate_one(p1, p2, d1, mid, final, subclass)
            if q:
                questions.append(q)
        return questions

# ---------- question: existential ----------
class ExistentialQuestionGenerator:
    def __init__(self, existentials, all_classes):
        self.existentials = existentials
        self.all_classes = all_classes

    def generate_one(self, cls, prop, filler):
        # Exclude: true fillers for the same (cls, prop)
        true_fillers = {f for (c, p, f) in self.existentials if c == cls and p == prop and isinstance(f, ThingClass)}
        distractors = get_distractors(filler, self.all_classes, exclude=list(true_fillers))
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [filler] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == filler)
        prompt = (f"Given that the class '{get_label(cls)}' is defined as having some '{get_label(prop)}' "
                  f"pointing to instances of '{get_label(filler)}', "
                  f"what class must instances of '{get_label(cls)}' be related to via '{get_label(prop)}'?")
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(cls.iri),
                'subject_label': get_label(cls),
                'subject_kind': 'class',
                'relation': 'existential_restriction',
                'object_iri': str(filler.iri),
                'object_label': get_label(filler),
                'object_kind': 'class',
                'class_context_iri': str(cls.iri),
                'class_context_label': get_label(cls),
                'depth': None,
                'sibling_count': None,
                'subclass_count': None,
                'parent_count': None,
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.existentials)
        for cls, prop, filler in self.existentials[:max_q or len(self.existentials)]:
            q = self.generate_one(cls, prop, filler)
            if q:
                questions.append(q)
        return questions

# ---------- 保存 ----------
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


def save_questions(questions, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(questions)} questions to {save_path}")

# ---------- 主流程 ----------
def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    max_questions: Optional[int],
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    concept_scope: str,
) -> None:
    world = World()
    if onto_paths:
        for p in onto_paths:
            try:
                pp = str(Path(p).resolve())
                if pp not in world._ontology_path:
                    world._ontology_path.append(pp)
                if pp not in onto_path:
                    onto_path.append(pp)
            except Exception:
                pass
    iri = f"file://{file_path.resolve()}"
    onto = world.get_ontology(iri)
    try:
        if load_imports:
            onto.load()
        else:
            onto.load(only_local=True)
    except Exception as e:
        if load_imports:
            logging.warning(f"Failed loading with imports; retrying local-only: {file_path} ({e})")
            try:
                onto.load(only_local=True)
            except Exception as e2:
                logging.error(f"Failed local-only: {file_path} ({e2})")
                return
        else:
            logging.error(f"Failed local-only: {file_path} ({e})")
            return

    triples = extract_property_info(onto)
    chains = extract_property_chain_info(onto)
    existentials = extract_existential_info(onto)

    # Apply concept-scope on subjects:
    if concept_scope != 'all':
        def is_native(ent):
            return getattr(getattr(ent, 'namespace', None), 'ontology', None) is onto
        def is_imported(ent):
            o = getattr(getattr(ent, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        def keep_prop(p):
            return is_native(p) if concept_scope == 'native' else is_imported(p)
        def keep_class(c):
            return is_native(c) if concept_scope == 'native' else is_imported(c)
        triples = [(p, d, r, sc) for (p, d, r, sc) in triples if keep_prop(p)]
        chains = [(p1, p2, d1, mid, fin, sc) for (p1, p2, d1, mid, fin, sc) in chains if keep_prop(p1) and keep_prop(p2)]
        existentials = [(c, p, f) for (c, p, f) in existentials if keep_class(c)]

    all_classes = [c for c in onto.classes() if isinstance(c, ThingClass)]
    if not all_classes:
        return

    q1 = RangeQuestionGenerator(triples, all_classes).generate_all(max_questions)
    q2 = ChainQuestionGenerator(chains, all_classes).generate_all(max_questions)
    q3 = ExistentialQuestionGenerator(existentials, all_classes).generate_all(max_questions)
    questions = q1 + q2 + q3
    random.shuffle(questions)

    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    save_path = out_dir / f'property_inheritance_{safe_stem}.json'
    if questions:
        save_questions(questions, save_path)

def main():
    parser = argparse.ArgumentParser(description='Generate property inheritance MCQs (range, chains, existential).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin (properties/classes) on subject side.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('process_2_2.log', 'w', 'utf-8')])
    if args.no_warnings:
        try:
            set_log_level(0)
        except Exception:
            pass
        warnings.filterwarnings('ignore')
        for name in ('owlready2','rdflib'):
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
        for root, _, fnames in os.walk(str(input_path)):
            for f in fnames:
                if f.lower().endswith(exts):
                    files.append(Path(root) / f)
    max_q = None if args.max_questions == 0 else args.max_questions
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                max_questions=max_q,
                load_imports=not args.no_imports,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                concept_scope=args.concept_scope,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()