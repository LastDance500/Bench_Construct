import json
import os
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from owlready2 import World, ThingClass, owl, onto_path, set_log_level


# Config and caches
MIN_DISTRACTORS = 2
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


def get_label(entity):
    """Get rdfs:label or prefLabel, fallback to name/str(entity)."""
    key = str(getattr(entity, 'iri', str(entity)))
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else (entity.name if hasattr(entity, 'name') else str(entity))
    label_cache[key] = label
    return label

def compute_depth(entity, memo=None):
    if memo is None:
        memo = {}
    if entity in memo:
        return memo[entity]
    if entity == owl.Thing:
        memo[entity] = 0
        return 0
    parents = [p for p in entity.is_a if isinstance(p, ThingClass)]
    depth = 1 if not parents else max(compute_depth(p, memo) for p in parents) + 1
    memo[entity] = depth
    return depth

def get_siblings(entity):
    sibs = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass):
            sibs.update(p.subclasses())
    sibs.discard(entity)
    return sibs

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

"""DataProperty domain/range extraction and question generation"""

def extract_dataproperty_info(onto):
    """Extract all DataProperty domain and range information.
    Returns: list of (property, domain_class, range_datatype) and a mapping prop -> {domains, ranges}.
    """
    triples = []
    prop_info = {}
    data_props = list(onto.data_properties())
    logging.info(f"Found {len(data_props)} data properties in ontology")
    for prop in data_props:
        domains = set()
        ranges = set()
        # Domains
        for domain in prop.domain:
            if isinstance(domain, ThingClass):
                domains.add(domain)
        # Ranges
        for range_type in prop.range:
            ranges.add(range_type)
        if domains:  # 只处理有明确 domain 的属性
            prop_info[prop] = {'domains': domains, 'ranges': ranges}
            for domain in domains:
                for range_type in ranges:
                    triples.append((prop, domain, range_type))
    logging.info(f"Generated {len(triples)} (Property, Domain, Range) triples")
    return triples, prop_info

"""Question generation"""

class PropertyDomainRangeQuestionGenerator:
    def __init__(self, triples, prop_info, all_classes):
        self.triples = triples
        self.prop_info = prop_info
        self.all_classes = all_classes
        self.all_properties = list(prop_info.keys())

    def generate_one(self, prop, domain_cls, range_type, num_choices=4):
        try:
            # Metadata
            depth = compute_depth(domain_cls)
            siblings = len(get_siblings(domain_cls))
            subclasses = len(list(domain_cls.subclasses()))
            parents = len([p for p in domain_cls.is_a if isinstance(p, ThingClass)])

            # Decide question type randomly
            question_type = random.choice(['domain', 'range'])
            correct_answer = domain_cls if question_type == 'domain' else range_type

            # Distractors from classes with minimal relation to the correct answer
            distractors = []
            candidates = [c for c in self.all_classes if c != correct_answer]
            logging.debug(f"Generating {question_type} question for property {get_label(prop)}, candidates: {len(candidates)}")
            random.shuffle(candidates)
            for candidate in candidates:
                if question_type == 'domain':
                    if candidate not in self.prop_info[prop]['domains']:
                        distractors.append(candidate)
                else:  # range
                    if candidate not in self.prop_info[prop]['ranges']:
                        distractors.append(candidate)
                if len(distractors) >= num_choices - 1:
                    break

            # Fallback if not enough distractors
            if len(distractors) < num_choices - 1:
                logging.warning(f"Only {len(distractors)} distractors found for {get_label(prop)}, need {num_choices - 1}")
                for candidate in candidates:
                    if candidate != correct_answer and candidate not in distractors:
                        distractors.append(candidate)
                    if len(distractors) >= num_choices - 1:
                        break

            # Ensure a minimum number of distractors
            if len(distractors) < MIN_DISTRACTORS:
                logging.warning(f"Insufficient distractors ({len(distractors)}) for {get_label(prop)}, skipping question")
                return None

            options = [correct_answer] + distractors[:num_choices - 1]
            random.shuffle(options)

            letters = ['A', 'B', 'C', 'D'][:num_choices]
            opts = []
            correct = None
            for i, choice in enumerate(options):
                label = get_label(choice)
                opts.append({'option_letter': letters[i], 'label': label})
                if choice == correct_answer:
                    correct = letters[i]

            prompt = (f"Which of the following is a valid {question_type} for the data property '{get_label(prop)}'?")

            # Safe handling of range_type IRI and label
            range_iri = str(getattr(range_type, 'iri', str(range_type))) if range_type else 'N/A'
            range_label = get_label(range_type) if range_type else str(range_type)

            return {
                'prompt': prompt,
                'options': opts,
                'correct_answer': correct,
                'meta': {
                    'subject_iri': str(prop.iri),
                    'subject_label': get_label(prop),
                    'subject_kind': 'property',
                    'relation': 'property_domain' if question_type == 'domain' else 'property_range',
                    'object_iri': str(domain_cls.iri) if question_type == 'domain' else range_iri,
                    'object_label': get_label(domain_cls) if question_type == 'domain' else range_label,
                    'object_kind': 'class' if question_type == 'domain' else 'datatype',
                    'class_context_iri': str(domain_cls.iri) if question_type == 'domain' else None,
                    'class_context_label': get_label(domain_cls) if question_type == 'domain' else None,
                    'depth': depth if question_type == 'domain' else None,
                    'sibling_count': siblings if question_type == 'domain' else None,
                    'subclass_count': subclasses if question_type == 'domain' else None,
                    'parent_count': parents if question_type == 'domain' else None,
                }
            }
        except Exception as e:
            logging.error(f"Failed to generate question for property {prop}: {e}")
            return None

    def generate_all(self, max_q=None):
        questions = []
        for prop, domain, range_type in self.triples:
            q = self.generate_one(prop, domain, range_type)
            if q:
                questions.append(q)
                if max_q and len(questions) >= max_q:
                    logging.info(f"Reached MAX_QUESTIONS limit: {max_q}")
                    break
        logging.info(f"Generated {len(questions)} questions for this ontology")
        return questions

def save_questions(questions, save_path: Path):
    if not questions:
        logging.info(f"No questions generated for {save_path}, skipping save")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


def process_owl_file(file_path: Path, input_root: Path, output_root: Path, max_questions: Optional[int], load_imports: bool, onto_paths: Optional[List[Path]], suppress_warnings: bool, concept_scope: str) -> None:
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
            logging.warning(f"Failed loading with imports, retrying local-only: {file_path} ({e})")
            try:
                onto.load(only_local=True)
            except Exception as e2:
                logging.error(f"Failed loading ontology local-only: {file_path} ({e2})")
                return
        else:
            logging.error(f"Failed loading ontology local-only: {file_path} ({e})")
            return

    all_classes = list(onto.classes())
    triples, prop_info = extract_dataproperty_info(onto)
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
        triples = [t for t in triples if t[0] in allowed]
    gen = PropertyDomainRangeQuestionGenerator(triples, prop_info, all_classes)
    questions = gen.generate_all(max_questions)

    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    out_file = out_dir / f"property2domain_range_{safe_stem}.json"
    if questions:
        save_questions(questions, out_file)
    world.close()


def main():
    parser = argparse.ArgumentParser(description='Generate DataProperty domain/range MCQs from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory with Windows-safe mirrored structure.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--concept-scope', type=str, choices=['all', 'native', 'imported'], default='all', help='Filter by origin of properties: all/native/imported.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('process_1_3.log', 'w', 'utf-8')],
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
    max_q = None if args.max_questions == 0 else args.max_questions
    logging.info(f"Found {len(files)} ontology files")
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                max_questions=max_q,
                load_imports=not args.no_imports,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                suppress_warnings=args.no_warnings,
                concept_scope=args.concept_scope,
            )
        except Exception as e:
            logging.error(f"Processing {fp} failed: {e}")


if __name__ == '__main__':
    main()