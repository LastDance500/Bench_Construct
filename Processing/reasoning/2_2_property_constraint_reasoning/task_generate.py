import json
import os
import random
import logging
import argparse
import warnings
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import World, ThingClass, ObjectPropertyClass, Restriction, owl, sync_reasoner, onto_path, set_log_level
from collections import OrderedDict
from functools import lru_cache
from rdflib import Graph, RDF, OWL, URIRef


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    class_depth,
    configure_world_paths,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    get_label,
    limit_questions_by_subject,
    load_ontology,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)

# Defaults
NUM_CHOICES = 4
MAX_CACHE_SIZE = 10000
MAX_CHAINS = 1000
MAX_DISTRACTOR_CANDIDATES = 500
MAX_SUBCLASSES_PER_DOMAIN = 1000

PROPP_EXCLUDED_LABELS = {
    'Linking from AaTh-Numbers to ATU-Numbers',
    'Linking back to the ATU source',
    'eTrap motif',
    'eTRAP motif',
    'eTRAP added motif',
}


def normalize_label(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text or '')
    text = text.replace('_', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', text).strip().lower()


def label_tokens(text: str) -> set[str]:
    return {token for token in re.split(r'[^a-z0-9]+', normalize_label(text)) if len(token) >= 3}


def is_blacklisted_label(text: str) -> bool:
    normalized = normalize_label(text)
    return any(
        normalized == normalize_label(label) or normalize_label(label) in normalized
        for label in PROPP_EXCLUDED_LABELS
    )


def is_valid_named_entity(entity) -> bool:
    label = get_label(entity)
    return bool(label and label != 'Unnamed' and not is_blacklisted_label(label))


def property_leaks_answer(prop_label: str, answer_label: str) -> bool:
    answer = label_tokens(answer_label)
    if not answer:
        return False
    prop = label_tokens(prop_label)
    return bool(answer & prop)


def entity_key(entity) -> str:
    return str(getattr(entity, "iri", entity))


def ontology_property_index(onto) -> dict[str, ObjectPropertyClass]:
    index = {}
    for prop in onto.object_properties():
        if isinstance(prop, ObjectPropertyClass):
            index[str(prop.iri)] = prop
    return index

# ---------- class depth ----------
@lru_cache(maxsize=10000)
def compute_depth(entity):
    depth = class_depth(entity)
    return depth if depth is not None else float('inf')

# ---------- extract property domain/range/subclass tuples ----------
def extract_property_info(onto):
    triples = []
    for prop in onto.object_properties():
        if not isinstance(prop, ObjectPropertyClass):
            continue
        if not is_valid_named_entity(prop):
            continue
        domains = [d for d in (getattr(prop, 'domain', []) or []) if isinstance(d, ThingClass)]
        ranges = [r for r in (getattr(prop, 'range', []) or []) if isinstance(r, ThingClass)]
        if not domains or not ranges:
            continue
        for domain in domains:
            for range_cls in ranges:
                if not (is_valid_named_entity(domain) and is_valid_named_entity(range_cls)):
                    continue
                subclasses = [sc for sc in domain.subclasses() if isinstance(sc, ThingClass) and sc != domain]
                if len(subclasses) > MAX_SUBCLASSES_PER_DOMAIN:
                    random.shuffle(subclasses)
                    subclasses = subclasses[:MAX_SUBCLASSES_PER_DOMAIN]
                for subclass in subclasses:
                    if not is_valid_named_entity(subclass):
                        continue
                    triples.append((prop, domain, range_cls, subclass))
    logging.info(f"Extracted {len(triples)} property-domain-range-subclass triples")
    return triples

# ---------- extract property chains ----------
def parse_rdf_list(graph: Graph, list_node) -> list:
    items = []
    seen = set()
    while list_node and list_node != RDF.nil and list_node not in seen:
        seen.add(list_node)
        first = graph.value(list_node, RDF.first)
        rest = graph.value(list_node, RDF.rest)
        if first is None:
            break
        items.append(first)
        list_node = rest
    return items


def parse_property_chains_rdflib(file_path: Path, prop_index: dict[str, ObjectPropertyClass]):
    chains = []
    graph = Graph()
    try:
        graph.parse(str(file_path))
    except Exception as exc:
        logging.debug("RDFLib failed parsing property chains from %s: %s", file_path, exc)
        return chains

    for super_prop_node, _, list_node in graph.triples((None, OWL.propertyChainAxiom, None)):
        if not isinstance(super_prop_node, URIRef):
            continue
        super_prop = prop_index.get(str(super_prop_node))
        if not super_prop or not is_valid_named_entity(super_prop):
            continue
        chain_nodes = parse_rdf_list(graph, list_node)
        chain_props = []
        for node in chain_nodes:
            if not isinstance(node, URIRef):
                continue
            prop = prop_index.get(str(node))
            if prop and is_valid_named_entity(prop):
                chain_props.append(prop)
        if len(chain_props) >= 2:
            chains.append({
                "chain": tuple(chain_props),
                "super_property": super_prop,
                "source_axiom": "owl:propertyChainAxiom",
                "extraction_backend": "rdflib_rdf_list",
            })
            if len(chains) >= MAX_CHAINS:
                break
    logging.info("RDFLib extracted %d explicit property-chain axioms", len(chains))
    return chains


def extract_property_chain_info(onto, file_path: Optional[Path] = None):
    chains = []
    for super_prop in onto.object_properties():
        if not isinstance(super_prop, ObjectPropertyClass):
            continue
        if not is_valid_named_entity(super_prop):
            continue
        raw_values = []
        try:
            value = getattr(super_prop, "property_chain", []) or []
            if value:
                raw_values.append(value)
        except Exception:
            pass
        try:
            value = super_prop.get_property_chain() or []
            if value:
                raw_values.append(value)
        except Exception:
            pass
        if not raw_values:
            continue

        for raw_chain in raw_values:
            try:
                raw_items = list(raw_chain)
            except TypeError:
                raw_items = [raw_chain]
            if raw_items and all(isinstance(item, ObjectPropertyClass) for item in raw_items):
                candidate_chains = [raw_items]
            else:
                candidate_chains = []
                for item in raw_items:
                    try:
                        candidate_chains.append(list(item))
                    except TypeError:
                        candidate_chains.append([item])
            for candidate_chain in candidate_chains:
                chain_props = [
                    prop for prop in candidate_chain
                    if isinstance(prop, ObjectPropertyClass) and is_valid_named_entity(prop)
                ]
                if len(chain_props) < 2:
                    continue
                chains.append({
                    "chain": tuple(chain_props),
                    "super_property": super_prop,
                    "source_axiom": "owl:propertyChainAxiom",
                    "extraction_backend": "owlready2_native",
                })
                if len(chains) >= MAX_CHAINS:
                    logging.info("Reached max chains limit: %d", MAX_CHAINS)
                    return chains
    if file_path is not None and len(chains) < MAX_CHAINS:
        chains.extend(parse_property_chains_rdflib(file_path, ontology_property_index(onto)))

    deduped = []
    seen = set()
    for item in chains:
        key = (
            tuple(entity_key(prop) for prop in item["chain"]),
            entity_key(item["super_property"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= MAX_CHAINS:
            break
    logging.info("Extracted %d explicit property-chain axioms after backend merge", len(deduped))
    return deduped

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
            if not is_valid_named_entity(prop) or not is_valid_named_entity(cls):
                continue
            filler = None
            # someValuesFrom
            if getattr(constraint, 'type', None) == owl.some:
                filler = getattr(constraint, 'some_values_from', None)
            # hasValue
            elif getattr(constraint, 'type', None) == owl.value:
                filler = getattr(constraint, 'value', None)
            if isinstance(filler, ThingClass):
                if not is_valid_named_entity(filler):
                    continue
                existentials.append((cls, prop, filler))
    logging.info(f"Extracted {len(existentials)} existential restrictions")
    return existentials

# ---------- distractors ----------
def get_distractors(correct, all_classes, exclude=None):
    candidates = [c for c in all_classes if c != correct and isinstance(c, ThingClass) and is_valid_named_entity(c)]
    if exclude:
        candidates = [c for c in candidates if c not in exclude]
    if len(candidates) > MAX_DISTRACTOR_CANDIDATES:
        random.shuffle(candidates)
        candidates = candidates[:MAX_DISTRACTOR_CANDIDATES]
    candidates = sorted(candidates, key=lambda c: abs(compute_depth(c) - compute_depth(correct)))
    if len(candidates) <= NUM_CHOICES - 1:
        return candidates
    nearby = candidates[: max(10, NUM_CHOICES - 1)]
    return random.sample(nearby, NUM_CHOICES - 1)

# ---------- question: range ----------
class RangeQuestionGenerator:
    def __init__(self, triples, all_classes):
        self.triples = triples
        self.all_classes = all_classes

    def generate_one(self, prop, domain, range_cls, subclass):
        answer_leak_risk = property_leaks_answer(get_label(prop), get_label(range_cls))
        # Exclude: domain and its subclasses; true ranges for (prop, subclass)
        exclude = set([domain] + list(domain.subclasses()))
        true_ranges = {r for (p, d, r, sc) in self.triples if p == prop and sc == subclass and isinstance(r, ThingClass)}
        exclude.update(true_ranges)
        distractors = get_distractors(range_cls, self.all_classes, exclude=list(exclude))
        if len(distractors) < NUM_CHOICES - 1:
            true_values_for_prop = {
                r for (p, _, r, _) in self.triples
                if p == prop and isinstance(r, ThingClass)
            }
            distractors = get_distractors(range_cls, self.all_classes, exclude=list(true_values_for_prop))
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [range_cls] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == range_cls)
        prompt = (
            f"The property '{get_label(prop)}' applies to instances of '{get_label(domain)}', "
            f"and '{get_label(subclass)}' is a subclass of '{get_label(domain)}'. "
            f"Which class can appear as the value of '{get_label(prop)}' for instances of '{get_label(subclass)}'?"
        )
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(prop.iri),
                'subject_label': get_label(prop),
                'subject_kind': 'property',
                'relation': 'property_range_inheritance',
                'constraint_kind': 'domain_range_inheritance',
                'source_axiom': 'rdfs:domain/rdfs:range + rdfs:subClassOf',
                'extraction_backend': 'owlready2_native',
                'answer_leak_risk': answer_leak_risk,
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
    def __init__(self, chains, all_properties):
        self.chains = chains
        self.all_properties = [
            prop for prop in all_properties
            if isinstance(prop, ObjectPropertyClass) and is_valid_named_entity(prop)
        ]

    def generate_one(self, chain_info):
        chain_props = chain_info["chain"]
        super_prop = chain_info["super_property"]
        if len(chain_props) < 2:
            return None
        chain_labels = [get_label(prop) for prop in chain_props]
        answer_leak_risk = any(label_tokens(label) & label_tokens(get_label(super_prop)) for label in chain_labels)
        true_super_props = {
            candidate["super_property"]
            for candidate in self.chains
            if tuple(candidate["chain"]) == tuple(chain_props)
        }
        candidates = [
            prop for prop in self.all_properties
            if prop not in true_super_props and prop is not super_prop
        ]
        random.shuffle(candidates)
        distractors = candidates[:NUM_CHOICES - 1]
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [super_prop] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(prop)} for l, prop in zip(letters, options)]
        correct = next(l for l, prop in zip(letters, options) if prop == super_prop)
        chain_text = " ◦ ".join(chain_labels)
        prompt = (
            f"The ontology states an object-property chain: {chain_text}. "
            "Which property is inferred between the start and end individuals?"
        )
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(chain_props[0].iri),
                'subject_label': chain_text,
                'subject_kind': 'property_chain',
                'relation': 'property_chain_axiom',
                'constraint_kind': 'property_chain',
                'source_axiom': chain_info.get('source_axiom', 'owl:propertyChainAxiom'),
                'extraction_backend': chain_info.get('extraction_backend', 'unknown'),
                'answer_leak_risk': answer_leak_risk,
                'object_iri': str(super_prop.iri),
                'object_label': get_label(super_prop),
                'object_kind': 'property',
                'class_context_iri': None,
                'class_context_label': None,
                'depth': None,
                'sibling_count': None,
                'subclass_count': None,
                'parent_count': None,
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.chains)
        for chain_info in self.chains[:max_q or len(self.chains)]:
            q = self.generate_one(chain_info)
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
        answer_leak_risk = property_leaks_answer(get_label(prop), get_label(filler))
        prompt = (
            f"The class '{get_label(cls)}' has an existential restriction using the property '{get_label(prop)}'. "
            f"Which class must its instances be related to via '{get_label(prop)}'?"
        )
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(cls.iri),
                'subject_label': get_label(cls),
                'subject_kind': 'class',
                'relation': 'existential_restriction',
                'constraint_kind': 'existential_restriction',
                'source_axiom': 'owl:someValuesFrom/owl:hasValue restriction',
                'extraction_backend': 'owlready2_native',
                'answer_leak_risk': answer_leak_risk,
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
def save_questions(questions, save_path: Path):
    save_json(questions, save_path, description="questions")


def save_composition_summary(questions, save_path: Path):
    composition = {}
    backend_counts = {}
    leak_risk_count = 0
    for question in questions:
        meta = question.get("meta", {})
        kind = meta.get("constraint_kind", "unknown")
        backend = meta.get("extraction_backend", "unknown")
        composition[kind] = composition.get(kind, 0) + 1
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if meta.get("answer_leak_risk"):
            leak_risk_count += 1
    total = len(questions)
    summary = {
        "total": total,
        "constraint_kind_counts": composition,
        "constraint_kind_percent": {
            kind: round(count / total, 4) if total else 0.0
            for kind, count in composition.items()
        },
        "extraction_backend_counts": backend_counts,
        "answer_leak_risk_count": leak_risk_count,
    }
    save_json(summary, save_path, description="R2 composition summary")

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
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f'property_inheritance_{safe_stem}.json'
    empty_path = empty_marker_path(out_dir, "property_inheritance", safe_stem)
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

    triples = extract_property_info(onto)
    chains = extract_property_chain_info(onto, file_path)
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
        chains = [chain for chain in chains if keep_prop(chain["super_property"])]
        existentials = [(c, p, f) for (c, p, f) in existentials if keep_class(c)]

    all_classes = [c for c in onto.classes() if isinstance(c, ThingClass)]
    if not all_classes:
        save_empty_marker(empty_path, source_file=file_path, reason="no_classes_for_property_constraint_reasoning")
        return

    q1 = RangeQuestionGenerator(triples, all_classes).generate_all(max_questions)
    q2 = ChainQuestionGenerator(chains, list(onto.object_properties())).generate_all(max_questions)
    q3 = ExistentialQuestionGenerator(existentials, all_classes).generate_all(max_questions)
    questions = q1 + q2 + q3
    random.shuffle(questions)
    questions = limit_questions_by_subject(questions, max_questions)
    composition = {}
    for question in questions:
        kind = question.get("meta", {}).get("constraint_kind", "unknown")
        composition[kind] = composition.get(kind, 0) + 1
    logging.info("R2 composition for %s: %s", file_path, composition)

    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    if questions:
        save_questions(questions, save_path)
        save_composition_summary(questions, out_dir / f'property_inheritance_{safe_stem}_summary.json')
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_property_constraint_questions",
            extra={"domain_range_triples": len(triples), "property_chains": len(chains), "existentials": len(existentials)},
        )

def main():
    parser = argparse.ArgumentParser(description='Generate property constraint reasoning MCQs (domain/range inheritance, chains, existential).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin (properties/classes) on subject side.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    configure_logging(args.log, "process_2_2.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')
    files, input_root = discover_ontology_files(input_path, exts)
    max_q = None if args.max_questions == 0 else args.max_questions
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
                    concept_scope=args.concept_scope,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()
