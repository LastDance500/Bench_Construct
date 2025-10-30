import json
import os
import random
import logging
import argparse
import warnings
import re
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import World, owl, ThingClass, onto_path, set_log_level

# Global caches for definitions and labels
definition_cache = {}
label_cache = {}

def get_definition(entity):
    """
    Retrieve the definition of an entity with preference for English definition.
    Returns a string definition or "No definition provided" if none exists.
    """
    try:
        key = str(entity.iri)
        if key in definition_cache:
            return definition_cache[key]

        definition = None
        # 1. Try IAO_0000115
        defs = getattr(entity, "IAO_0000115", None)
        if defs and len(defs) > 0:
            definition = next((d for d in defs if getattr(d, 'lang', None) == 'en'), defs[0])

        # 2. skos:definition
        if not definition:
            skos_defs = getattr(entity, "definition", None)
            if skos_defs and len(skos_defs) > 0:
                definition = next((d for d in skos_defs if getattr(d, 'lang', None) == 'en'), skos_defs[0])

        # 3. rdfs:comment
        if not definition:
            comment = getattr(entity, "comment", None)
            if comment:
                if isinstance(comment, list) and len(comment) > 0:
                    definition = next((d for d in comment if getattr(d, 'lang', None) == 'en'), comment[0])
                elif isinstance(comment, str):
                    definition = comment

        # Convert to string and ensure it's not empty
        definition = str(definition) if definition and str(definition).strip() else "No definition provided."
        definition_cache[key] = definition
        return definition
    except Exception as e:
        logging.warning(f"Error retrieving definition for {getattr(entity, 'name', 'unknown')}: {e}")
        return "No definition provided."

def get_label(entity):
    """
    Retrieve the label for an entity, preferring rdfs:label, otherwise using entity name.
    Returns a string label.
    """
    try:
        key = str(entity.iri)
        if key in label_cache:
            return label_cache[key]
        label = getattr(entity, "label", None)
        result = str(label[0]) if label and len(label) > 0 and isinstance(label, list) else str(entity.name)
        label_cache[key] = result
        return result
    except Exception as e:
        logging.warning(f"Error retrieving label for {getattr(entity, 'name', 'unknown')}: {e}")
        return str(entity.name)

def compute_depth(entity, memo=None, visiting=None):
    """
    Compute the depth of a concept in the ontology hierarchy from owl.Thing.
    Uses memoization to avoid redundant calculations.
    """
    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()
    if entity in memo:
        return memo[entity]
    if entity in visiting:
        # Cycle detected; treat as minimal additional depth to break recursion
        return 1
    if entity == owl.Thing:
        memo[entity] = 0
        return 0
    visiting.add(entity)
    parents = [p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing]
    depth = 1 if not parents else max((compute_depth(p, memo, visiting) for p in parents), default=1) + 1
    visiting.discard(entity)
    memo[entity] = depth
    return depth

def get_siblings(entity):
    """
    Get all sibling concepts (subclasses of the same parent(s)).
    """
    siblings = set()
    for parent in entity.is_a:
        if isinstance(parent, ThingClass) and parent != owl.Thing:
            siblings.update(parent.subclasses())
    siblings.discard(entity)
    return siblings

def compute_global_metrics(classes):
    """
    Compute global stats for normalization: max depth, sibling, subclass, parent counts.
    """
    max_depth = max_sibling = max_subclass = max_parent = 0
    for cls in classes:
        try:
            d = compute_depth(cls)
            s = len(get_siblings(cls))
            sub = len(list(cls.subclasses()))
            par = len([p for p in cls.is_a if isinstance(p, ThingClass) and p != owl.Thing])
            max_depth = max(max_depth, d)
            max_sibling = max(max_sibling, s)
            max_subclass = max(max_subclass, sub)
            max_parent = max(max_parent, par)
        except Exception as e:
            logging.warning(f"Error computing metrics for {getattr(cls, 'name', 'unknown')}: {e}")
    return {
        "max_depth": max(max_depth, 1),  # Avoid division by zero
        "max_sibling_count": max(max_sibling, 1),
        "max_subclass_count": max(max_subclass, 1),
        "max_parent_count": max(max_parent, 1)
    }

def compute_selection_weight(entity, global_metrics):
    """
    Compute selection weight for an entity based on normalized metrics.
    Weight = (norm_depth * (norm_sibling + 1)) / (norm_subclass + norm_parent + 1)
    """
    try:
        depth = compute_depth(entity)
        sib = len(get_siblings(entity))
        sub = len(list(entity.subclasses()))
        par = len([p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing])

        gm = global_metrics
        nd = depth / gm["max_depth"]
        ns = sib / gm["max_sibling_count"]
        nc = sub / gm["max_subclass_count"]
        np = par / gm["max_parent_count"]

        return nd * (ns + 1) / (nc + np + 1)
    except Exception as e:
        logging.warning(f"Error computing weight for {getattr(entity, 'name', 'unknown')}: {e}")
        return 0.0

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


class OntologyLoader:
    """
    Load an OWL ontology in an isolated World to avoid conflicts.
    """
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
        """
        Load the ontology from the specified file path.
        Returns the loaded ontology or None if loading fails.
        """
        logging.info(f"Loading ontology: {self.file_path}")
        ontology_ref = f"file://{self.file_path.resolve()}"
        onto = self.world.get_ontology(ontology_ref)
        try:
            if self.load_imports:
                onto.load()
            else:
                onto.load(only_local=True)
        except Exception as e:
            if self.load_imports:
                logging.warning(f"Failed loading with imports; retrying local-only: {self.file_path} ({e})")
                try:
                    onto.load(only_local=True)
                except Exception as e2:
                    logging.error(f"Failed local-only: {self.file_path} ({e2})")
                    return None
            else:
                logging.error(f"Failed local-only: {self.file_path} ({e})")
                return None
        self.onto = onto
        return onto

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
        return [cls for cls in self.onto.classes() if cls != owl.Thing and get_definition(cls) != "No definition provided."]

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
            depth = compute_depth(target)
            sibling_count = len(get_siblings(target))
            subclass_count = len(list(target.subclasses()))
            parent_count = len([p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing])

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
    """
    Save questions to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved {len(questions)} questions to {save_path}")
    except Exception as e:
        logging.error(f"Error saving questions to {save_path}: {e}")

def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    concept_scope: str,
    max_questions: int,
) -> None:
    """Process a single OWL file, generate questions, and save them."""
    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    save_path = out_dir / f"class_definitions_{safe_stem}.json"
    if save_path.exists():
        logging.info(f"Skip existing: {save_path}")
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
        return
    global_metrics = compute_global_metrics(classes)
    gen = DefinitionQuestionGenerator(classes, global_metrics, max_questions=max_questions, mask_concept=True)
    questions = gen.generate_all_questions()
    logging.info(f"Generated {len(questions)} questions for {file_path}")
    if questions:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_questions(questions, str(save_path))
    # Clear caches
    definition_cache.clear()
    label_cache.clear()

def main():
    parser = argparse.ArgumentParser(description='Generate open-ended class definition questions from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=100, help='Max questions per ontology.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of classes.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()
    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('process_3_1.log','w','utf-8')])
    if args.no_warnings:
        try:
            set_log_level(0)
        except Exception:
            pass
        warnings.filterwarnings('ignore')

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl','.rdf','.rdfs','.ttl')
    files: List[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in exts:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, filenames in os.walk(str(input_path)):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    files.append(Path(root)/fname)

    logging.info(f"Found {len(files)} files.")
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                load_imports=not args.no_imports,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                concept_scope=args.concept_scope,
                max_questions=args.max_questions,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == "__main__":
    main()