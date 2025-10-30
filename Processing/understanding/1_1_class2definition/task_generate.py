import json
import os
import random
import logging
import argparse
import warnings
import sys
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Set
import re

from rdflib import URIRef, Literal
from owlready2 import World, ThingClass, owl, onto_path, set_log_level


# Caches to avoid repeated lookups
definition_cache: Dict[str, str] = {}
label_cache: Dict[str, str] = {}


def slugify_for_windows(name: str) -> str:
    """Create a Windows-safe slug for folder/file names.

    - Replace non-alphanumeric characters with underscore
    - Collapse repeated underscores
    - Trim leading/trailing underscores
    - Keep case to be readable
    """
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
    """Context manager to silence stdout/stderr when enabled."""
    if not enabled:
        # no-op context manager
        class _Noop:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        return _Noop()
    # Redirect both stdout and stderr to devnull-like sink
    null = _NullWriter()
    stack = ExitStack()
    stack.enter_context(redirect_stdout(null))
    stack.enter_context(redirect_stderr(null))
    return stack


def get_definition(entity) -> str:
    """Collect a best-effort English definition for an entity.

    Strategy:
    1) Direct properties: IAO_0000115, skos:definition (as `definition`), rdfs:comment, skos:prefLabel
    2) Scan world annotation properties whose local name contains "definition"
    3) Fallback to scanning RDF graph for predicates whose local name contains "definition"
    4) Final fallback to rdfs:label
    Prefer English literals if available; otherwise take the first found.
    """
    key = str(entity.iri)
    if key in definition_cache:
        return definition_cache[key]

    defs: List = []

    # 1) Common definition-bearing annotation properties
    defs.extend(getattr(entity, "IAO_0000115", []) or [])
    defs.extend(getattr(entity, "definition", []) or [])  # skos:definition in many ontologies

    comments = getattr(entity, "comment", []) or []
    if isinstance(comments, str):
        defs.append(comments)
    else:
        defs.extend(comments)

    defs.extend(getattr(entity, "prefLabel", []) or [])  # as a weak fallback source

    # 2) Scan all annotation properties containing "definition" in their local name
    world_obj = getattr(entity, "world", getattr(entity.namespace, "world", None))
    if world_obj:
        try:
            for ap in world_obj.annotation_properties():
                ap_iri = str(ap.iri)
                ap_local = ap_iri.split("#")[-1] if "#" in ap_iri else ap_iri.rsplit("/", 1)[-1]
                if "definition" in ap_local.lower():
                    vals = getattr(entity, ap.python_name, []) or []
                    if not isinstance(vals, (list, tuple)):
                        vals = [vals]
                    defs.extend(vals)
        except Exception as e:
            logging.debug(f"Failed scanning annotation properties for definitions: {e}")

        # 3) RDF graph predicate scan for *definition*-like properties
        try:
            graph = world_obj.as_rdflib_graph()
            subj = URIRef(entity.iri)
            for pred, obj in graph.predicate_objects(subj):
                pred_str = str(pred)
                local = pred_str.split('#')[-1] if '#' in pred_str else pred_str.rsplit('/', 1)[-1]
                if "definition" in local.lower():
                    defs.append(obj)
        except Exception as e:
            logging.debug(f"Failed scanning RDF graph for definitions: {e}")

    # 4) Final fallback to rdfs:label
    if not defs:
        labels = getattr(entity, "label", []) or []
        defs.extend(labels)

    # Prefer English literals
    definition: Optional[str] = None
    for d in defs:
        if getattr(d, "lang", None) == 'en':
            definition = str(d)
            break
    if not definition and defs:
        definition = str(defs[0])
    if not definition:
        definition = "No definition provided."

    definition_cache[key] = definition
    return definition


def get_label(entity) -> str:
    """Get a readable label for an entity. Prefer rdfs:label/prefLabel; fallback to `entity.name`."""
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    labels = getattr(entity, "label", []) or getattr(entity, "prefLabel", [])
    label = labels[0] if labels else entity.name
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
    parents = [p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing]
    depth = 1 if not parents else max(compute_depth(p, memo) for p in parents) + 1
    memo[entity] = depth
    return depth

def get_siblings(entity):
    sibs = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass) and p != owl.Thing:
            sibs.update(c for c in p.subclasses())
    sibs.discard(entity)
    return sibs

def compute_global_metrics(classes):
    max_depth = max_sib = max_sub = max_par = 0
    memo = {}
    for e in classes:
        d   = compute_depth(e, memo)
        s   = len(get_siblings(e))
        sub = len(list(e.subclasses()))
        par = len([p for p in e.is_a if isinstance(p, ThingClass) and p != owl.Thing])
        max_depth = max(max_depth, d)
        max_sib   = max(max_sib, s)
        max_sub   = max(max_sub, sub)
        max_par   = max(max_par, par)
    return dict(
        max_depth=max_depth,
        max_sibling_count=max_sib,
        max_subclass_count=max_sub,
        max_parent_count=max_par
    )

def compute_selection_weight(entity, gm):
    d   = compute_depth(entity)
    s   = len(get_siblings(entity))
    sub = len(list(entity.subclasses()))
    par = len([p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing])
    nd   = d   / gm["max_depth"]            if gm["max_depth"]            else 0
    ns   = s   / gm["max_sibling_count"]    if gm["max_sibling_count"]    else 0
    nsub = sub / gm["max_subclass_count"]   if gm["max_subclass_count"]   else 0
    npar = par / gm["max_parent_count"]     if gm["max_parent_count"]     else 0
    return nd * (ns + 1) / (nsub + npar + 1)

class OntologyLoader:
    def __init__(self, file_path: Path, load_imports: bool = True, onto_paths: Optional[List[Path]] = None):
        self.file_path = Path(file_path)
        self.world = World()
        self.onto = None
        self.load_imports = load_imports
        # Configure additional local search paths for resolving owl:imports
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
        """Load ontology with optional well-known annotation imports."""
        # Ensure common annotation ontologies are available
        for ont in (
            "http://purl.obolibrary.org/obo/iao.owl",
            "http://www.w3.org/2004/02/skos/core#",
        ):
            try:
                self.world.get_ontology(ont).load()
            except Exception as e:
                logging.debug(f"Failed loading annotation ontology {ont}: {e}")

        iri = f"file://{self.file_path.resolve()}"
        onto = self.world.get_ontology(iri)
        try:
            if self.load_imports:
                onto.load()
            else:
                onto.load(only_local=True)
        except Exception as e:
            if self.load_imports:
                logging.warning(
                    f"Failed loading ontology with imports; retrying local-only. File: {self.file_path} ({e})"
                )
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

    def preload_entities(self):
        """Touch common annotation attributes to warm caches and expand Python accessors."""
        if not self.onto:
            return
        for cls in self.onto.classes():
            _ = getattr(cls, "IAO_0000115", None)
            _ = getattr(cls, "definition", None)
            _ = getattr(cls, "comment", None)
            _ = getattr(cls, "label", None)
            _ = getattr(cls, "prefLabel", None)

    def get_all_classes_with_definition(self):
        if not self.onto:
            return []
        return [
            cls
            for cls in self.onto.classes()
            if cls != owl.Thing and get_definition(cls) != "No definition provided."
        ]

class QuestionGenerator:
    def __init__(self, classes: Iterable[ThingClass], mask_concept: bool = True):
        self.classes = list(classes)
        self.mask_concept = mask_concept

    def _collect_aliases(self, entity: ThingClass) -> List[str]:
        aliases: Set[str] = set()
        for lab in (getattr(entity, "label", []) or []) + (getattr(entity, "prefLabel", []) or []):
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
        variants: Set[str] = set()
        for a in list(aliases):
            s = a.replace('_', ' ').replace('-', ' ')
            variants.add(s)
        aliases |= variants
        return [a for a in aliases if len(a.strip()) >= 2]

    def _mask_definition_text(self, text: str, entity: ThingClass) -> str:
        if not self.mask_concept or not text:
            return text
        aliases = self._collect_aliases(entity)
        masked = text
        for alias in sorted(aliases, key=len, reverse=True):
            escaped = re.escape(alias)
            pattern = re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")
            masked = pattern.sub("the concept", masked)
        return masked

    def get_candidate_distractors(self, target: ThingClass) -> List[ThingClass]:
        cand = set()
        # Siblings (other subclasses of the same parent)
        for p in target.is_a:
            if isinstance(p, ThingClass) and p != owl.Thing:
                cand |= {
                    s for s in p.subclasses()
                    if s != target and get_definition(s) != "No definition provided."
                }
        # Parents
        cand |= {
            p for p in target.is_a
            if isinstance(p, ThingClass) and p != owl.Thing and get_definition(p) != "No definition provided."
        }
        # Children
        cand |= {
            c for c in target.subclasses()
            if c != target and get_definition(c) != "No definition provided."
        }
        # If fewer than 3 candidates, add random others as fallback
        if len(cand) < 3:
            others = [c for c in self.classes if c != target]
            random.shuffle(others)
            for o in others:
                if o not in cand:
                    cand.add(o)
                if len(cand) >= 3:
                    break
        cand.discard(target)
        return list(cand)

    def generate_question_for_target(self, target: ThingClass) -> Dict:
        d   = compute_depth(target)
        s   = len(get_siblings(target))
        sub = len(list(target.subclasses()))
        par = len([p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing])

        # Correct option
        defs = get_definition(target)
        lbl  = get_label(target)
        defs = self._mask_definition_text(defs, target)
        options = [{"label": lbl, "definition": defs, "is_correct": True}]
        # Distractors
        distractors = random.sample(self.get_candidate_distractors(target), 3)
        for dsc in distractors:
            options.append({
                "label": get_label(dsc),
                "definition": get_definition(dsc),
                "is_correct": False
            })
        random.shuffle(options)

        letters = ['A', 'B', 'C', 'D']
        opts = []
        correct = None
        for i, o in enumerate(options):
            opts.append({
                "option_letter": letters[i],
                "definition": o["definition"]
            })
            if o["is_correct"]:
                correct = letters[i]

        return {
            "prompt": f"Which of the following definitions best describes '{lbl}'?",
            "options": opts,
            "correct_answer": correct,
            "meta": {
                "subject_iri":          str(target.iri),
                "subject_label":        lbl,
                "subject_kind":         "class",
                "relation":             "class_definition",
                "object_iri":           None,
                "object_label":         None,
                "object_kind":          None,
                "class_context_iri":    str(target.iri),
                "class_context_label":  lbl,
                "depth":                d,
                "sibling_count":        s,
                "subclass_count":       sub,
                "parent_count":         par
            }
        }

    def generate_all_questions(self) -> Tuple[List[Dict], int]:
        if not self.classes:
            return [], 0
        gm = compute_global_metrics(self.classes)
        weights = [(e, compute_selection_weight(e, gm)) for e in self.classes]
        max_w = max(w for _, w in weights) or 1.0
        questions, skipped = [], 0
        for e, w in weights:
            if random.random() < (w / max_w):
                try:
                    questions.append(self.generate_question_for_target(e))
                except Exception:
                    skipped += 1
            else:
                skipped += 1
        return questions, skipped

def save_questions(questions: List[Dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    load_imports: bool,
    concept_source: str,
    concept_scope: str,
    concept_list: Optional[Set[str]] = None,
    skip_existing: bool = True,
    onto_paths: Optional[List[Path]] = None,
    suppress_io: bool = False,
) -> None:
    """Process a single OWL/RDF/Turtle ontology file and write questions JSON.

    - Derives a mirrored, Windows-safe output path under `output_root`.
    - If `concept_source` is "external", only includes classes whose IRI is present in `concept_list`.
    - If `concept_source` is "ontology", includes classes found within the ontology.
    """
    # Build mirrored relative path
    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name

    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)

    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    out_file = out_dir / f"class2def_{safe_stem}.json"

    if skip_existing and out_file.exists():
        logging.info(f"Skip existing: {out_file}")
        return

    loader = OntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    with silence_stdio(suppress_io):
        onto = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    with silence_stdio(suppress_io):
        loader.preload_entities()

    # Determine candidate classes by source
    if concept_source == "external":
        selected: List[ThingClass] = []
        iri_to_entity = {str(c.iri): c for c in onto.classes()}
        if not concept_list:
            logging.warning("External concept source selected but no concept list provided; nothing to do.")
            selected = []
        else:
            for iri in concept_list:
                ent = iri_to_entity.get(iri)
                if ent is not None and ent != owl.Thing and get_definition(ent) != "No definition provided.":
                    selected.append(ent)
                else:
                    logging.debug(f"External concept not found or without definition: {iri}")
    else:  # ontology
        selected = loader.get_all_classes_with_definition()

    # Filter by concept scope (all, native, imported)
    if concept_scope != "all":
        imported_ontos = set(getattr(loader.onto, "imported_ontologies", []) or [])
        def is_native(cls: ThingClass) -> bool:
            cls_onto = getattr(getattr(cls, "namespace", None), "ontology", None)
            return cls_onto is loader.onto
        def is_imported(cls: ThingClass) -> bool:
            cls_onto = getattr(getattr(cls, "namespace", None), "ontology", None)
            return (cls_onto is not None) and (cls_onto is not loader.onto)
        if concept_scope == "native":
            selected = [c for c in selected if is_native(c)]
        elif concept_scope == "imported":
            selected = [c for c in selected if is_imported(c)]

    gen = QuestionGenerator(selected)
    with silence_stdio(suppress_io):
        q, sk = gen.generate_all_questions()
    logging.info(f"Generated {len(q)} questions (skipped {sk}) from {file_path.name}.")
    if q:
        save_questions(q, out_file)

def parse_concept_file(path: Optional[Path]) -> Optional[Set[str]]:
    if not path:
        return None
    if not path.exists():
        logging.error(f"Concept file does not exist: {path}")
        return None
    concepts: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if not v or v.startswith("#"):
                continue
            concepts.add(v)
    return concepts or None


def main():
    parser = argparse.ArgumentParser(description="Generate class-to-definition MCQs from ontologies.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ontology file or directory to search for ontologies (.owl/.rdf/.rdfs/.ttl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output root directory where Windows-safe mirrored folders and JSON will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling.",
    )
    parser.add_argument(
        "--no-imports",
        action="store_true",
        help="Do not load imports (local-only). Speeds up and avoids network.",
    )
    parser.add_argument(
        "--onto-path",
        action="append",
        default=None,
        help="Additional local directories to resolve owl:imports (can be passed multiple times).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip files with existing output JSON.",
    )
    parser.add_argument(
        "--concept-source",
        type=str,
        choices=["ontology", "external"],
        default="ontology",
        help="Select concept source: all eligible classes in ontology, or an external IRI list.",
    )
    parser.add_argument(
        "--concept-scope",
        type=str,
        choices=["all", "native", "imported"],
        default="all",
        help="Filter concepts by origin: all classes, only classes defined in the main ontology (native), or only classes coming from imported ontologies (imported).",
    )
    parser.add_argument(
        "--concept-file",
        type=str,
        default=None,
        help="Path to a text file (one IRI per line) used when concept-source=external.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="info",
        help="Logging level: debug, info, warning, error.",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress warnings from Owlready2 and Python warnings to keep output clean.",
    )

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("process.log", "w", "utf-8"),
        ],
    )

    # Optionally suppress warnings from libraries and Python warnings module
    if args.no_warnings:
        try:
            set_log_level(0)  # Silence Owlready2's own WARNING/INFO prints
        except Exception:
            pass
        warnings.filterwarnings("ignore")
        for name in ("owlready2", "rdflib"):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass

    random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    exts = (".owl", ".rdf", ".rdfs", ".ttl")
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

    logging.info(f"Found {len(files)} ontology files.")

    concept_list = None
    if args.concept_source == "external":
        concept_list = parse_concept_file(Path(args.concept_file) if args.concept_file else None)
        if not concept_list:
            logging.warning("No concepts loaded from external file; run will produce no questions unless some are matched.")

    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                load_imports=not args.no_imports,
                concept_source=args.concept_source,
                concept_scope=args.concept_scope,
                concept_list=concept_list,
                skip_existing=not args.no_skip_existing,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                suppress_io=args.no_warnings,
            )
        except Exception as e:
            logging.error(f"Failed processing {fp}: {e}")

if __name__ == "__main__":
    main()
