import json
import os
import random
import logging
import argparse
import warnings
import sys
import signal
from contextlib import contextmanager
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Set
import re

from rdflib import URIRef, Literal
from owlready2 import World, ThingClass, owl, onto_path, set_log_level


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
    get_label as shared_get_label,
    global_class_metrics,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    selection_weight,
    slugify_for_windows,
    siblings as class_siblings,
    suppress_library_noise,
)


# Caches to avoid repeated lookups
definition_cache: Dict[str, str] = {}
label_cache: Dict[str, str] = {}
depth_cache: Dict[ThingClass, Optional[int]] = {}


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
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    label = shared_get_label(entity)
    label_cache[key] = label
    return label


def is_low_quality_definition(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return (
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
        or len(normalized.split()) < 4
        or len(normalized) < 24
        or "index the concept of tmi" in normalized
        or normalized.endswith("the concept of tmi")
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").replace("-", " ")).strip().lower()


def normalize_option_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def text_tokens(text: str) -> Set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_text(text)) if len(token) >= 3}


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def is_generic_top_class(entity: ThingClass) -> bool:
    label = get_label(entity).strip()
    if not label:
        return True
    normalized = normalize_text(label).strip(".")
    if normalized in {"thing", "motif", "function", "person", "type", "class", "entity"}:
        return True
    alpha = re.sub(r"[^A-Za-z]", "", label)
    depth = compute_depth(entity)
    return bool(alpha) and len(alpha) >= 4 and alpha.isupper() and depth is not None and depth <= 2

def compute_depth(entity, memo=None):
    return class_depth(entity, memo=memo if memo is not None else depth_cache)


def depth_distance(entity, target_depth: Optional[int]) -> int:
    depth = compute_depth(entity)
    if depth is None or target_depth is None:
        return 9999
    return abs(depth - target_depth)

def get_siblings(entity):
    return class_siblings(entity)

def compute_global_metrics(classes):
    return global_class_metrics(classes)

def compute_selection_weight(entity, gm):
    return selection_weight(entity, gm)

class OntologyLoader(BaseOntologyLoader):
    def __init__(self, file_path: Path, load_imports: bool = True, onto_paths: Optional[List[Path]] = None):
        super().__init__(file_path, load_imports=load_imports, onto_paths=onto_paths)

    def annotation_ontology_iris(self) -> tuple[str, ...]:
        return (
            "http://purl.obolibrary.org/obo/iao.owl",
            "http://www.w3.org/2004/02/skos/core#",
        )

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

    def get_all_classes_with_definition(self, max_candidates: Optional[int] = None):
        if not self.onto:
            return []
        candidates = [cls for cls in self.onto.classes() if cls != owl.Thing]
        random.shuffle(candidates)
        selected = []
        for cls in candidates:
            definition = get_definition(cls)
            if (
                definition != "No definition provided."
                and not is_low_quality_definition(definition)
                and not is_generic_top_class(cls)
            ):
                selected.append(cls)
                if max_candidates and len(selected) >= max_candidates:
                    break
        return selected

class QuestionGenerator:
    def __init__(self, classes: Iterable[ThingClass], mask_concept: bool = True, max_questions: Optional[int] = None):
        self.classes = list(classes)
        self.mask_concept = mask_concept
        self.max_questions = max_questions

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
        target_depth = compute_depth(target)
        # Siblings (other subclasses of the same parent)
        for p in target.is_a:
            if isinstance(p, ThingClass) and p != owl.Thing:
                cand |= {
                    s for s in p.subclasses()
                    if s != target
                    and get_definition(s) != "No definition provided."
                    and not is_low_quality_definition(get_definition(s))
                    and not is_generic_top_class(s)
                    and depth_distance(s, target_depth) <= 1
                }
        # Children
        cand |= {
            c for c in target.subclasses()
            if c != target
            and get_definition(c) != "No definition provided."
            and not is_low_quality_definition(get_definition(c))
            and not is_generic_top_class(c)
            and depth_distance(c, target_depth) <= 1
        }
        # If fewer than 3 candidates, add nearby classes only; avoid global random fallback.
        if len(cand) < 3:
            others = list(self.classes)
            random.shuffle(others)
            for o in others:
                if o != target and o not in cand and depth_distance(o, target_depth) <= 1:
                    cand.add(o)
                if len(cand) >= 3:
                    break
        cand.discard(target)
        return list(cand)

    def generate_question_for_target(self, target: ThingClass) -> Dict:
        stats = class_stats(target)
        d = stats.depth
        s = stats.sibling_count
        sub = stats.subclass_count
        par = stats.parent_count

        # Correct option
        defs = get_definition(target)
        lbl  = get_label(target)
        if is_generic_top_class(target):
            raise ValueError("generic top class")
        defs = self._mask_definition_text(defs, target)
        if is_low_quality_definition(defs):
            raise ValueError("low quality masked definition")
        if lexical_overlap(lbl, defs) > 0.45:
            raise ValueError("definition leaks target label")
        used_definitions = {normalize_option_text(defs)}
        options = [{"label": lbl, "definition": defs, "is_correct": True}]
        # Distractors
        candidate_distractors = self.get_candidate_distractors(target)
        if len(candidate_distractors) < 3:
            raise ValueError("not enough quality distractors")
        random.shuffle(candidate_distractors)
        for dsc in candidate_distractors:
            masked = self._mask_definition_text(get_definition(dsc), dsc)
            normalized_masked = normalize_option_text(masked)
            if not normalized_masked or normalized_masked in used_definitions:
                continue
            if is_low_quality_definition(masked):
                continue
            if lexical_overlap(lbl, masked) > 0.45:
                continue
            options.append({
                "label": get_label(dsc),
                "definition": masked,
                "is_correct": False
            })
            used_definitions.add(normalized_masked)
            if len(options) >= 4:
                break
        if len(options) < 4:
            raise ValueError("not enough non-leaking distractors")
        if len({normalize_option_text(o["definition"]) for o in options}) != 4:
            raise ValueError("duplicate option definitions")
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
        random.shuffle(self.classes)
        questions, skipped = [], 0
        # Keep generation deterministic and high-recall for single-ontology runs
        # such as Propp. Previously weighted sampling could drop most classes.
        for e in self.classes:
            if self.max_questions and len(questions) >= self.max_questions:
                break
            try:
                questions.append(self.generate_question_for_target(e))
            except Exception:
                skipped += 1
        return questions, skipped

def save_questions(questions: List[Dict], save_path: Path) -> None:
    save_json(questions, save_path, description="questions")


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
    max_questions: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> None:
    """Process a single OWL/RDF/Turtle ontology file and write questions JSON.

    - Derives a mirrored, Windows-safe output path under `output_root`.
    - If `concept_source` is "external", only includes classes whose IRI is present in `concept_list`.
    - If `concept_source` is "ontology", includes classes found within the ontology.
    """
    # Build mirrored relative path
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    out_file = out_dir / f"class2def_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "class2def", safe_stem)

    if skip_existing and out_file.exists():
        logging.info(f"Skip existing: {out_file}")
        return
    if skip_existing and empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    loader = OntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    with silence_stdio(suppress_io):
        onto = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    if not max_candidates:
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
        selected = loader.get_all_classes_with_definition(max_candidates=max_candidates)

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

    gen = QuestionGenerator(selected, max_questions=max_questions)
    with silence_stdio(suppress_io):
        q, sk = gen.generate_all_questions()
    logging.info(f"Generated {len(q)} questions (skipped {sk}) from {file_path.name}.")
    if q:
        save_questions(q, out_file)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_class_definition_questions",
            extra={"candidates": len(selected), "skipped": sk},
        )
    depth_cache.clear()
    definition_cache.clear()
    label_cache.clear()

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
        "--max-questions",
        type=int,
        default=0,
        help="Max questions per ontology (0 means all).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Max eligible class candidates to scan per ontology before generating questions (0 means all).",
    )
    parser.add_argument(
        "--file-timeout-seconds",
        type=int,
        default=0,
        help="Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=0,
        help="Skip ontology files larger than this many MB (0 means no size filter).",
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
    configure_logging(args.log, "process.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    exts = (".owl", ".rdf", ".rdfs", ".ttl")
    files, input_root = discover_ontology_files(input_path, exts)

    logging.info(f"Found {len(files)} ontology files.")

    concept_list = None
    if args.concept_source == "external":
        concept_list = parse_concept_file(Path(args.concept_file) if args.concept_file else None)
        if not concept_list:
            logging.warning("No concepts loaded from external file; run will produce no questions unless some are matched.")

    for fp in files:
        try:
            if args.max_file_mb and fp.stat().st_size > args.max_file_mb * 1024 * 1024:
                logging.warning("Skip large ontology file over %.1f MB: %s", args.max_file_mb, fp)
                continue
            with file_timeout(args.file_timeout_seconds):
                process_owl_file(
                    file_path=fp,
                    input_root=input_root,
                    output_root=output_root,
                    load_imports=not args.no_imports,
                    concept_source=args.concept_source,
                    concept_scope=args.concept_scope,
                    concept_list=concept_list,
                    skip_existing=not args.no_skip_existing,
                    onto_paths=resolve_onto_paths(args.onto_path),
                    suppress_io=args.no_warnings,
                    max_questions=args.max_questions or None,
                    max_candidates=args.max_candidates or None,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"Failed processing {fp}: {e}")

if __name__ == "__main__":
    main()
