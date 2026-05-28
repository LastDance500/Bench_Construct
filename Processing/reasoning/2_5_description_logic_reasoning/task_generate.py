#!/usr/bin/env python3
"""
Satisfiability puzzle generator (manual pattern detection with optional reasoner).
"""
import os
import re
import json
import random
import logging
import types
import multiprocessing
import queue
import argparse
import warnings
import sys
import signal
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import List, Optional
from owlready2 import World, Thing, ObjectProperty, AllDisjoint, sync_reasoner, Not, Nothing, ThingClass, ObjectPropertyClass, onto_path, set_log_level


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    configure_logging,
    configure_world_paths,
    discover_ontology_files,
    empty_marker_path,
    file_timeout,
    FileProcessingTimeout,
    load_ontology,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)

# Defaults
EXTENSIONS = (".owl", ".rdf", ".ttl", ".rdfs")
NUM_PUZZLES_PER_ONTOLOGY = 100
MIN_COMPLEXITY = 3
MAX_CLASSES = 10000
REASONER_TIMEOUT = 600
SKIP_FILES = {"Thesaurus.owl"}
LARGE_ONTOLOGY_THRESHOLD = 5000
SAMPLE_SIZE_LARGE = 2000
MODULE_MAX_CLASSES = 80
MODULE_MAX_PROPERTIES = 35
MODULE_SEED_CLASSES = 5

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)
os.environ.setdefault("JAVA_MEMORY", "2g")
_ACTIVE_REASONER_PROCESSES: set[multiprocessing.Process] = set()
_SIGNAL_HANDLERS_INSTALLED = False

# Multiprocessing start method
try:
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass


def safe_load_ontology(world, iri):
    """Safely load ontology (local-only by default)."""
    try:
        configure_world_paths(world, None)
        return load_ontology(world, Path(iri.replace("file://", "")), load_imports=False)
    except Exception as e:
        logger.error(f"Failed to load {iri}: {e}")
        return None


def _kill_process_group(pid: Optional[int], sig: int) -> None:
    """Best-effort kill for the reasoner child and any Java process it spawned."""
    if os.name != "posix" or not pid:
        return
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        logger.warning("No permission to signal reasoner process group %s: %s", pid, exc)
    except Exception as exc:
        logger.debug("Could not signal reasoner process group %s: %s", pid, exc)


def _terminate_reasoner_process(process: multiprocessing.Process, grace_seconds: int = 15) -> None:
    """Terminate both the Python worker and the Java reasoner subprocess.

    Owlready2 starts HermiT as a Java subprocess. Killing only the Python
    multiprocessing child can leave Java running, so the worker is placed in
    its own process group and the whole group is signalled here.
    """
    pid = getattr(process, "pid", None)
    _kill_process_group(pid, signal.SIGTERM)
    if process.is_alive():
        process.terminate()
    process.join(grace_seconds)
    if process.is_alive():
        logger.warning("Reasoner worker still alive after SIGTERM; sending SIGKILL")
        _kill_process_group(pid, signal.SIGKILL)
        process.kill()
        process.join(3)


def _cleanup_active_reasoners() -> None:
    for process in list(_ACTIVE_REASONER_PROCESSES):
        try:
            _terminate_reasoner_process(process, grace_seconds=5)
        except Exception as exc:
            logger.debug("Failed cleaning active reasoner process: %s", exc)
        finally:
            _ACTIVE_REASONER_PROCESSES.discard(process)


def _install_shutdown_handlers() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED or os.name != "posix":
        return

    def _handle_shutdown(signum, _frame):
        logger.warning("Received signal %s; cleaning active reasoner processes", signum)
        _cleanup_active_reasoners()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    _SIGNAL_HANDLERS_INSTALLED = True


@contextmanager
def run_reasoner_with_timeout(world, timeout=REASONER_TIMEOUT):
    """Run reasoner with timeout in a separate process."""
    def target(out_q):
        if os.name == "posix":
            try:
                os.setsid()
            except Exception:
                pass
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull), world:
                    sync_reasoner(world)
            unsat_names = []
            try:
                unsat_names.extend(
                    getattr(cls, "name", "")
                    for cls in world.inconsistent_classes()
                    if getattr(cls, "name", "").startswith("Tmp")
                )
            except Exception:
                pass
            for ontology in list(getattr(world, "ontologies", {}).values()):
                try:
                    for cls in ontology.classes():
                        if not getattr(cls, "name", "").startswith("Tmp"):
                            continue
                        if cls == Nothing or cls.equivalent_to == [Nothing] or Nothing in cls.ancestors():
                            unsat_names.append(cls.name)
                except Exception:
                    continue
            out_q.put({"status": "done", "unsat_names": sorted(set(unsat_names))})
        except Exception as e:
            out_q.put(e)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(q,))
    p.daemon = False
    try:
        p.start()
        _ACTIVE_REASONER_PROCESSES.add(p)
        yield p, q
        p.join(timeout)
        if p.is_alive():
            logger.info("Reasoner timed out; terminating reasoner process group")
            raise TimeoutError("Reasoning timed out")
    finally:
        _terminate_reasoner_process(p)
        _ACTIVE_REASONER_PROCESSES.discard(p)
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
        try:
            p.close()
        except Exception:
            pass


def dl_str(expr_str: str) -> str:
    """Convert Owlready2 expression string into DL-like notation."""
    s = expr_str
    s = re.sub(r'([A-Za-z0-9_])\-([A-Za-z0-9_])', r'\1_\2', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.some\(\s*([^\)]+?)\s*\)', r'∃\1.(\2)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.only\(\s*([^\)]+?)\s*\)', r'∀\1.(\2)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.min\(\s*([0-9]+)\s*,\s*([^\)]+?)\s*\)', r'≥\2 \1.(\3)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.max\(\s*([0-9]+)\s*,\s*([^\)]+?)\s*\)', r'≤\2 \1.(\3)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.exactly\(\s*([0-9]+)\s*,\s*([^\)]+?)\s*\)', r'=\2 \1.(\3)', s)
    s = re.sub(r'Not\(\s*([^\)]+?)\s*\)', r'¬(\1)', s)
    s = s.replace(' & ', ' ⊓ ').replace(' | ', ' ⊔ ')
    s = re.sub(r'\b(?:[A-Za-z0-9_]+\.)+([A-Z][A-Za-z0-9_]+)\b', r'\1', s)
    return s


def apply_labels_to_dl(dl: str, class_map: dict, prop_map: dict) -> str:
    """Replace identifiers by labels for readability."""
    for ident, label in sorted(prop_map.items(), key=lambda x: -len(x[0])):
        dl = re.sub(r'\b' + re.escape(ident) + r'\b', label, dl)
    for ident, label in sorted(class_map.items(), key=lambda x: -len(x[0])):
        dl = re.sub(r'\b' + re.escape(ident) + r'\b', label, dl)
    return dl


def complexity_score(dl: str) -> int:
    """Compute a balanced complexity score."""
    score = 0
    score += len(re.findall(r'[∃∀]', dl)) * 2
    score += len(re.findall(r'[≥≤=]', dl)) * 2
    score += len(re.findall(r'[⊓⊔]', dl)) * 1
    score += len(re.findall(r'[¬]', dl)) * 1
    score += len(re.findall(r'\([^)]*\([^)]*\)[^)]*\)', dl)) * 3
    score += len(re.findall(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*', dl)) * 2
    concepts = set(re.findall(r'[A-Z][A-Za-z0-9_]*', dl))
    score += min(len(concepts) * 0.5, 4)
    score += len(re.findall(r'[∃∀][^)]*[∃∀]', dl)) * 3
    score += len(re.findall(r'¬\([^)]*[⊓⊔][^)]*\)', dl)) * 2
    return max(int(score), 1)


def analyze_expression_features(dl_expr: str) -> dict:
    """Analyze DL expression features."""
    return {
        "has_quantifiers": bool(re.search(r'[∃∀]', dl_expr)),
        "has_number_restrictions": bool(re.search(r'[≥≤=]', dl_expr)),
        "has_negation": bool(re.search(r'[¬]', dl_expr)),
        "has_conjunction": bool(re.search(r'[⊓]', dl_expr)),
        "has_disjunction": bool(re.search(r'[⊔]', dl_expr)),
        "has_nested_quantifiers": bool(research := re.search(r'[∃∀][^)]*[∃∀]', dl_expr)),
        "has_role_chains": bool(re.search(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*', dl_expr)),
        "is_basic_contradiction": bool(re.search(r'(\w+)\s*⊓\s*¬\(\1\)', dl_expr)),
        "is_number_contradiction": bool(re.search(r'≥\d+.*≤\d+|≤\d+.*≥\d+|=0.*∃|∃.*=0', dl_expr)),
        "quantifier_count": len(re.findall(r'[∃∀]', dl_expr)),
        "concept_count": len(set(re.findall(r'[A-Z][A-Za-z0-9_]*', dl_expr))),
        "role_count": len(set(re.findall(r'[a-z][A-Za-z0-9_]*', dl_expr))),
        "nesting_depth": count_nesting_depth(dl_expr),
    }


def count_nesting_depth(expr: str) -> int:
    """Count parenthesis nesting depth."""
    max_depth = 0
    current_depth = 0
    for char in expr:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    return max_depth


def check_satisfiability_manual_from_string(expr_str):
    """Pattern-based satisfiability detection from expression string."""
    try:
        if re.search(r'(\w+)\s*&\s*Not\(\s*\1\s*\)', expr_str):
            return False
        dl_notation = dl_str(expr_str)
        if re.search(r'([^\s⊓]+)\s*⊓\s*¬\(\1\)', dl_notation):
            return False
        if 'min(' in expr_str and 'max(' in expr_str:
            min_match = re.search(r'\.min\(\s*(\d+)\s*', expr_str)
            max_match = re.search(r'\.max\(\s*(\d+)\s*', expr_str)
            if min_match and max_match and int(min_match.group(1)) > int(max_match.group(1)):
                return False
        if 'exactly(0' in expr_str and 'some(' in expr_str:
            return False
        nested_match = re.search(r'(\w+)\s*&\s*\([^)]*&\s*Not\(\s*(\w+)\s*\)', expr_str)
        if nested_match and (nested_match.group(1) == nested_match.group(2)):
            return False
        if re.search(r'\([^)]*\|\s*Not\([^)]*\)\)\s*&\s*(\w+)', expr_str):
            return False
        if re.search(r'(\w+)\.some\([^)]+\)\s*&\s*\1\.only\(\s*Not\([^)]+\)\s*\)', expr_str):
            return False
        m = re.search(r'=\s*0\s+([^.]+)\.([^⊓\s]+)', dl_notation)
        if m:
            role, concept = m.group(1), m.group(2)
            if re.search(rf'∃{re.escape(role)}\.{re.escape(concept)}', dl_notation) or re.search(rf'≥\s*1\s+{re.escape(role)}\.{re.escape(concept)}', dl_notation):
                return False
        m2 = re.search(r'∃\s+([^.]+)\.([^⊓\s]+)', dl_notation)
        if m2:
            role, concept = m2.group(1), m2.group(2)
            if re.search(rf'=\s*0\s+{re.escape(role)}\.{re.escape(concept)}', dl_notation):
                return False
        mins = re.findall(r'≥\s*(\d+)\s+([^.]+)\.([^⊓\s]+)', dl_notation)
        maxs = re.findall(r'≤\s*(\d+)\s+([^.]+)\.([^⊓\s]+)', dl_notation)
        for min_val, min_role, min_concept in mins:
            for max_val, max_role, max_concept in maxs:
                if min_role == max_role and min_concept == max_concept and int(min_val) > int(max_val):
                    return False
        return True
    except Exception as e:
        logger.warning(f"Error checking satisfiability from string: {e}")
        return True


def check_satisfiability_manual(cls):
    """Pattern-based satisfiability detection from a temporary class."""
    try:
        expr_str = str(cls.equivalent_to[0]) if cls.equivalent_to else str(cls)
        if re.search(r'(\w+)\s*&\s*Not\(\s*\1\s*\)', expr_str):
            return False
        dl_notation = dl_str(expr_str)
        if re.search(r'([^\s⊓]+)\s*⊓\s*¬\(\1\)', dl_notation):
            return False
        if 'min(' in expr_str and 'max(' in expr_str:
            min_match = re.search(r'\.min\(\s*(\d+)\s*', expr_str)
            max_match = re.search(r'\.max\(\s*(\d+)\s*', expr_str)
            if min_match and max_match and int(min_match.group(1)) > int(max_match.group(1)):
                return False
        if 'exactly(0' in expr_str and 'some(' in expr_str:
            return False
        if cls.equivalent_to == [Nothing]:
            return False
        if Nothing in cls.ancestors():
            return False
        if cls == Nothing:
            return False
        if re.search(r'(\w+)\s*&\s*\([^)]*&\s*Not\(\s*\1\s*\)', expr_str):
            return False
        if re.search(r'\([^)]*\|\s*Not\([^)]*\)\)\s*&\s*(\w+)', expr_str):
            return False
        if re.search(r'(\w+)\.some\([^)]+\)\s*&\s*\1\.only\(\s*Not\([^)]+\)\s*\)', expr_str):
            return False
        return True
    except Exception as e:
        logger.warning(f"Error checking satisfiability: {e}")
        return True


def smart_sample_classes(classes, target_size):
    """Sample meaningful classes from a large pool."""
    if len(classes) <= target_size:
        return classes
    meaningful_classes = []
    for cls in classes:
        try:
            if cls.name in ['Thing', 'Nothing', 'topObjectProperty', 'bottomObjectProperty']:
                continue
            if re.match(r'^[A-Z]+_\d+$', cls.name):
                continue
            if re.match(r'^[A-Z]{2,}_\d+', cls.name):
                continue
            if re.match(r'^[A-Z]{3,}_\d+', cls.name):
                continue
            if re.match(r'^[A-Z]+:\d+', cls.name):
                continue
            if re.match(r'^[A-Z]+\.\d+', cls.name):
                continue
            if re.match(r'^[A-Z]{2,}\d+$', cls.name):
                continue
            if not hasattr(cls, 'label') or not cls.label:
                continue
            meaningful_words = ['person','organization','event','activity','object','document','type','class','category','kind','property','attribute','work','resource','area','format','version','subdivision']
            if any(word in cls.name.lower() for word in meaningful_words):
                meaningful_classes.append(cls)
            elif len(cls.name) > 5 and not re.search(r'[_\d]', cls.name):
                meaningful_classes.append(cls)
        except Exception:
            continue
    if len(meaningful_classes) <= target_size:
        return meaningful_classes
    def class_priority(cls):
        name_len = len(cls.name) if hasattr(cls, 'name') else 0
        return abs(name_len - 10)
    meaningful_classes.sort(key=class_priority)
    return meaningful_classes[:target_size]


def smart_sample_properties(props, target_size):
    """Sample meaningful properties from a large pool."""
    if len(props) <= target_size:
        return props
    meaningful_props = []
    for prop in props:
        try:
            if prop.name in ['topObjectProperty', 'bottomObjectProperty']:
                continue
            if re.match(r'^[A-Z]+_\d+$', prop.name):
                continue
            if re.match(r'^[A-Z]{2,}_\d+', prop.name):
                continue
            if re.match(r'^[A-Z]{3,}_\d+', prop.name):
                continue
            if re.match(r'^[A-Z]+:\d+', prop.name):
                continue
            if re.match(r'^[A-Z]+\.\d+', prop.name):
                continue
            if re.match(r'^[A-Z]{2,}\d+$', prop.name):
                continue
            meaningful_words = ['has','is','of','for','with','by','from','to','in','on','contains','includes','relates','connects','describes','represents','part','member','type','kind','category','class','property']
            if any(word in prop.name.lower() for word in meaningful_words):
                meaningful_props.append(prop)
            elif len(prop.name) > 5 and not re.search(r'[_\d]', prop.name):
                meaningful_props.append(prop)
        except Exception:
            continue
    if len(meaningful_props) <= target_size:
        return meaningful_props
    def prop_priority(prop):
        name_len = len(prop.name) if hasattr(prop, 'name') else 0
        return abs(name_len - 10)
    meaningful_props.sort(key=prop_priority)
    return meaningful_props[:target_size]


def safe_entity_name(name: str, prefix: str, used: set[str]) -> str:
    base = re.sub(r'[^0-9A-Za-z_]+', '_', str(name or '')).strip('_')
    if not base:
        base = prefix
    if base[0].isdigit():
        base = f"{prefix}_{base}"
    candidate = base
    counter = 1
    while candidate in used:
        counter += 1
        candidate = f"{base}_{counter}"
    used.add(candidate)
    return candidate


def is_usable_class(cls) -> bool:
    try:
        return isinstance(cls, ThingClass) and cls not in (Thing, Nothing) and bool(getattr(cls, "name", None))
    except Exception:
        return False


def is_usable_object_property(prop) -> bool:
    try:
        return isinstance(prop, ObjectPropertyClass) and bool(getattr(prop, "name", None))
    except Exception:
        return False


def class_local_score(cls, props) -> int:
    score = 0
    try:
        score += len([p for p in getattr(cls, "is_a", []) if isinstance(p, ThingClass)])
        score += min(len(list(cls.subclasses())), 8)
    except Exception:
        pass
    for prop in props:
        try:
            if cls in (getattr(prop, "domain", []) or []):
                score += 2
            if cls in (getattr(prop, "range", []) or []):
                score += 2
        except Exception:
            continue
    try:
        score += len(getattr(cls, "equivalent_to", []) or [])
    except Exception:
        pass
    return score


def add_bounded(target: set, values, limit: int) -> None:
    for value in values:
        if len(target) >= limit:
            return
        if value is not None:
            target.add(value)


def extract_local_module_entities(onto, candidate_classes, candidate_props):
    """Sample a bounded local module around high-signal seed classes."""
    classes = [cls for cls in candidate_classes if is_usable_class(cls)]
    props = [prop for prop in candidate_props if is_usable_object_property(prop)]
    if not classes or not props:
        return set(), set()

    scored = sorted(classes, key=lambda cls: class_local_score(cls, props), reverse=True)
    seed_pool = scored[: min(max(MODULE_SEED_CLASSES * 8, 20), len(scored))]
    seeds = random.sample(seed_pool, min(MODULE_SEED_CLASSES, len(seed_pool)))

    module_classes = set()
    for seed in seeds:
        if len(module_classes) >= MODULE_MAX_CLASSES:
            break
        module_classes.add(seed)
        parents = [p for p in getattr(seed, "is_a", []) if is_usable_class(p)]
        add_bounded(module_classes, parents[:3], MODULE_MAX_CLASSES)
        for parent in parents[:2]:
            add_bounded(module_classes, [gp for gp in getattr(parent, "is_a", []) if is_usable_class(gp)][:2], MODULE_MAX_CLASSES)
            siblings = [sib for sib in parent.subclasses() if is_usable_class(sib) and sib is not seed]
            random.shuffle(siblings)
            add_bounded(module_classes, siblings[:3], MODULE_MAX_CLASSES)
        children = [child for child in seed.subclasses() if is_usable_class(child)]
        random.shuffle(children)
        add_bounded(module_classes, children[:4], MODULE_MAX_CLASSES)
        for expr in (getattr(seed, "equivalent_to", []) or []) + (getattr(seed, "is_a", []) or []):
            filler = getattr(expr, "value", None) or getattr(expr, "some_values_from", None)
            if is_usable_class(filler):
                add_bounded(module_classes, [filler], MODULE_MAX_CLASSES)

    module_props = set()
    for prop in props:
        if len(module_props) >= MODULE_MAX_PROPERTIES:
            break
        domains = [d for d in (getattr(prop, "domain", []) or []) if is_usable_class(d)]
        ranges = [r for r in (getattr(prop, "range", []) or []) if is_usable_class(r)]
        if set(domains) & module_classes or set(ranges) & module_classes:
            module_props.add(prop)
            add_bounded(module_classes, domains[:2] + ranges[:2], MODULE_MAX_CLASSES)

    if not module_props:
        for prop in props[:MODULE_MAX_PROPERTIES]:
            domains = [d for d in (getattr(prop, "domain", []) or []) if is_usable_class(d)]
            ranges = [r for r in (getattr(prop, "range", []) or []) if is_usable_class(r)]
            if domains and ranges:
                module_props.add(prop)
                add_bounded(module_classes, domains[:1] + ranges[:1], MODULE_MAX_CLASSES)
            if len(module_props) >= MODULE_MAX_PROPERTIES:
                break

    return module_classes, module_props


def clone_local_module(onto, source_classes, source_props, path: str):
    """Clone a bounded source fragment into a fresh ontology for reasoning."""
    module_world = World()
    safe_stem = safe_entity_name(Path(path).stem, "ontology", set())
    module_onto = module_world.get_ontology(f"http://ontobench.org/r5_module/{safe_stem}.owl")
    used_names = set()
    class_clone = {}
    prop_clone = {}
    class_label_map = {}
    prop_label_map = {}

    with module_onto:
        for source in sorted(source_classes, key=lambda cls: getattr(cls, "name", "")):
            name = safe_entity_name(getattr(source, "name", ""), "Class", used_names)
            clone = types.new_class(name, (Thing,), {})
            class_clone[source] = clone
            class_label_map[name] = getattr(source, "name", name)

        for source in sorted(source_props, key=lambda prop: getattr(prop, "name", "")):
            name = safe_entity_name(getattr(source, "name", ""), "property", used_names)
            clone = types.new_class(name, (ObjectProperty,), {})
            prop_clone[source] = clone
            prop_label_map[name] = getattr(source, "name", name)

    axiom_counts = {"subclass": 0, "domain": 0, "range": 0, "disjoint": 0}
    for source, clone in class_clone.items():
        for parent in getattr(source, "is_a", []) or []:
            if parent in class_clone and class_clone[parent] not in clone.is_a:
                clone.is_a.append(class_clone[parent])
                axiom_counts["subclass"] += 1

    for source, clone in prop_clone.items():
        for domain in getattr(source, "domain", []) or []:
            if domain in class_clone and class_clone[domain] not in clone.domain:
                clone.domain.append(class_clone[domain])
                axiom_counts["domain"] += 1
        for range_cls in getattr(source, "range", []) or []:
            if range_cls in class_clone and class_clone[range_cls] not in clone.range:
                clone.range.append(class_clone[range_cls])
                axiom_counts["range"] += 1

    try:
        with module_onto:
            for disjoint in onto.disjoint_classes():
                entities = [class_clone[entity] for entity in getattr(disjoint, "entities", []) if entity in class_clone]
                if len(entities) >= 2:
                    AllDisjoint(entities)
                    axiom_counts["disjoint"] += 1
    except Exception as exc:
        logger.debug("Failed copying disjoint axioms into local module: %s", exc)

    module_stats = {
        "gold_scope": "local_module",
        "reasoner_validated": True,
        "module_class_count": len(class_clone),
        "module_property_count": len(prop_clone),
        "module_axiom_counts": axiom_counts,
        "module_classes": list(class_label_map.values())[:MODULE_MAX_CLASSES],
        "module_properties": list(prop_label_map.values())[:MODULE_MAX_PROPERTIES],
        "reasoner_timeout": REASONER_TIMEOUT,
    }
    return module_world, module_onto, list(class_clone.values()), list(prop_clone.values()), class_label_map, prop_label_map, module_stats


def class_ancestors(cls):
    try:
        return {c for c in cls.ancestors() if isinstance(c, ThingClass)}
    except Exception:
        return {cls}


def is_compatible_filler(prop, cls) -> bool:
    """Conservatively require fillers to fit an explicit object-property range.

    OWL can infer additional range membership, but these generated puzzles are
    meant to be readable. Using an obviously wrong filler, e.g. a motif category
    as the winner of a character relation, produces misleading questions.
    """
    if not isinstance(prop, ObjectPropertyClass) or not isinstance(cls, ThingClass):
        return False
    ranges = [r for r in getattr(prop, "range", []) if isinstance(r, ThingClass)]
    if not ranges:
        return True
    ancestors = class_ancestors(cls)
    return any(r in ancestors or r == cls for r in ranges)


def choose_property_and_filler(props, classes):
    pairs = [
        (prop, cls)
        for prop in props
        for cls in classes
        if is_compatible_filler(prop, cls)
    ]
    if not pairs:
        return None, None
    return random.choice(pairs)


def choose_compatible_class(prop, classes):
    compatible = [cls for cls in classes if is_compatible_filler(prop, cls)]
    return random.choice(compatible) if compatible else None


def generate_cardinality_contradiction(classes, props):
    """Generate cardinality contradiction: =0 R.C ⊓ ∃R.C."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        return R.exactly(0, C) & R.some(C)
    except:
        return None


def generate_cardinality_range_contradiction(classes, props):
    """Generate cardinality range contradiction: ≥n R.C ⊓ ≤m R.C (n > m)."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        n = random.randint(2, 5)
        m = random.randint(1, n-1)
        return R.min(n, C) & R.max(m, C)
    except:
        return None


def generate_role_chain_contradiction(classes, props):
    """Generate role chain contradiction: R1.R2.C ⊓ ¬R1.R2.C."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        C = choose_compatible_class(R2, classes)
        if not C:
            return None
        return R1.some(R2.some(C)) & Not(R1.some(R2.some(C)))
    except:
        return None


def generate_nested_contradiction(classes, props):
    """Generate nested contradiction: A ⊓ (B ⊓ ¬A) and A = B."""
    if len(classes) < 1:
        return None
    try:
        A = random.choice(classes)
        return A & (A & Not(A))
    except:
        return None


def generate_multi_role_contradiction(classes, props):
    """Generate multi-role contradiction: R1.C ⊓ R2.C ⊓ ¬R1.C."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        compatible = [cls for cls in classes if is_compatible_filler(R1, cls) and is_compatible_filler(R2, cls)]
        if not compatible:
            return None
        C = random.choice(compatible)
        return R1.some(C) & R2.some(C) & Not(R1.some(C))
    except:
        return None


def generate_quantifier_contradiction(classes, props):
    """Generate quantifier restriction contradiction: R.some(C) ⊓ R.only(Not(C))."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        return R.some(C) & R.only(Not(C))
    except:
        return None


def generate_property_restriction_contradiction(classes, props):
    """Generate property restriction contradiction: R.some(C) ⊓ R.exactly(0, C)."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        return R.some(C) & R.exactly(0, C)
    except:
        return None


def generate_complex_cardinality_contradiction(classes, props):
    """Generate complex cardinality contradiction: R.exactly(n, C) ⊓ R.min(n+1, C)."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        n = random.randint(1, 3)
        return R.exactly(n, C) & R.min(n+1, C)
    except:
        return None


def generate_role_chain_complex_contradiction(classes, props):
    """Generate complex role chain contradiction: R1.some(R2.only(C)) ⊓ R1.only(R2.some(Not(C)))."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        C = choose_compatible_class(R2, classes)
        if not C:
            return None
        return R1.some(R2.only(C)) & R1.only(R2.some(Not(C)))
    except:
        return None


def generate_nested_quantifier_contradiction(classes, props):
    """Generate nested quantifier contradiction: R.some(C1 & R.only(Not(C1)))."""
    if not props or not classes:
        return None
    try:
        R, C1 = choose_property_and_filler(props, classes)
        if not R or not C1:
            return None
        return R.some(C1 & R.only(Not(C1)))
    except:
        return None


def generate_multi_property_contradiction(classes, props):
    """Generate multi-property contradiction: R1.some(C) ⊓ R2.only(Not(C)) ⊓ R1 ≡ R2."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        compatible = [cls for cls in classes if is_compatible_filler(R1, cls) and is_compatible_filler(R2, cls)]
        if not compatible:
            return None
        C = random.choice(compatible)
        # Simulate attribute equivalence
        return R1.some(C) & R2.only(Not(C)) & R1.some(Not(C))
    except:
        return None


def generate_cardinality_chain_contradiction(classes, props):
    """Generate cardinality chain contradiction: R1.exactly(1, C) ⊓ R2.exactly(1, C) ⊓ R1.some(R2.only(Not(C)))."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        compatible = [cls for cls in classes if is_compatible_filler(R1, cls) and is_compatible_filler(R2, cls)]
        if not compatible:
            return None
        C = random.choice(compatible)
        return R1.exactly(1, C) & R2.exactly(1, C) & R1.some(R2.only(Not(C)))
    except:
        return None


def generate_basic_quantifier_expression(classes, props):
    """Generate basic quantifier expression."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        if random.random() < 0.5:
            return R.some(C)
        else:
            return R.only(C)
    except:
        return None


def generate_complex_quantifier_expression(classes, props):
    """Generate complex quantifier combination expression."""
    if len(classes) < 2 or not props:
        return None
    try:
        R = random.choice(props)
        compatible = [cls for cls in classes if is_compatible_filler(R, cls)]
        if len(compatible) < 2:
            return None
        C1, C2 = random.sample(compatible, 2)
        if random.random() < 0.5:
            return R.some(C1 & C2)
        else:
            return R.only(C1 | C2)
    except:
        return None


def generate_cardinality_expression(classes, props):
    """Generate cardinality restriction expression."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        n = random.randint(1, 3)
        if random.random() < 0.5:
            return R.exactly(n, C)
        else:
            return R.min(n, C)
    except:
        return None


def generate_role_chain_expression(classes, props):
    """Generate role chain expression."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        C = choose_compatible_class(R2, classes)
        if not C:
            return None
        return R1.some(R2.some(C))
    except:
        return None


def generate_complex_nested_expression(classes, props):
    """Generate complex nested expression."""
    if len(classes) < 2 or not props:
        return None
    try:
        R = random.choice(props)
        compatible = [cls for cls in classes if is_compatible_filler(R, cls)]
        if len(compatible) < 2:
            return None
        C1, C2 = random.sample(compatible, 2)
        return R.some(C1 & R.only(C2))
    except:
        return None


def generate_multi_role_expression(classes, props):
    """Generate multi-role expression."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        compatible = [cls for cls in classes if is_compatible_filler(R1, cls) and is_compatible_filler(R2, cls)]
        if not compatible:
            return None
        C = random.choice(compatible)
        return R1.some(C) & R2.some(C)
    except:
        return None


def generate_property_restriction_expression(classes, props):
    """Generate property restriction expression."""
    if not props or not classes:
        return None
    try:
        R, C = choose_property_and_filler(props, classes)
        if not R or not C:
            return None
        n = random.randint(1, 2)
        return R.min(n, C) & R.max(n+2, C)
    except:
        return None


def generate_candidate_expressions(classes, props, max_puzzles=NUM_PUZZLES_PER_ONTOLOGY):
    candidates = []
    seen_expressions = set()
    
    def add_unique_candidate(expr):
        expr_str = str(expr)
        if expr_str not in seen_expressions:
            candidates.append(expr)
            seen_expressions.add(expr_str)
    
    # 生成可满足的表达式
    max_attempts = min(max_puzzles * 2, 200)  # 限制最大尝试次数
    attempts = 0
    target_sat = min(max_puzzles // 2, 50)  # 限制目标SAT数量

    for _ in range(target_sat):
        if attempts >= max_attempts or len(candidates) >= max_puzzles:
            logger.warning(f"Reached max attempts ({max_attempts}) or max candidates for SAT generation")
            break
            
        if props and classes:
            R, C = choose_property_and_filler(props, classes)
            if not R or not C:
                attempts += 1
                continue
            if random.random() < 0.3:
                # 基础量词
                add_unique_candidate(R.some(C))
            elif random.random() < 0.6:
                add_unique_candidate(R.only(C))
            elif random.random() < 0.8:
                # 复杂量词组合
                D = choose_compatible_class(R, classes)
                if C != D:
                    add_unique_candidate(R.some(C & D))
            else:
                # 嵌套量词
                R2 = random.choice(props)
                C2 = choose_compatible_class(R2, classes)
                if C2:
                    add_unique_candidate(R.some(R2.only(C2)))
        elif classes:
            C = random.choice(classes)
            add_unique_candidate(C)
        
        attempts += 1
    
    # 生成不可满足的表达式 - 使用新的生成函数
    unsat_generators = [
        generate_cardinality_contradiction,
        generate_cardinality_range_contradiction,
        generate_role_chain_contradiction,
        generate_quantifier_contradiction,
        generate_property_restriction_contradiction,
        generate_complex_cardinality_contradiction,
        generate_role_chain_complex_contradiction,
        generate_nested_quantifier_contradiction,
        generate_multi_property_contradiction,
        generate_cardinality_chain_contradiction
    ]
    
    # 生成UNSAT表达式
    for generator in unsat_generators:
        try:
            pattern = generator(classes, props)
            if pattern:
                add_unique_candidate(pattern)
        except Exception as e:
            logger.warning(f"Failed to generate UNSAT pattern: {e}")
    
    # 如果UNSAT模式不够，尝试生成更多复杂模式而不是基础矛盾
    max_unsat_attempts = 30  # 减少基础矛盾尝试次数
    unsat_attempts = 0
    target_unsat = min(max_puzzles // 2, 50)
    
    # 优先尝试生成更多复杂UNSAT模式
    complex_unsat_generators = [
        generate_cardinality_contradiction,
        generate_cardinality_range_contradiction,
        generate_role_chain_contradiction,
        generate_quantifier_contradiction,
        generate_property_restriction_contradiction,
        generate_complex_cardinality_contradiction,
        generate_role_chain_complex_contradiction,
        generate_nested_quantifier_contradiction,
        generate_multi_property_contradiction,
        generate_cardinality_chain_contradiction
    ]
    
    while (len([c for c in candidates if str(c).count('Not(') > 0]) < target_unsat and 
           unsat_attempts < max_unsat_attempts and
           len(candidates) < max_puzzles):
        if classes and props:
            # 优先使用复杂模式
            generator = random.choice(complex_unsat_generators)
            try:
                pattern = generator(classes, props)
                if pattern:
                    add_unique_candidate(pattern)
            except Exception as e:
                logger.warning(f"Failed to generate complex UNSAT pattern: {e}")
        elif classes:
            # 只有在没有属性时才使用基础矛盾
            try:
                C = random.choice(classes)
                add_unique_candidate(C & Not(C))
            except Exception as e:
                logger.warning(f"Failed to add basic contradiction: {e}")
        else:
            break
        unsat_attempts += 1
    
    if unsat_attempts >= max_unsat_attempts:
        logger.warning(f"Reached max UNSAT attempts ({max_unsat_attempts})")
    
    # 生成可满足的表达式 - 使用新的生成函数
    sat_generators = [
        generate_basic_quantifier_expression,
        generate_complex_quantifier_expression,
        generate_cardinality_expression,
        generate_role_chain_expression,
        generate_complex_nested_expression,
        generate_multi_role_expression,
        generate_property_restriction_expression
    ]
    
    # 生成SAT表达式
    for generator in sat_generators:
        try:
            pattern = generator(classes, props)
            if pattern:
                add_unique_candidate(pattern)
        except Exception as e:
            logger.warning(f"Failed to generate SAT pattern: {e}")
    
    # 如果候选表达式不够，生成更多基础表达式
    while len(candidates) < max_puzzles and attempts < max_attempts:
        if classes and len(classes) >= 2:
            C1, C2 = random.sample(classes, 2)
            if random.random() < 0.5:
                add_unique_candidate(C1 & C2)
            else:
                add_unique_candidate(C1 | C2)
        else:
            break
        attempts += 1
    
    logger.info(f"Generated {len(candidates)} total candidates")
    return candidates


def generate_puzzles_for_ontology(
    path: str,
    world: World,
    concept_scope: str = 'all',
    max_questions: int = NUM_PUZZLES_PER_ONTOLOGY,
) -> list:
    """Generate local-module, reasoner-validated DL puzzles for one ontology file."""
    logger.info(f"Generating puzzles for {path}")
    iri = f"file://{os.path.abspath(path)}"
    onto = safe_load_ontology(world, iri)
    if onto is None:
        return None

    all_classes = list(onto.classes())
    all_props = list(onto.object_properties())
    if concept_scope != 'all':
        def is_native(ent):
            return getattr(getattr(ent, 'namespace', None), 'ontology', None) is onto
        def is_imported(ent):
            o = getattr(getattr(ent, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            all_classes = [c for c in all_classes if is_native(c)]
            all_props = [p for p in all_props if is_native(p)]
        else:
            all_classes = [c for c in all_classes if is_imported(c)]
            all_props = [p for p in all_props if is_imported(p)]

    source_classes, source_props = extract_local_module_entities(onto, all_classes, all_props)
    if not source_classes or not source_props:
        logger.warning("Skipping %s: no usable local module (%d classes, %d props)", path, len(source_classes), len(source_props))
        return [], {
            "num_classes": len(all_classes),
            "num_properties": len(all_props),
            "r5_generation": {
                "source_class_count": len(source_classes),
                "source_property_count": len(source_props),
                "reason": "no_usable_local_module",
            },
        }

    module_world, tmp_onto, classes, props, class_map, prop_map, module_stats = clone_local_module(
        onto, source_classes, source_props, path
    )
    if not classes or not props:
        logger.warning("Skipping %s: local module clone is empty", path)
        module_world.close()
        return [], {
            "num_classes": len(all_classes),
            "num_properties": len(all_props),
            "r5_generation": module_stats | {"reason": "local_module_clone_empty"},
        }

    target_count = max(1, max_questions or NUM_PUZZLES_PER_ONTOLOGY)
    candidates = generate_candidate_expressions(classes, props, max_puzzles=target_count)
    if not candidates:
        logger.warning("Skipping %s: no DL expressions generated from local module", path)
        module_world.close()
        return [], {
            "num_classes": len(all_classes),
            "num_properties": len(all_props),
            "r5_generation": module_stats | {"reason": "no_dl_expressions_generated"},
        }

    logger.info("Creating %d temporary expression classes in local module", len(candidates))
    for idx, expr in enumerate(candidates):
        try:
            with tmp_onto:
                cls = types.new_class(f"Tmp{idx}", (Thing,), {})
                cls.equivalent_to = [expr]
        except Exception as e:
            logger.warning(f"Failed to create temporary class Tmp{idx}: {e}")
            continue

    try:
        logger.info("Running reasoner on local module for %s", path)
        with run_reasoner_with_timeout(module_world, timeout=REASONER_TIMEOUT) as (p, q):
            try:
                result = q.get(timeout=REASONER_TIMEOUT + 5)
            except queue.Empty:
                raise TimeoutError("Reasoning timed out")
        if isinstance(result, Exception):
            raise result
        reasoner_unsat_names = set(result.get("unsat_names", [])) if isinstance(result, dict) else set()
        logger.info("Reasoner completed for local module of %s", path)
    except Exception as e:
        logger.error("Reasoning failed for local module of %s: %s", path, e)
        try:
            module_world.close()
        except Exception:
            pass
        return None

    sats, unsats = [], []
    total_candidates = len(candidates)
    
    for idx, expr in enumerate(candidates):
        if idx % 10 == 0:  # 每10个显示一次进度
            logger.info(f"Processing candidate {idx+1}/{total_candidates} for {path}")
        is_sat = f"Tmp{idx}" not in reasoner_unsat_names
        
        raw = dl_str(str(expr))
        labeled = apply_labels_to_dl(raw, class_map, prop_map)
        comp = complexity_score(labeled)
        
        # 分析表达式特征
        expr_features = analyze_expression_features(labeled)
        
        entry = {
            "question": (
                "Given the following Description Logic expression, determine whether it is satisfiable "
                "with respect to the local ontology module. Answer only true or false.\n\n"
                f"Expression:\n{labeled}"
            ),
            "answer": "true" if is_sat else "false",
            "expression": labeled, 
            "satisfiable": is_sat, 
            "complexity": comp,
            "meta": {
                "raw_expression": str(expr),
                "dl_notation": raw,
                "features": expr_features,
                "generation_method": "manual_pattern_generation",
                "gold_method": "local_module_reasoner_classification",
                "gold_scope": "local_module",
                "reasoner_validated": True,
                "source_ontology": os.path.basename(path),
                "module": {
                    "class_count": module_stats["module_class_count"],
                    "property_count": module_stats["module_property_count"],
                    "axiom_counts": module_stats["module_axiom_counts"],
                },
            }
        }
        
        if is_sat and comp >= MIN_COMPLEXITY:
            sats.append(entry)
        elif not is_sat:
            unsats.append(entry)

    # 平衡选择
    random.shuffle(sats)
    random.shuffle(unsats)
    
    # 确保有足够的SAT和UNSAT问题
    sat_target = target_count // 2
    unsat_target = target_count - sat_target
    
    basic_unsats = [q for q in unsats if q.get("meta", {}).get("features", {}).get("is_basic_contradiction")]
    non_basic_unsats = [q for q in unsats if not q.get("meta", {}).get("features", {}).get("is_basic_contradiction")]
    selected_unsats = non_basic_unsats[:unsat_target]
    remaining_unsat = unsat_target - len(selected_unsats)
    if remaining_unsat > 0:
        selected_unsats += basic_unsats[:remaining_unsat]

    final = sats[:sat_target] + selected_unsats
    
    # 如果不够，用剩余的问题补充
    remaining = target_count - len(final)
    if remaining > 0:
        if len(sats) > sat_target:
            final += sats[sat_target:sat_target + remaining]
        elif len(non_basic_unsats) > len(selected_unsats):
            final += non_basic_unsats[len(selected_unsats):len(selected_unsats) + remaining]
        elif len(basic_unsats) > max(0, remaining_unsat):
            final += basic_unsats[max(0, remaining_unsat):max(0, remaining_unsat) + remaining]
    
    random.shuffle(final)
    
    logger.info(f"Generated {len(sats)} SAT, {len(unsats)} UNSAT problems")
    
    # 收集本体统计信息
    try:
        num_axioms = len(onto.axioms()) if hasattr(onto, 'axioms') and callable(getattr(onto, 'axioms')) else 0
    except:
        num_axioms = 0
    
    try:
        num_individuals = len(list(onto.individuals())) if hasattr(onto, 'individuals') and callable(getattr(onto, 'individuals')) else 0
    except:
        num_individuals = 0
    
    try:
        ontology_iri = str(onto.base_iri) if hasattr(onto, 'base_iri') and onto.base_iri else None
    except:
        ontology_iri = None
    
    try:
        imported_count = len(onto.imported_ontologies) if hasattr(onto, 'imported_ontologies') else 0
    except:
        imported_count = 0
    
    ontology_stats = {
        "num_classes": len(all_classes),
        "num_properties": len(all_props),
        "num_axioms": num_axioms,
        "num_individuals": num_individuals,
        "ontology_iri": ontology_iri,
        "imported_ontologies": imported_count,
        "r5_generation": module_stats,
    }
    
    try:
        tmp_onto.destroy()
    except Exception:
        pass
    try:
        module_world.close()
    except Exception:
        pass
    try:
        world.ontologies.clear()
    except Exception:
        pass
    return final, ontology_stats


def save_questions(questions: list, save_path: str, ontology_path: str, ontology_stats: dict = None) -> None:
    """Save questions to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 统计问题类型
    sat_count = sum(1 for q in questions if q.get("satisfiable", True))
    unsat_count = len(questions) - sat_count
    
    # 统计复杂度分布
    complexities = [q.get("complexity", 0) for q in questions]
    complexity_stats = {
        "min": min(complexities) if complexities else 0,
        "max": max(complexities) if complexities else 0,
        "avg": sum(complexities) / len(complexities) if complexities else 0
    }
    
    # 统计表达式类型
    expression_types = {
        "basic_contradiction": sum(1 for q in questions if "⊓ ¬(" in q.get("expression", "")),
        "quantifier_restriction": sum(1 for q in questions if any(sym in q.get("expression", "") for sym in ["∃", "∀"])),
        "number_restriction": sum(1 for q in questions if any(sym in q.get("expression", "") for sym in ["≥", "≤", "="])),
        "nested_quantifier": sum(1 for q in questions if q.get("expression", "").count("∃") + q.get("expression", "").count("∀") > 1)
    }
    
    data = {
        "metadata": {
            "ontology_file": os.path.basename(ontology_path),
            "ontology_path": ontology_path,
            "num_questions": len(questions),
            "satisfiable_count": sat_count,
            "unsatisfiable_count": unsat_count,
            "complexity_stats": complexity_stats,
            "expression_types": expression_types,
            "ontology_stats": ontology_stats or {},
            "generation_info": {
                "generator": "task_generate.py",
                "version": "2.0",
                "description": "Complex logic reasoning satisfiability puzzles"
            }
        },
        "questions": questions
    }
    save_json(data, Path(save_path), description="puzzle bundles")


def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    concept_scope: str,
    max_questions: int,
) -> None:
    """Process a single OWL file and write puzzles JSON."""
    logger.info(f"Processing {file_path}")
    if not file_path.exists() or not os.access(file_path, os.R_OK):
        logger.warning(f"Skipping {file_path}: file not accessible")
        return

    if os.path.basename(str(file_path)) in SKIP_FILES:
        logger.info(f"Skipping {file_path}: in skip list")
        return

    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f"satisfiability_puzzles_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "satisfiability_puzzles", safe_stem)

    if save_path.exists():
        logger.info(f"Skipping {file_path}: output exists")
        return
    if empty_path.exists():
        logger.info("Skip empty ontology marker: %s", empty_path)
        return

    world = World()
    try:
        result = generate_puzzles_for_ontology(
            str(file_path),
            world,
            concept_scope=concept_scope,
            max_questions=max_questions,
        )
        if result:
            puzzles, ontology_stats = result
            if puzzles:
                out_dir.mkdir(parents=True, exist_ok=True)
                save_questions(puzzles, str(save_path), str(file_path), ontology_stats)
            else:
                save_empty_marker(
                    empty_path,
                    source_file=file_path,
                    reason="no_valid_dl_satisfiability_puzzles",
                    extra={"ontology_stats": ontology_stats},
                )
    finally:
        world.close()


def main():
    global REASONER_TIMEOUT
    parser = argparse.ArgumentParser(description='Generate bounded description logic satisfiability puzzles from ontology modules.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=NUM_PUZZLES_PER_ONTOLOGY, help='Max questions per ontology.')
    parser.add_argument('--reasoner-timeout-seconds', type=int, default=REASONER_TIMEOUT, help='Timeout for each local-module reasoner run.')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if total R5 processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--no-imports', action='store_true', help='Accepted for CLI consistency; R5 is always loaded local-only.')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of classes/properties.')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()
    configure_logging(args.log)
    suppress_library_noise(args.no_warnings)
    REASONER_TIMEOUT = max(1, args.reasoner_timeout_seconds)
    _install_shutdown_handlers()

    if args.onto_path:
        for p in args.onto_path:
            try:
                pp = str(Path(p).resolve())
                if pp not in onto_path:
                    onto_path.append(pp)
            except Exception:
                pass

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    files, input_root = discover_ontology_files(input_path, EXTENSIONS)
    logger.info(f"Found {len(files)} files")
    try:
        for fp in files:
            try:
                with file_timeout(args.file_timeout_seconds):
                    process_owl_file(
                        fp,
                        input_root,
                        output_root,
                        concept_scope=args.concept_scope,
                        max_questions=args.max_questions,
                    )
            except FileProcessingTimeout as exc:
                logger.error("Timeout processing %s: %s", fp, exc)
                _cleanup_active_reasoners()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, cleaning up...")
        for world in list(World._instances):
            try:
                world.close()
            except Exception as e:
                logger.warning(f"Error closing World: {e}")
        raise SystemExit("Terminated by user")


if __name__ == "__main__":
    main()
