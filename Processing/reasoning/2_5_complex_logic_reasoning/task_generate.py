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
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional
from owlready2 import World, Thing, sync_reasoner, Not, Nothing, ThingClass, onto_path, set_log_level

# Defaults
EXTENSIONS = (".owl", ".rdf", ".ttl", ".rdfs")
NUM_PUZZLES_PER_ONTOLOGY = 100
MIN_COMPLEXITY = 3
MAX_CLASSES = 10000
REASONER_TIMEOUT = 600
SKIP_FILES = {"Thesaurus.owl"}
LARGE_ONTOLOGY_THRESHOLD = 5000
SAMPLE_SIZE_LARGE = 2000

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)
os.environ["JAVA_MEMORY"] = "8g"

# Multiprocessing start method
try:
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass


def slugify_for_windows(name: str) -> str:
    """Create a Windows-safe slug for folder/file names.

    Replace non-alphanumeric chars with underscore, collapse repeats,
    and trim leading/trailing underscores. Keep case for readability.
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

def safe_load_ontology(world, iri):
    """Safely load ontology (local-only by default)."""
    try:
        onto = world.get_ontology(iri)
        onto.load(only_local=True)
        return onto
    except Exception as e:
        logger.error(f"Failed to load {iri}: {e}")
        return None


@contextmanager
def run_reasoner_with_timeout(world, timeout=REASONER_TIMEOUT):
    """Run reasoner with timeout in a separate process."""
    def target(out_q):
        try:
            with world:
                sync_reasoner()
            out_q.put("done")
        except Exception as e:
            out_q.put(e)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(q,))
    p.daemon = False
    try:
        p.start()
        yield p, q
        p.join(timeout)
        if p.is_alive():
            logger.info("Sending SIGTERM to reasoner process")
            p.terminate()
            p.join(15)
            if p.is_alive():
                logger.warning("Process still alive, sending SIGKILL")
                p.kill()
                p.join(3)
            raise TimeoutError("Reasoning timed out")
    finally:
        if p.is_alive():
            p.terminate()
            p.join(15)
            if p.is_alive():
                p.kill()
                p.join(3)
        p.close()


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


def generate_cardinality_contradiction(classes, props):
    """Generate cardinality contradiction: =0 R.C ⊓ ∃R.C."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
        return R.exactly(0, C) & R.some(C)
    except:
        return None


def generate_cardinality_range_contradiction(classes, props):
    """Generate cardinality range contradiction: ≥n R.C ⊓ ≤m R.C (n > m)."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
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
        C = random.choice(classes)
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
        C = random.choice(classes)
        return R1.some(C) & R2.some(C) & Not(R1.some(C))
    except:
        return None


def generate_quantifier_contradiction(classes, props):
    """Generate quantifier restriction contradiction: R.some(C) ⊓ R.only(Not(C))."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
        return R.some(C) & R.only(Not(C))
    except:
        return None


def generate_property_restriction_contradiction(classes, props):
    """Generate property restriction contradiction: R.some(C) ⊓ R.exactly(0, C)."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
        return R.some(C) & R.exactly(0, C)
    except:
        return None


def generate_complex_cardinality_contradiction(classes, props):
    """Generate complex cardinality contradiction: R.exactly(n, C) ⊓ R.min(n+1, C)."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
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
        C = random.choice(classes)
        return R1.some(R2.only(C)) & R1.only(R2.some(Not(C)))
    except:
        return None


def generate_nested_quantifier_contradiction(classes, props):
    """Generate nested quantifier contradiction: R.some(C1 & R.only(Not(C1)))."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C1 = random.choice(classes)
        return R.some(C1 & R.only(Not(C1)))
    except:
        return None


def generate_multi_property_contradiction(classes, props):
    """Generate multi-property contradiction: R1.some(C) ⊓ R2.only(Not(C)) ⊓ R1 ≡ R2."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        C = random.choice(classes)
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
        C = random.choice(classes)
        return R1.exactly(1, C) & R2.exactly(1, C) & R1.some(R2.only(Not(C)))
    except:
        return None


def generate_basic_quantifier_expression(classes, props):
    """Generate basic quantifier expression."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
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
        C1, C2 = random.sample(classes, 2)
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
        R = random.choice(props)
        C = random.choice(classes)
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
        C = random.choice(classes)
        return R1.some(R2.some(C))
    except:
        return None


def generate_complex_nested_expression(classes, props):
    """Generate complex nested expression."""
    if len(classes) < 2 or not props:
        return None
    try:
        R = random.choice(props)
        C1, C2 = random.sample(classes, 2)
        return R.some(C1 & R.only(C2))
    except:
        return None


def generate_multi_role_expression(classes, props):
    """Generate multi-role expression."""
    if len(props) < 2 or not classes:
        return None
    try:
        R1, R2 = random.sample(props, 2)
        C = random.choice(classes)
        return R1.some(C) & R2.some(C)
    except:
        return None


def generate_property_restriction_expression(classes, props):
    """Generate property restriction expression."""
    if not props or not classes:
        return None
    try:
        R = random.choice(props)
        C = random.choice(classes)
        n = random.randint(1, 2)
        return R.min(n, C) & R.max(n+2, C)
    except:
        return None


def generate_puzzles_for_ontology(path: str, world: World, concept_scope: str = 'all') -> list:
    """Generate puzzles for one ontology file."""
    logger.info(f"Generating puzzles for {path}")
    iri = f"file://{os.path.abspath(path)}"
    onto = safe_load_ontology(world, iri)
    if onto is None:
        return []

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

    # smart sampling
    if len(all_classes) > LARGE_ONTOLOGY_THRESHOLD:
        logger.info(f"Large ontology detected ({len(all_classes)} classes), sampling...")
        classes = smart_sample_classes(all_classes, SAMPLE_SIZE_LARGE)
        props = smart_sample_properties(all_props, 500)
    else:
        classes = smart_sample_classes(all_classes, len(all_classes))
        props = smart_sample_properties(all_props, len(all_props))

    if len(classes) > MAX_CLASSES:
        logger.warning(f"Skipping {path}: too many classes after sampling ({len(classes)})")
        return []
    if not classes or not props:
        logger.warning(f"Skipping {path}: {len(classes)} classes, {len(props)} props")
        return []

    class_map = {cls.name: cls.name for cls in classes}
    prop_map = {prop.name: prop.name for prop in props}

    # 生成候选表达式
    candidates = []
    seen_expressions = set()  # 用于去重
    
    def add_unique_candidate(expr):
        """添加唯一的候选表达式"""
        expr_str = str(expr)
        if expr_str not in seen_expressions:
            candidates.append(expr)
            seen_expressions.add(expr_str)
    
    # 生成可满足的表达式
    max_attempts = min(NUM_PUZZLES_PER_ONTOLOGY * 2, 200)  # 限制最大尝试次数
    attempts = 0
    target_sat = min(NUM_PUZZLES_PER_ONTOLOGY // 2, 50)  # 限制目标SAT数量
    
    for _ in range(target_sat):
        if attempts >= max_attempts or len(candidates) >= NUM_PUZZLES_PER_ONTOLOGY:
            logger.warning(f"Reached max attempts ({max_attempts}) or max candidates for SAT generation")
            break
            
        if props and classes:
            R = random.choice(props)
            C = random.choice(classes)
            if random.random() < 0.3:
                # 基础量词
                add_unique_candidate(R.some(C))
            elif random.random() < 0.6:
                add_unique_candidate(R.only(C))
            elif random.random() < 0.8:
                # 复杂量词组合
                D = random.choice(classes)
                if C != D:
                    add_unique_candidate(R.some(C & D))
            else:
                # 嵌套量词
                R2 = random.choice(props)
                add_unique_candidate(R.some(R2.only(C)))
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
    target_unsat = min(NUM_PUZZLES_PER_ONTOLOGY // 2, 50)
    
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
           len(candidates) < NUM_PUZZLES_PER_ONTOLOGY):
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
    while len(candidates) < NUM_PUZZLES_PER_ONTOLOGY and attempts < max_attempts:
        if classes and len(classes) >= 2:
            C1, C2 = random.sample(classes, 2)
            if random.random() < 0.5:
                add_unique_candidate(C1 & C2)
            else:
                add_unique_candidate(C1 | C2)
        else:
            break
        attempts += 1
        logger.warning(f"Reached max UNSAT attempts ({max_unsat_attempts})")
    
    logger.info(f"Generated {len(candidates)} total candidates")

    # 对于大本体，跳过推理器，直接使用手动检测
    is_large_ontology = len(all_classes) > LARGE_ONTOLOGY_THRESHOLD
    
    if not is_large_ontology:
        # 创建临时本体并运行推理器
        try:
            tmp_iri = f"http://temp.org/tmp_{os.path.splitext(os.path.basename(path))[0]}.owl"
            tmp_onto = world.get_ontology(tmp_iri)
            tmp_onto.imported_ontologies.append(onto)
            
            logger.info(f"Creating {len(candidates)} temporary classes...")
            for idx, expr in enumerate(candidates):
                try:
                    with tmp_onto:
                        cls = types.new_class(f"Tmp{idx}", (Thing,), {})
                        cls.equivalent_to = [expr]
                except Exception as e:
                    logger.warning(f"Failed to create temporary class Tmp{idx}: {e}")
                    continue

            try:
                logger.info(f"Running reasoner for {path}")
                with run_reasoner_with_timeout(world, timeout=REASONER_TIMEOUT) as (p, q):
                    try:
                        result = q.get(timeout=REASONER_TIMEOUT + 5)
                    except queue.Empty:
                        raise TimeoutError("Reasoning timed out")
                if isinstance(result, Exception):
                    raise result
                logger.info(f"Reasoner completed for {path}")
            except Exception as e:
                logger.error(f"Reasoning failed for {path}: {e}")
                return []
        except Exception as e:
            logger.error(f"Failed to create temporary ontology: {e}")
            return []
    else:
        logger.info(f"Skipping reasoner for large ontology {path}, using manual detection only")
        tmp_onto = None

    # 使用手动检测方法
    sats, unsats = [], []
    total_candidates = len(candidates)
    
    for idx, expr in enumerate(candidates):
        if idx % 10 == 0:  # 每10个显示一次进度
            logger.info(f"Processing candidate {idx+1}/{total_candidates} for {path}")
        # 对于大本体，直接使用表达式字符串进行检测
        if is_large_ontology:
            expr_str = str(expr)
            is_sat = check_satisfiability_manual_from_string(expr_str)
        else:
            cls = tmp_onto[f"Tmp{idx}"]
            is_sat = check_satisfiability_manual(cls)
        
        raw = dl_str(str(expr))
        labeled = apply_labels_to_dl(raw, class_map, prop_map)
        comp = complexity_score(labeled)
        
        # 分析表达式特征
        expr_features = analyze_expression_features(labeled)
        
        entry = {
            "expression": labeled, 
            "satisfiable": is_sat, 
            "complexity": comp,
            "meta": {
                "raw_expression": str(expr),
                "dl_notation": raw,
                "features": expr_features,
                "generation_method": "manual_pattern_detection" + ("_no_reasoner" if is_large_ontology else "")
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
    sat_target = NUM_PUZZLES_PER_ONTOLOGY // 2
    unsat_target = NUM_PUZZLES_PER_ONTOLOGY - sat_target
    
    final = sats[:sat_target] + unsats[:unsat_target]
    
    # 如果不够，用剩余的问题补充
    remaining = NUM_PUZZLES_PER_ONTOLOGY - len(final)
    if remaining > 0:
        if len(sats) > sat_target:
            final += sats[sat_target:sat_target + remaining]
        elif len(unsats) > unsat_target:
            final += unsats[unsat_target:unsat_target + remaining]
    
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
        "num_classes": len(classes),
        "num_properties": len(props),
        "num_axioms": num_axioms,
        "num_individuals": num_individuals,
        "ontology_iri": ontology_iri,
        "imported_ontologies": imported_count
    }
    
    if tmp_onto:
        tmp_onto.destroy()
    world.ontologies.clear()
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
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(questions)} puzzles to {save_path}")


def process_owl_file(file_path: Path, input_root: Path, output_root: Path, concept_scope: str) -> None:
    """Process a single OWL file and write puzzles JSON."""
    logger.info(f"Processing {file_path}")
    if not file_path.exists() or not os.access(file_path, os.R_OK):
        logger.warning(f"Skipping {file_path}: file not accessible")
        return

    if os.path.basename(str(file_path)) in SKIP_FILES:
        logger.info(f"Skipping {file_path}: in skip list")
        return

    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    save_path = out_dir / f"satisfiability_puzzles_{safe_stem}.json"

    if save_path.exists():
        logger.info(f"Skipping {file_path}: output exists")
        return

    world = World()
    try:
        result = generate_puzzles_for_ontology(str(file_path), world, concept_scope=concept_scope)
        if result:
            puzzles, ontology_stats = result
            if puzzles:
                out_dir.mkdir(parents=True, exist_ok=True)
                save_questions(puzzles, str(save_path), str(file_path), ontology_stats)
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description='Generate complex logic satisfiability puzzles from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of classes/properties.')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()
    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')
    if args.no_warnings:
        try:
            set_log_level(0)
        except Exception:
            pass
        warnings.filterwarnings('ignore')

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
    files: List[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in EXTENSIONS:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, fnames in os.walk(str(input_path)):
            for fn in fnames:
                if fn.lower().endswith(EXTENSIONS):
                    files.append(Path(root)/fn)
    logger.info(f"Found {len(files)} files")
    try:
        for fp in files:
            process_owl_file(fp, input_root, output_root, concept_scope=args.concept_scope)
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