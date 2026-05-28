import json
import signal
import os
import random
import logging
import argparse
import warnings
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional
import owlready2
from owlready2 import (
    ThingClass, Restriction, And,
    SOME, ONLY, VALUE, MIN, MAX, EXACTLY,
    DataPropertyClass, ObjectPropertyClass, get_ontology, onto_path, set_log_level
)


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    configure_world_paths,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    get_label,
    limit_questions_by_subject,
    load_ontology,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)

class OntologyTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise OntologyTimeoutError()

signal.signal(signal.SIGALRM, _timeout_handler)


MAX_QUESTIONS = 500
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')
PROPP_EXCLUDED_LABELS = {
    'eTrap motif',
    'eTRAP motif',
    'eTRAP added motif',
    'Linking from AaTh-Numbers to ATU-Numbers',
    'Linking back to the ATU source',
}
TRIVIAL_SINGLE_HEADS = {
    'family_member',
    'child',
    'parent',
    'sibling',
    'biol_parent',
    'father',
    'mother',
    'stepparent',
    'step_parent',
    'dramatis_personae',
    'fictional_character',
    'proppian_function',
    'motif',
    'human',
    'person',
    'thing',
    'tale',
    'move',
    'publication',
    'object',
    'class',
    'type',
}


def normalize_atom(atom: str) -> tuple[str, ...]:
    parts = [part.strip() for part in atom.split('∧')]
    return tuple(sorted(parts))


def is_valid_label(text: str) -> bool:
    if not text or text == 'Unnamed' or text == '""':
        return False
    normalized = str(text).lower()
    return not any(term.lower() in normalized for term in PROPP_EXCLUDED_LABELS)


def is_valid_atom(atom: str) -> bool:
    pred = atom.split('(')[0].strip()
    if not is_valid_label(pred):
        return False
    if len(normalized_predicate(atom)) < 2:
        return False
    return bool(re.search(r"[A-Za-z]", pred))


def normalized_predicate(atom: str) -> str:
    pred = atom.split('(')[0].strip()
    pred = re.sub(r'([a-z])([A-Z])', r'\1_\2', pred)
    pred = re.sub(r'[^a-zA-Z0-9]+', '_', pred).strip('_').lower()
    return pred


def is_trivial_single_atom(atom: str) -> bool:
    return normalized_predicate(atom) in TRIVIAL_SINGLE_HEADS


def is_trivial_class_label(label: str) -> bool:
    return normalized_predicate(f"{label}(?x)") in TRIVIAL_SINGLE_HEADS


def is_propp_function_like(label: str) -> bool:
    return (
        "_" in label
        and '"' not in label
        and len(label) <= 90
        and bool(re.search(r"[A-Za-zΑ-ωβγδεζηθικλμνξοπρστυφχψω]", label))
    )


def is_role_like(label: str) -> bool:
    if '"' in label or len(label) > 50 or is_trivial_class_label(label):
        return False
    if re.search(r"\d", label):
        return False
    return bool(re.fullmatch(r"[A-Za-z_ '\-]+", label))


def compute_class_depth(cls, cache=None):
    if cache is None:
        cache = {}
    if cls in cache:
        return cache[cls]
    parents = [p for p in getattr(cls, 'is_a', []) if isinstance(p, ThingClass)]
    if not parents or cls.name == 'Thing':
        cache[cls] = 0
        return 0
    depths = []
    for p in parents:
        depths.append(compute_class_depth(p, cache))
    d = min(depths) + 1
    cache[cls] = d
    return d


def parse_swrl_expression(expr, var_map=None, visited=None, depth=0, max_depth=10):
    if var_map is None:
        var_map = {'x': '?x', 'y': '?y'}
    if visited is None:
        visited = set()
    expr_id = id(expr)
    if depth > max_depth or expr_id in visited:
        return []
    visited.add(expr_id)

    atoms = []
    if isinstance(expr, ThingClass):
        label = get_label(expr)
        if is_valid_label(label):
            atoms.append(f"{label}({var_map['x']})")
    elif isinstance(expr, Restriction):
        prop_label = get_label(expr.property)
        if not is_valid_label(prop_label):
            return []
        if expr.type in (SOME, ONLY, VALUE):
            filler = getattr(expr, 'value', None)
            if isinstance(filler, ThingClass):
                filler_label = get_label(filler)
                if is_valid_label(filler_label):
                    atoms.append(f"{prop_label}({var_map['x']}, {var_map['y']})")
                    atoms.append(f"{filler_label}({var_map['y']})")
        elif expr.type in (MIN, MAX, EXACTLY):
            atoms.append(f"{prop_label}({var_map['x']}, {var_map['y']})")
        else:
            logging.warning(f"Unsupported restriction type {expr.type} on {prop_label}")
    elif isinstance(expr, And):
        for part in expr.Classes:
            atoms.extend(parse_swrl_expression(part, var_map, visited, depth+1, max_depth))
    return atoms


def format_swrl_argument(arg) -> str:
    text = str(arg)
    if text.startswith("?"):
        return text
    try:
        return get_label(arg)
    except Exception:
        return text


def swrl_atom_to_text(atom):
    pred = getattr(atom, "class_predicate", None) or getattr(atom, "property_predicate", None)
    if pred is None:
        return None
    label = get_label(pred)
    if not is_valid_label(label):
        return None
    args = [format_swrl_argument(arg) for arg in (getattr(atom, "arguments", []) or [])]
    if not args:
        return None
    return f"{label}({', '.join(args)})"


def extract_explicit_swrl_rules(onto, max_rules: Optional[int] = None):
    rules = []
    try:
        raw_rules = list(onto.rules())
    except Exception as exc:
        logging.debug("Failed reading explicit SWRL rules: %s", exc)
        return rules
    for rule in raw_rules:
        if max_rules is not None and len(rules) >= max_rules:
            break
        body_atoms = [swrl_atom_to_text(atom) for atom in (getattr(rule, "body", []) or [])]
        head_atoms = [swrl_atom_to_text(atom) for atom in (getattr(rule, "head", []) or [])]
        body_atoms = [atom for atom in body_atoms if atom]
        head_atoms = [atom for atom in head_atoms if atom]
        if not body_atoms or not head_atoms:
            continue
        rules.append({
            "body": body_atoms,
            "head_atoms": head_atoms,
            "class": None,
            "label": getattr(rule, "name", "SWRL rule"),
            "depth": None,
            "rule_iri": str(getattr(rule, "iri", "")),
            "source": "explicit_swrl",
        })
    logging.info("Extracted %d explicit SWRL rules", len(rules))
    return rules


def extract_swrl_rules(onto, max_rules: Optional[int] = None):
    """Extract simple subclass-implied rules per class.
    For each C ⊑ D:
      - If D is a named class: head has D(?x)
      - If D is a restriction (e.g., ∃R.E): head may contain composite atoms like R(?x, ?y), E(?y)
      - Skip trivial Thing
    Returns list of dict with fields: body, head_atoms, class, label, depth.
    """
    rules = []
    depth_cache = {}
    for cls in onto.classes():
        if max_rules is not None and len(rules) >= max_rules:
            break
        lbl = get_label(cls)
        if not is_valid_label(lbl):
            continue
        d = compute_class_depth(cls, depth_cache)
        # 前提统一为命名类断言 C(?x)
        body_atoms = [f"{lbl}(?x)"]
        for parent in getattr(cls, 'is_a', []):
            head_atoms = parse_swrl_expression(parent)
            if not head_atoms:
                continue
            # 跳过平凡 Thing(?x)
            if any(a.startswith('Thing(') for a in head_atoms):
                # 若仅包含 Thing，则跳过；若包含其它原子，滤掉 Thing
                head_atoms = [a for a in head_atoms if not a.startswith('Thing(')]
                if not head_atoms:
                    continue
            rules.append({
                'body': body_atoms,
                'head_atoms': head_atoms,
                'class': cls,
                'label': lbl,
                'depth': d,
                'rule_iri': None,
                'source': 'axiom_derived'
            })
    return rules


def get_related_entities(onto, entity, max_items=50):
    """Collect related entities for distractors."""
    related = set()
    try:
        if isinstance(entity, ThingClass):
            for sup in getattr(entity, 'is_a', []):
                if isinstance(sup, ThingClass):
                    for sib in sup.subclasses():
                        if len(related) >= max_items: break
                        if sib is not entity:
                            related.add(get_label(sib))
            for sub in entity.subclasses():
                if len(related) >= max_items: break
                if sub is not entity:
                    related.add(get_label(sub))
        elif isinstance(entity, ObjectPropertyClass):
            for prop in onto.object_properties():
                if len(related) >= max_items: break
                if prop is not entity:
                    related.add(get_label(prop))
        elif isinstance(entity, DataPropertyClass):
            for prop in onto.data_properties():
                if len(related) >= max_items: break
                if prop is not entity:
                    related.add(get_label(prop))
    except Exception as e:
        logging.warning(f"Error in get_related_entities: {e}")
    return list(related)


def get_swrl_distractors(atom, onto, all_preds, all_classes, num_choices=3):
    """Generate distractors for a single-atom head."""
    pred = atom.split('(')[0]
    vars_ = [v.strip() for v in atom[atom.find('(')+1:atom.find(')')].split(',')]
    distractors = set()
    pool = all_preds + all_classes
    random.shuffle(pool)
    entity = onto.search_one(iri=f"*{pred}") or \
             next((c for c in onto.classes() if get_label(c)==pred), None)
    if entity:
        for r in get_related_entities(onto, entity):
            distractors.add(f"{r}({', '.join(vars_)})")
            if len(distractors) >= num_choices:
                break
    i = 0
    while len(distractors) < num_choices and i < len(pool)*3:
        p = pool[i % len(pool)]
        cand = f"{p}({', '.join(vars_)})"
        if cand != atom and cand not in distractors:
            distractors.add(cand)
        i += 1
    if len(vars_) == 2:
        distractors.add(f"not {pred}({', '.join(vars_)})")
        distractors.add(f"{pred}({vars_[1]}, {vars_[0]})")
    # 保证数量
    dis_list = [d for d in distractors if d != atom and is_valid_atom(d)]
    random.shuffle(dis_list)
    return dis_list[:num_choices]

def get_composite_distractors(correct, all_preds, all_classes, num_choices=3):
    parts = [p.strip() for p in correct.split('∧')]
    distractors = []
    seen = {normalize_atom(correct)}
    unary = [atom for atom in parts if atom_variables(atom) == ['?y']]
    binary = [atom for atom in parts if atom_variables(atom) == ['?x', '?y']]
    if unary and binary:
        unary_atom = unary[0]
        binary_atom = binary[0]
        unary_pred = atom_predicate(unary_atom)
        if is_propp_function_like(unary_pred):
            role_pool = [
                label for label in all_classes
                if label != unary_pred and is_valid_label(label) and is_propp_function_like(label)
            ]
        else:
            role_pool = [
                label for label in all_classes
                if label != unary_pred and is_valid_label(label) and is_role_like(label)
            ]
        pred_pool = [
            label for label in all_preds
            if is_valid_label(label)
            and label != atom_predicate(binary_atom)
        ]
        random.shuffle(role_pool)
        random.shuffle(pred_pool)
        for label in role_pool:
            txt = " ∧ ".join(sorted([f"{label}(?y)", binary_atom]))
            key = normalize_atom(txt)
            if key not in seen:
                seen.add(key)
                distractors.append(txt)
            if len(distractors) >= 2:
                break
        for label in pred_pool:
            txt = " ∧ ".join(sorted([unary_atom, f"{label}(?x, ?y)"]))
            key = normalize_atom(txt)
            if key not in seen:
                seen.add(key)
                distractors.append(txt)
            if len(distractors) >= num_choices:
                break
    pool = [
        label for label in all_classes
        if is_valid_label(label) and (is_role_like(label) or is_propp_function_like(label))
    ]
    i = 0
    while len(distractors) < num_choices and i < len(pool) * 3:
        a = random.choice(pool)
        b = random.choice(pool)
        if not (is_valid_label(a) and is_valid_label(b)):
            i += 1
            continue
        vars_ = atom_variables(parts[0]) or ['?y']
        txt = f"{a}({', '.join(vars_)}) ∧ {b}({', '.join(vars_)})"
        key = normalize_atom(txt)
        if key not in seen:
            seen.add(key)
            distractors.append(txt)
        i += 1
    return distractors[:num_choices]


def atom_predicate(atom: str) -> str:
    return atom.split('(')[0].strip()


def atom_variables(atom: str) -> List[str]:
    if '(' not in atom or ')' not in atom:
        return []
    return [v.strip() for v in atom[atom.find('(')+1:atom.find(')')].split(',')]


def reverse_characterization_head(head_atoms: List[str], class_label: str):
    """Build a conservative reverse-pattern question for R(?x,?y) ∧ C(?y).

    This is phrased as characterization rather than formal entailment because
    many ontology restrictions are necessary conditions, not equivalence axioms.
    """
    if len(head_atoms) != 2:
        return None
    binary = [atom for atom in head_atoms if len(atom_variables(atom)) == 2]
    unary_y = [atom for atom in head_atoms if atom_variables(atom) == ['?y']]
    if len(binary) != 1 or len(unary_y) != 1:
        return None
    relation_label = atom_predicate(binary[0]).lower()
    class_y_label = atom_predicate(unary_y[0]).lower()
    if class_y_label in relation_label or relation_label.endswith(class_y_label):
        return None
    pattern = " ∧ ".join(sorted([binary[0], unary_y[0]]))
    return pattern, f"{class_label}(?x)"


def get_reverse_class_distractors(correct_class_atom, all_classes, num_choices=3):
    correct_label = atom_predicate(correct_class_atom)
    if "_" in correct_label:
        pool = [
            label for label in all_classes
            if label != correct_label
            and "_" in label
            and '"' not in label
            and is_valid_label(label)
            and not is_trivial_class_label(label)
        ]
    else:
        pool = [
            label for label in all_classes
            if label != correct_label
            and '"' not in label
            and is_valid_label(label)
            and not is_trivial_class_label(label)
        ]
    random.shuffle(pool)
    selected = []
    for label in pool:
        atom = f"{label}(?x)"
        if atom != correct_class_atom and atom not in selected:
            selected.append(atom)
        if len(selected) >= num_choices:
            break
    return selected[:num_choices]


def generate_swrl_questions(rules, onto, all_preds, all_classes, max_q=None):
    """Build MCQs from rules with unified metadata."""
    questions = []
    letters = ['A', 'B', 'C', 'D']
    for idx, r in enumerate(rules):
        if max_q and len(questions) >= max_q:
            break

        body = r['body']
        if any(not is_valid_atom(atom) for atom in body):
            continue
        head_atoms = r.get('head_atoms', [])
        if not head_atoms:
            continue
        if any(not is_valid_atom(atom) for atom in head_atoms):
            continue
        # 复合头部（来自限制）
        if len(head_atoms) >= 2:
            if any(is_trivial_single_atom(atom) for atom in body):
                continue
            head = " ∧ ".join(sorted(head_atoms[:2])) if len(head_atoms) == 2 else " ∧ ".join(sorted(head_atoms))
            prompt = f"Suppose an individual ?x satisfies: {' and '.join(body)}. Which composite conclusion is inferred?"
            distractors = get_composite_distractors(head, all_preds, all_classes, 3)
            opts = distractors + [head]
            reverse = None
        else:
            head = head_atoms[0]
            if head.startswith('Thing('):
                continue
            if head in body:
                continue
            if is_trivial_single_atom(head) or any(is_trivial_single_atom(atom) for atom in body):
                continue
            prompt = f"Suppose an individual ?x satisfies: {' and '.join(body)}. Which conclusion is inferred?"
            distractors = get_swrl_distractors(head, onto, all_preds, all_classes, 3)
            opts = distractors + [head]
            reverse = None

        random.shuffle(opts)
        # 去重并确保数量=4
        uniq_opts = []
        seen = set()
        for o in opts:
            if o not in seen:
                uniq_opts.append(o)
                seen.add(o)
        if head not in seen:
            uniq_opts.append(head)
        # 补齐或裁剪到4项
        if len(uniq_opts) < 4:
            # 无法保证唯一性与足够干扰项，跳过该题
            continue
        uniq_opts = uniq_opts[:4]
        if head not in uniq_opts:
            # 确保包含正确答案
            uniq_opts[-1] = head
        if any(option in body for option in uniq_opts):
            continue
        correct = letters[uniq_opts.index(head)]
        subject = r.get('class')
        subject_iri = str(subject.iri) if subject is not None and hasattr(subject, 'iri') else r.get('rule_iri')
        subject_label = r.get('label') or r.get('source') or 'SWRL rule'
        questions.append({
            'prompt': prompt,
            'options': uniq_opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': subject_iri,
                'subject_label': subject_label,
                'subject_kind': 'rule' if r.get('source') == 'explicit_swrl' else 'class',
                'relation': 'swrl_rule' if r.get('source') == 'explicit_swrl' else 'swrl_like_axiom_rule',
                'object_iri': None,
                'object_label': None,
                'object_kind': None,
                'class_context_iri': str(subject.iri) if subject is not None and hasattr(subject, 'iri') else None,
                'class_context_label': r['label'] if subject is not None else None,
                'depth': r['depth'],
                'sibling_count': None,
                'subclass_count': None,
                'parent_count': None,
                'rule_source': r.get('source', 'unknown'),
            }
        })

        if max_q and len(questions) >= max_q:
            break
        if reverse:
            reverse_body, reverse_head = reverse
            reverse_distractors = get_reverse_class_distractors(reverse_head, all_classes, 3)
            reverse_opts = reverse_distractors + [reverse_head]
            random.shuffle(reverse_opts)
            reverse_uniq = []
            reverse_seen = set()
            for option in reverse_opts:
                if option not in reverse_seen:
                    reverse_uniq.append(option)
                    reverse_seen.add(option)
            if len(reverse_uniq) == 4 and reverse_head in reverse_uniq:
                reverse_correct = letters[reverse_uniq.index(reverse_head)]
                questions.append({
                    'prompt': f"Suppose a pattern holds: {reverse_body}. Which class is best characterized by this pattern?",
                    'options': reverse_uniq,
                    'correct_answer': reverse_correct,
                    'meta': {
                        'subject_iri': subject_iri,
                        'subject_label': subject_label,
                        'subject_kind': 'class',
                        'relation': 'swrl_rule_reverse_characterization',
                        'object_iri': None,
                        'object_label': None,
                        'object_kind': None,
                        'class_context_iri': str(subject.iri) if subject is not None and hasattr(subject, 'iri') else None,
                        'class_context_label': r['label'] if subject is not None else None,
                        'depth': r['depth'],
                        'sibling_count': None,
                        'subclass_count': None,
                        'parent_count': None,
                        'rule_source': r.get('source', 'unknown'),
                    }
                })

    return questions


def save_questions(questions, save_path: Path):
    save_json(questions, save_path, description="questions")


def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    max_questions: int,
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    concept_scope: str,
    no_warnings: bool,
) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f"swrl_questions_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "swrl_questions", safe_stem)
    if save_path.exists():
        logging.info("Skip existing: %s", save_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    timeout = 600
    signal.alarm(timeout)
    try:
        world = owlready2.World()
        configure_world_paths(world, onto_paths)
        onto = load_ontology(world, file_path, load_imports=load_imports)
        if onto is None:
            return

        rule_limit = None
        if max_questions:
            rule_limit = max_questions * 3
        explicit_rules = extract_explicit_swrl_rules(onto, max_rules=rule_limit)
        derived_rules = extract_swrl_rules(onto, max_rules=rule_limit)
        rules = explicit_rules + derived_rules
        if concept_scope != 'all':
            def is_native(c):
                return getattr(getattr(c, 'namespace', None), 'ontology', None) is onto
            def is_imported(c):
                o = getattr(getattr(c, 'namespace', None), 'ontology', None)
                return (o is not None) and (o is not onto)
            if concept_scope == 'native':
                rules = [r for r in rules if r.get('class') is None or is_native(r['class'])]
            else:
                rules = [r for r in rules if r.get('class') is None or is_imported(r['class'])]
        if not rules:
            save_empty_marker(empty_path, source_file=file_path, reason="no_swrl_or_swrl_like_rules")
            return
        all_preds = [get_label(p) for p in onto.object_properties() if is_valid_label(get_label(p))]
        all_classes = [get_label(c) for c in onto.classes() if get_label(c) != 'Thing' and is_valid_label(get_label(c))]
        qs = generate_swrl_questions(rules, onto, all_preds, all_classes, max_questions * 2 if max_questions else max_questions)
        qs = limit_questions_by_subject(qs, max_questions)
        if not qs:
            save_empty_marker(
                empty_path,
                source_file=file_path,
                reason="no_valid_swrl_rule_questions",
                extra={"rules": len(rules)},
            )
            return
        save_questions(qs, save_path)
    except OntologyTimeoutError:
        logging.error(f"Timeout processing {file_path} after {timeout}s")
    finally:
        signal.alarm(0)
        try:
            world.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Generate SWRL-like rule reasoning MCQs from subclass axioms.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=500, help='Max questions per ontology.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of subject classes.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()
    configure_logging(args.log, "process_2_4.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    files, input_root = discover_ontology_files(input_path, EXTENSIONS)
    onto_paths = resolve_onto_paths(args.onto_path)
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                max_questions=args.max_questions,
                load_imports=not args.no_imports,
                onto_paths=onto_paths,
                concept_scope=args.concept_scope,
                no_warnings=args.no_warnings,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()
