import json
import signal
import os
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import (
    ThingClass, Restriction, And,
    SOME, ONLY, VALUE, MIN, MAX, EXACTLY,
    DataPropertyClass, ObjectPropertyClass, get_ontology, onto_path, set_log_level
)

class OntologyTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise OntologyTimeoutError()

signal.signal(signal.SIGALRM, _timeout_handler)


MAX_QUESTIONS = 500
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')


def get_label(entity):
    """Readable label: prefer rdfs:label, else name, else str(entity)."""
    if hasattr(entity, 'label') and entity.label:
        return entity.label[0]
    if hasattr(entity, 'name'):
        return entity.name
    return str(entity)


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
        atoms.append(f"{get_label(expr)}({var_map['x']})")
    elif isinstance(expr, Restriction):
        prop_label = get_label(expr.property)
        if expr.type in (SOME, ONLY, VALUE):
            filler = getattr(expr, 'value', None)
            if isinstance(filler, ThingClass):
                atoms.append(f"{prop_label}({var_map['x']}, {var_map['y']})")
                atoms.append(f"{get_label(filler)}({var_map['y']})")
        elif expr.type in (MIN, MAX, EXACTLY):
            atoms.append(f"{prop_label}({var_map['x']}, {var_map['y']})")
        else:
            logging.warning(f"Unsupported restriction type {expr.type} on {prop_label}")
    elif isinstance(expr, And):
        for part in expr.Classes:
            atoms.extend(parse_swrl_expression(part, var_map, visited, depth+1, max_depth))
    return atoms


def extract_swrl_rules(onto):
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
        lbl = get_label(cls)
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
                'depth': d
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
        if cand != atom:
            distractors.add(cand)
        i += 1
    if len(vars_) == 2:
        distractors.add(f"not {pred}({', '.join(vars_)})")
        distractors.add(f"{pred}({vars_[1]}, {vars_[0]})")
    # 保证数量
    dis_list = [d for d in distractors if d != atom]
    random.shuffle(dis_list)
    return dis_list[:num_choices]

def get_composite_distractors(correct, all_preds, all_classes, num_choices=3):
    parts = [p.strip() for p in correct.split('∧')]
    vars_ = [v.strip() for v in parts[0][parts[0].find('(')+1:parts[0].find(')')].split(',')]
    distractors = set()
    pool = all_preds + all_classes
    i = 0
    while len(distractors) < num_choices and i < len(pool)*3:
        a = random.choice(pool)
        b = random.choice(pool)
        txt = f"{a}({', '.join(vars_)}) ∧ {b}({', '.join(vars_)})"
        if txt != correct:
            distractors.add(txt)
        i += 1
    # 加入拆分、顺序翻转
    if len(parts) == 2:
        rev = f"{parts[1]} ∧ {parts[0]}"
        distractors.add(rev)
    distractors.add(parts[0])
    if len(parts) > 1:
        distractors.add(parts[1])
    dis_list = [d for d in distractors if d != correct]
    random.shuffle(dis_list)
    return dis_list[:num_choices]


def generate_swrl_questions(rules, onto, all_preds, all_classes, max_q=None):
    """Build MCQs from rules with unified metadata."""
    questions = []
    letters = ['A', 'B', 'C', 'D']
    for idx, r in enumerate(rules):
        if max_q and len(questions) >= max_q:
            break

        body = r['body']
        head_atoms = r.get('head_atoms', [])
        if not head_atoms:
            continue
        # 复合头部（来自限制）
        if len(head_atoms) >= 2:
            head = " ∧ ".join(head_atoms[:2]) if len(head_atoms) == 2 else " ∧ ".join(head_atoms)
            prompt = f"Suppose an individual ?x satisfies: {' and '.join(body)}. Which composite conclusion is inferred?"
            distractors = get_composite_distractors(head, all_preds, all_classes, 3)
            opts = distractors + [head]
        else:
            head = head_atoms[0]
            if head.startswith('Thing('):
                continue
            prompt = f"Suppose an individual ?x satisfies: {' and '.join(body)}. Which conclusion is inferred?"
            distractors = get_swrl_distractors(head, onto, all_preds, all_classes, 3)
            opts = distractors + [head]

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
        correct = letters[uniq_opts.index(head)]
        questions.append({
            'prompt': prompt,
            'options': uniq_opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri': str(r['class'].iri),
                'subject_label': r['label'],
                'subject_kind': 'class',
                'relation': 'swrl_rule',
                'object_iri': None,
                'object_label': None,
                'object_kind': None,
                'class_context_iri': str(r['class'].iri),
                'class_context_label': r['label'],
                'depth': r['depth'],
                'sibling_count': None,
                'subclass_count': None,
                'parent_count': None,
            }
        })

    return questions


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
        json.dump(questions, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(questions)} questions to {save_path}")


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
    timeout = 120
    signal.alarm(timeout)
    try:
        if onto_paths:
            for p in onto_paths:
                pp = str(Path(p).resolve())
                if pp not in onto_path:
                    onto_path.append(pp)
        onto = get_ontology(f"file://{file_path.resolve()}")
        try:
            if load_imports:
                onto = onto.load()
            else:
                onto = onto.load(only_local=True)
        except Exception as e:
            if load_imports:
                logging.warning(f"Failed loading with imports; retrying local-only: {file_path} ({e})")
                try:
                    onto = get_ontology(f"file://{file_path.resolve()}").load(only_local=True)
                except Exception as e2:
                    logging.error(f"Failed local-only: {file_path} ({e2})")
                    return
            else:
                logging.error(f"Failed local-only: {file_path} ({e})")
                return

        rules = extract_swrl_rules(onto)
        if concept_scope != 'all':
            def is_native(c):
                return getattr(getattr(c, 'namespace', None), 'ontology', None) is onto
            def is_imported(c):
                o = getattr(getattr(c, 'namespace', None), 'ontology', None)
                return (o is not None) and (o is not onto)
            if concept_scope == 'native':
                rules = [r for r in rules if is_native(r['class'])]
            else:
                rules = [r for r in rules if is_imported(r['class'])]
        if not rules:
            return
        all_preds = [get_label(p) for p in onto.object_properties()] + [get_label(p) for p in onto.data_properties()]
        all_classes = [get_label(c) for c in onto.classes() if get_label(c) != 'Thing']
        qs = generate_swrl_questions(rules, onto, all_preds, all_classes, max_questions)
        if not qs:
            return
        try:
            rel = file_path.relative_to(input_root)
        except Exception:
            rel = file_path.name
        rel_parts = list(Path(rel).parts)
        safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
        safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
        out_dir = output_root.joinpath(*safe_parts, safe_stem)
        save_path = out_dir / f"swrl_questions_{safe_stem}.json"
        save_questions(qs, save_path)
    except OntologyTimeoutError:
        logging.error(f"Timeout processing {file_path} after {timeout}s")
    finally:
        signal.alarm(0)


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
    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('process_2_4.log','w','utf-8')])
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
    files: List[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in EXTENSIONS:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, fnames in os.walk(str(input_path)):
            for f in fnames:
                if f.lower().endswith(EXTENSIONS):
                    files.append(Path(root)/f)
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                max_questions=args.max_questions,
                load_imports=not args.no_imports,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                concept_scope=args.concept_scope,
                no_warnings=args.no_warnings,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()
