import json
import signal
import os
import random
import logging
from owlready2 import (
    ThingClass, Restriction, And,
    SOME, ONLY, VALUE, MIN, MAX, EXACTLY,
    DataPropertyClass, ObjectPropertyClass, get_ontology
)

class OntologyTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise OntologyTimeoutError()

signal.signal(signal.SIGALRM, _timeout_handler)


MAX_QUESTIONS = 500
BASE_DIR = '../../../data'
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')


def get_label(entity):
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
    rules = []
    depth_cache = {}
    for cls in onto.classes():
        lbl = get_label(cls)
        d = compute_class_depth(cls, depth_cache)
        for parent in cls.is_a:
            body_atoms = parse_swrl_expression(cls)
            head_atoms = parse_swrl_expression(parent)
            if body_atoms and head_atoms:
                rules.append({
                    'body': body_atoms,
                    'head': head_atoms[0],
                    'class': cls,
                    'label': lbl,
                    'depth': d
                })
    return rules


def get_related_entities(onto, entity, max_items=50):
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
    while len(distractors) < num_choices and i < len(pool)*2:
        p = pool[i % len(pool)]
        distractors.add(f"{p}({', '.join(vars_)})")
        i += 1
    if len(vars_) == 2:
        distractors.add(f"not {pred}({', '.join(vars_)})")
        distractors.add(f"{pred}({vars_[1]}, {vars_[0]})")
    return random.sample(distractors, min(num_choices, len(distractors)))

def get_composite_distractors(correct, all_preds, all_classes, num_choices=3):
    parts = [p.strip() for p in correct.split('∧')]
    vars_ = [v.strip() for v in parts[0][parts[0].find('(')+1:parts[0].find(')')].split(',')]
    distractors = set()
    pool = all_preds + all_classes
    i = 0
    while len(distractors) < num_choices and i < len(pool)*2:
        a = random.choice(pool)
        b = random.choice(pool)
        txt = f"{a}({', '.join(vars_)}) ∧ {b}({', '.join(vars_)})"
        if txt != correct:
            distractors.add(txt)
        i += 1
    if len(parts) == 2:
        rev = f"{parts[1]} ∧ {parts[0]}"
        distractors.add(rev)
    distractors.add(parts[0])
    if len(parts) > 1:
        distractors.add(parts[1])
    return random.sample(list(distractors), min(num_choices, len(distractors)))


def generate_swrl_questions(rules, onto, all_preds, all_classes, max_q=None):
    questions = []
    letters = ['A', 'B', 'C', 'D']
    for idx, r in enumerate(rules):
        if max_q and len(questions) >= max_q:
            break

        if len(rules) >= 2 and random.random() < 0.2:
            r2 = random.choice(rules)
            body = r['body'] + r2['body']
            head = f"{r['head']} ∧ {r2['head']}"
            if head.startswith('Thing('):
                continue
            prompt = f"Consider an individual ?x that satisfies: {' and '.join(body)}. Which composite inference follows?"
            opts = get_composite_distractors(head, all_preds, all_classes, 3) + [head]
        else:
            body = r['body']
            head = r['head']
            if head.startswith('Thing('):
                continue
            prompt = f"Suppose an individual ?x satisfies: {' and '.join(body)}. Which conclusion is inferred?"
            opts = get_swrl_distractors(head, onto, all_preds, all_classes, 3) + [head]

        random.shuffle(opts)
        correct = letters[opts.index(head)]
        questions.append({
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'class_iri': str(r['class'].iri),
                'label': r['label'],
                'depth': r['depth']
            }
        })

    return questions


def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(questions)} questions to {save_path}")


def process_owl_file(file_path, max_q=None):
    timeout = 120
    signal.alarm(timeout)
    try:
        onto = get_ontology(f"file://{os.path.abspath(file_path)}").load()
        rules = extract_swrl_rules(onto)
        if not rules:
            logging.warning(f"No rules in {file_path}")
            return
        all_preds = [get_label(p) for p in onto.object_properties()] + [get_label(p) for p in onto.data_properties()]
        all_classes = [get_label(c) for c in onto.classes()]
        qs = generate_swrl_questions(rules, onto, all_preds, all_classes, max_q)
        if not qs:
            logging.warning(f"No questions for {file_path}")
            return
        out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_2_4')
        save_questions(qs, os.path.join(out_dir, f"swrl_questions_{os.path.basename(file_path)}.json"))
    except OntologyTimeoutError:
        logging.error(f"Timeout processing {file_path} after {timeout}s")
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')
    random.seed(42)
    for root, _, files in os.walk(BASE_DIR):
        for fname in files:
            if fname.lower().endswith(EXTENSIONS):
                process_owl_file(os.path.join(root, fname), MAX_QUESTIONS)
