import json
import os
import random
import logging
import uuid
from collections import deque

import rdflib
import owlready2
from owlready2 import *

CONFIG = {
    'BASE_DIR': '../../../data',
    'EXTENSIONS': ('.owl', '.rdf', '.rdfs', '.ttl', '.xml', '.n3'),
    'MAX_SUBGRAPH_SIZE': 15,
    'MIN_SUBGRAPH_SIZE': 8,
    'DEPTH_OPTIONS': [2, 3, 4, 5, 6, 7, 8],
    'MAX_SUBGRAPH_RETRIES': 20,
    'NUM_PROPERTY_SETS_MAX': 100,
    'PROPERTIES_PER_SET_MAX': 10,
    'MIN_PROPERTIES_PER_SET': 5
}

if not hasattr(owlready2.World, '_get_obj_triples'):
    def _stub_get_obj_triples(self, *args, **kwargs):
        return []
    owlready2.World._get_obj_triples = _stub_get_obj_triples
if not hasattr(owlready2.World, '_get_obj_triples_cspo_cspo'):
    owlready2.World._get_obj_triples_cspo_cspo = owlready2.World._get_obj_triples

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ontology_processing.log'),
        logging.StreamHandler()
    ]
)

definition_cache = {}

def get_label(entity):
    labels = getattr(entity, 'label', []) or []
    return labels[0] if labels else getattr(entity, 'name', str(entity))

def get_comment(entity):
    comments = getattr(entity, 'comment', []) or []
    return comments[0] if comments else None

def get_definition(entity):
    try:
        key = str(entity.iri)
        if key in definition_cache:
            return definition_cache[key]
        definition = None
        defs = getattr(entity, 'IAO_0000115', None)
        if defs:
            definition = next((d for d in defs if getattr(d, 'lang', None) == 'en'), defs[0])
        if not definition:
            skos_defs = getattr(entity, 'definition', None)
            if skos_defs:
                definition = next((d for d in skos_defs if getattr(d, 'lang', None) == 'en'), skos_defs[0])
        if not definition:
            comment = getattr(entity, 'comment', None)
            if comment:
                if isinstance(comment, list) and comment:
                    definition = next((d for d in comment if getattr(d, 'lang', None) == 'en'), comment[0])
                else:
                    definition = comment
        definition = str(definition).strip() if definition and str(definition).strip() else 'No definition provided.'
        definition_cache[key] = definition
        return definition
    except Exception as e:
        logging.warning(f'Error retrieving definition for {entity}: {e}')
        return 'No definition provided.'

def is_valid_property(prop, onto):
    return ((isinstance(prop, owlready2.ObjectPropertyClass) or
             isinstance(prop, owlready2.DataPropertyClass)) and
            prop in onto.properties())

def get_subgraph_around_properties(onto, input_properties, depth=2):
    obj_props = [p for p in input_properties if is_valid_property(p, onto)]

    initial_classes = set()
    for prop in obj_props:
        for d in prop.domain:
            if isinstance(d, ThingClass) and d != Thing:
                initial_classes.add(d)
        for r in prop.range:
            if isinstance(r, ThingClass) and r != Thing:
                initial_classes.add(r)

    for attempt in range(CONFIG['MAX_SUBGRAPH_RETRIES'] + 1):
        related_classes = set()
        constraints = []
        annotations = {}
        visited = set()
        queue = deque([(cls, 0) for cls in initial_classes])
        target = depth

        while queue and len(related_classes) < CONFIG['MAX_SUBGRAPH_SIZE']:
            cls, d = queue.popleft()
            if cls in visited:
                continue
            visited.add(cls)
            related_classes.add(cls)
            if (cmt := get_comment(cls)):
                annotations[cls] = cmt
            if d < target:
                for sup in cls.is_a:
                    if isinstance(sup, ThingClass) and sup != Thing:
                        queue.append((sup, d+1))
                for sub in cls.subclasses():
                    queue.append((sub, d+1))
        if len(related_classes) < CONFIG['MIN_SUBGRAPH_SIZE']:
            logging.warning(f'Subgraph too small ({len(related_classes)}), retrying')
            target = min(target+1, max(CONFIG['DEPTH_OPTIONS']))
            continue
        for prop in obj_props:
            for domain in prop.domain:
                if domain in related_classes:
                    constraints.append((prop, 'domain', domain))
            for range_ in prop.range:
                if range_ in related_classes:
                    constraints.append((prop, 'range', range_))
            if getattr(prop, 'is_functional', False):
                constraints.append((prop, 'functional', True))
        if constraints:
            logging.info(f'Subgraph ready with {len(related_classes)} classes and {len(constraints)} constraints')
            return related_classes, constraints, annotations
        target = min(target+1, max(CONFIG['DEPTH_OPTIONS']))
        logging.warning(f'No constraints found, retrying at depth {target}')

    logging.error('Failed to build constraint subgraph')
    return set(), [], {}

def generate_constraint_triples(onto, input_properties):
    classes, constraints, annotations = get_subgraph_around_properties(onto, input_properties)
    if not classes or not constraints:
        logging.warning('Empty classes or constraints, skip')
        return None

    triples = []
    for prop, ctype, val in sorted(
        constraints,
        key=lambda x: (get_label(x[0]), x[1], get_label(x[2]) if isinstance(x[2], ThingClass) else str(x[2]))
    ):
        p_lbl = get_label(prop)
        v_lbl = 'True' if ctype == 'functional' else get_label(val)
        triples.append({'triple': (p_lbl, ctype, v_lbl), 'text': f'{p_lbl} {ctype} {v_lbl}.'})

    classes_def = []
    for c in sorted(classes, key=get_label):
        dfn = get_definition(c)
        classes_def.append(f'{get_label(c)}: {dfn}' if dfn != 'No definition provided.' else get_label(c))
    props_lbl = [get_label(p) for p in input_properties if is_valid_property(p, onto)]

    return {
        'classes': classes_def,
        'properties': props_lbl,
        'triples': triples,
        'annotations': {get_label(e): annotations[e] for e in annotations}
    }

def describe_constraint_task(classes, properties, annotations):
    lines = ['## Property Constraint Learning Task',
             'Given classes and properties, generate property constraints.']
    lines.append('### Classes')
    for c in classes:
        lines.append(f'- **{c}**')
    lines.append('### Properties')
    for p in properties:
        lines.append(f'- **{p}**')
    if annotations:
        lines.append('\n### Note')
        for k, v in annotations.items():
            lines.append(f'- **{k}**: {v}')
    lines.append('\n### Task')
    lines.append('- Generate domain and range constraints.')
    lines.append('- Generate functional constraints.')
    return '\n'.join(lines)

def process_for_constraint_task(
    file_path,
    num_property_sets_max=CONFIG['NUM_PROPERTY_SETS_MAX'],
    properties_per_set_max=CONFIG['PROPERTIES_PER_SET_MAX']
):
    logging.info(f'Processing: {file_path}')
    onto = load_ontology_with_fallback(file_path)
    if not onto:
        raise RuntimeError(f'Load failed: {file_path}')
    all_props = [p for p in onto.properties() if is_valid_property(p, onto)]
    if len(all_props) < CONFIG['MIN_PROPERTIES_PER_SET']:
        logging.warning(f'Not enough properties ({len(all_props)}) to generate tasks')
        return

    per = min(properties_per_set_max, len(all_props))
    count = min(num_property_sets_max, max(1, len(all_props)//CONFIG['MIN_PROPERTIES_PER_SET']))
    logging.info(f'Generating {count} tasks with up to {per} properties each')
    tasks = []

    for _ in range(count):
        props = random.sample(all_props, per)
        if len(props) < CONFIG['MIN_PROPERTIES_PER_SET']:
            continue
        data = generate_constraint_triples(onto, props)
        if not data:
            continue
        desc = describe_constraint_task(data['classes'], data['properties'], data['annotations'])
        tasks.append({
            'task_description': desc,
            'classes': data['classes'],
            'properties': data['properties'],
            'triples': data['triples']
        })

    if not tasks:
        logging.warning('No tasks generated; skipping save')
        return

    out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_3_5')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'constraint_{os.path.basename(file_path)}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    logging.info(f'Saved {len(tasks)} tasks to {out_path}')

def rdflib_to_owlready(g):
    tmp = f'tmp_{uuid.uuid4()}.owl'
    g.serialize(tmp, format='xml')
    onto = get_ontology(f'file://{os.path.abspath(tmp)}').load()
    os.remove(tmp)
    return onto


def load_ontology_with_fallback(file_path):
    try:
        return get_ontology(f'file://{os.path.abspath(file_path)}').load()
    except:
        g = rdflib.Graph()
        for fmt in ['xml', 'turtle', 'n3', 'trig']:
            try:
                g.parse(file_path, format=fmt)
                return rdflib_to_owlready(g)
            except:
                continue
    return None


if __name__ == '__main__':
    random.seed(42)
    files = []
    for root, _, fs in os.walk(CONFIG['BASE_DIR']):
        for fn in fs:
            if fn.lower().endswith(CONFIG['EXTENSIONS']):
                files.append(os.path.join(root, fn))
    for fp in files:
        try:
            process_for_constraint_task(fp)
        except Exception as e:
            logging.error(f'Error: {e}')
