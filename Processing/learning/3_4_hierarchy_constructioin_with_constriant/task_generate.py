import json
import os
import random
import logging
import uuid
import argparse
import warnings
from pathlib import Path
from collections import deque

import rdflib
import owlready2
from owlready2 import *

# ---------- configuration ----------
CONFIG = {
    'EXTENSIONS': ('.owl', '.rdf', '.rdfs', '.ttl', '.xml', '.n3'),
    'MAX_SUBGRAPH_SIZE': 15,  # 最大子图大小（上限）
    'MIN_SUBGRAPH_SIZE': 8,   # 最小子图大小（下限）
    'DEPTH_OPTIONS': [2, 3, 4, 5, 6, 7, 8],
    'MAX_SUBGRAPH_RETRIES': 20,
    'NUM_PROPERTY_SETS_MAX': 100,
    'PROPERTIES_PER_SET_MAX': 10,  # 最大类集大小（上限）
    'MIN_PROPERTIES_PER_SET': 5    # 最小类集大小（下限）
}

# ---------- 兼容性补丁 ----------
if not hasattr(owlready2.World, '_get_obj_triples'):
    def _stub_get_obj_triples(self, *args, **kwargs):
        return []
    owlready2.World._get_obj_triples = _stub_get_obj_triples
if not hasattr(owlready2.World, '_get_obj_triples_cspo_cspo'):
    owlready2.World._get_obj_triples_cspo_cspo = owlready2.World._get_obj_triples

# ---------- logging ----------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ontology_processing.log'),
        logging.StreamHandler()
    ]
)

# ---------- 全局缓存 ----------
definition_cache = {}

# ---------- helpers ----------
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

# ---------- subgraph extraction ----------
def get_subgraph_around_properties(onto, input_properties, depth=2):
    obj_props = [p for p in input_properties if is_valid_property(p, onto)]

    # 初始类别集
    initial_classes = set()
    for prop in obj_props:
        for d in prop.domain:
            if isinstance(d, ThingClass) and d != Thing:
                initial_classes.add(d)
        for r in prop.range:
            if isinstance(r, ThingClass) and r != Thing:
                initial_classes.add(r)

    current_depth = min(depth, max(CONFIG['DEPTH_OPTIONS']))
    for attempt in range(CONFIG['MAX_SUBGRAPH_RETRIES'] + 1):
        related_classes = set()
        constraints = []
        annotations = {}
        visited = set()
        queue = deque([(cls, 0) for cls in initial_classes])
        target = current_depth

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
        # 检查最小子图规模
        if len(related_classes) < CONFIG['MIN_SUBGRAPH_SIZE']:
            logging.warning(f'Subgraph too small ({len(related_classes)}), retrying with deeper search')
            current_depth = min(current_depth + 1, max(CONFIG['DEPTH_OPTIONS']))
            continue
        # 收集约束
        for prop in obj_props:
            for domain in prop.domain:
                if domain in related_classes:
                    constraints.append((prop, 'domain', domain))
            for range_ in prop.range:
                if range_ in related_classes:
                    constraints.append((prop, 'range', range_))
            if isinstance(prop, FunctionalProperty):
                constraints.append((prop, 'functional', True))
        if constraints:
            logging.info(f'Subgraph ready with {len(related_classes)} classes and {len(constraints)} constraints')
            return related_classes, constraints, annotations
        current_depth = min(current_depth + 1, max(CONFIG['DEPTH_OPTIONS']))
        logging.warning(f'No constraints found, retrying at depth {current_depth}')

    logging.error('Failed to build constraint subgraph')
    return set(), [], {}

# ---------- constraint extraction ----------
def generate_constraint_triples(onto, input_properties):
    classes, constraints, annotations = get_subgraph_around_properties(onto, input_properties)
    if not classes or not constraints:
        logging.warning('Empty classes or constraints, skip')
        return None

    # 格式化三元组（携带 IRI 元信息，保证唯一性）
    triples = []
    for prop, ctype, val in sorted(
        constraints,
        key=lambda x: (get_label(x[0]), x[1], get_label(x[2]) if isinstance(x[2], ThingClass) else str(x[2]))
    ):
        p_lbl = get_label(prop)
        v_lbl = 'True' if ctype == 'functional' else get_label(val)
        triples.append({
            'triple': (p_lbl, ctype, v_lbl),
            'text': f'{p_lbl} {ctype} {v_lbl}.',
            'meta': {
                'property_iri': str(prop.iri) if hasattr(prop, 'iri') else None,
                'value_iri': (str(val.iri) if isinstance(val, ThingClass) and hasattr(val, 'iri') else None)
            }
        })

    # 类定义列表
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

# ---------- task description ----------
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
    file_path: Path,
    input_root: Path,
    output_root: Path,
    num_property_sets_max=CONFIG['NUM_PROPERTY_SETS_MAX'],
    properties_per_set_max=CONFIG['PROPERTIES_PER_SET_MAX'],
    concept_scope: str = 'all'
):
    logging.info(f'Processing: {file_path}')
    onto = load_ontology_with_fallback(str(file_path))
    if not onto:
        raise RuntimeError(f'Load failed: {file_path}')
    all_props = [p for p in onto.properties() if is_valid_property(p, onto)]
    # concept-scope on properties
    if concept_scope != 'all':
        def is_native(ent):
            return getattr(getattr(ent, 'namespace', None), 'ontology', None) is onto
        def is_imported(ent):
            o = getattr(getattr(ent, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            all_props = [p for p in all_props if is_native(p)]
        else:
            all_props = [p for p in all_props if is_imported(p)]
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

    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [p for p in rel_parts[:-1]]
    safe_stem = Path(rel_parts[-1]).stem if rel_parts else file_path.stem
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'constraint_{safe_stem}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    logging.info(f'Saved {len(tasks)} tasks to {out_path}')

# ---------- 加载与转换 ----------
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


def main():
    parser = argparse.ArgumentParser(description='Generate property constraint tasks (domain, range, functional).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory.')
    parser.add_argument('--output', type=str, required=True, help='Output root (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of properties.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')
    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s')
    random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = CONFIG['EXTENSIONS']
    files = []
    if input_path.is_file() and input_path.suffix.lower() in exts:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, fs in os.walk(str(input_path)):
            for fn in fs:
                if fn.lower().endswith(exts):
                    files.append(Path(root)/fn)
    for fp in files:
        try:
            process_for_constraint_task(fp, input_root, output_root, concept_scope=args.concept_scope)
        except Exception as e:
            logging.error(f'Error: {e}')

if __name__ == '__main__':
    main()
