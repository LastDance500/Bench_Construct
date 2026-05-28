import json
import csv
import os
import random
import logging
import argparse
import warnings
import sys
from pathlib import Path
from collections import deque

import owlready2
from owlready2 import *


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import build_mirrored_output_dir, configure_logging, configure_world_paths, discover_ontology_files, empty_marker_path, get_comment, get_definition, get_label, load_ontology, save_empty_marker, save_json, suppress_library_noise

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


def render_constraint_value(value):
    if value is str:
        return "string"
    if value is int:
        return "integer"
    if value is float:
        return "float"
    if value is bool:
        return "boolean"
    if value is True:
        return "True"
    return get_label(value)

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

def is_valid_property(prop, onto):
    is_object = isinstance(prop, owlready2.ObjectPropertyClass)
    is_data = isinstance(prop, owlready2.DataPropertyClass)
    if not ((is_object ^ is_data) and prop in onto.properties()):
        return False
    object_property_names = {candidate.name for candidate in onto.object_properties()}
    data_property_names = {candidate.name for candidate in onto.data_properties()}
    return prop.name not in (object_property_names & data_property_names)

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
                if range_ in related_classes or not isinstance(range_, ThingClass):
                    constraints.append((prop, 'range', range_))
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
        key=lambda x: (get_label(x[0]), x[1], render_constraint_value(x[2]))
    ):
        p_lbl = get_label(prop)
        v_lbl = 'True' if ctype in {'functional', 'symmetric', 'transitive'} else render_constraint_value(val)
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
    lines.append('- Generate only domain and range constraints.')
    lines.append('- Do not generate property characteristics such as functional, symmetric, or transitive.')
    return '\n'.join(lines)

def process_for_constraint_task(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    num_property_sets_max=CONFIG['NUM_PROPERTY_SETS_MAX'],
    properties_per_set_max=CONFIG['PROPERTIES_PER_SET_MAX'],
    concept_scope: str = 'all',
    load_imports: bool = True,
    onto_paths: list | None = None,
):
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root, sanitize_parts=False)
    out_path = out_dir / f'constraint_{safe_stem}.json'
    csv_path = out_dir / f'constraint_{safe_stem}.csv'
    empty_path = empty_marker_path(out_dir, "constraint", safe_stem)
    if out_path.exists() and csv_path.exists():
        logging.info("Skip existing: %s", out_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    logging.info(f'Processing: {file_path}')
    onto = load_task_ontology(file_path, load_imports=load_imports, onto_paths=onto_paths)
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
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="not_enough_properties_for_constraint_task",
            extra={"properties": len(all_props)},
        )
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
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_constraint_tasks",
            extra={"properties": len(all_props)},
        )
        return

    save_json(tasks, out_path, description="constraint tasks")
    try:
        domain = "/".join(file_path.relative_to(input_root).parts[:-1]) or file_path.parent.name
    except Exception:
        domain = file_path.parent.name
    write_task_csv(tasks, csv_path, task_label="3_5", domain=domain)


def format_triples(triples: list[dict]) -> str:
    return "\n".join(str(tuple(item.get("triple", ()))) for item in triples if item.get("triple"))


def write_task_csv(tasks: list[dict], csv_path: Path, task_label: str, domain: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "definition", "task_label", "iri", "class_label", "domain"])
        writer.writeheader()
        for task in tasks:
            classes = task.get("classes", []) or []
            writer.writerow(
                {
                    "question": task.get("task_description", ""),
                    "definition": format_triples(task.get("triples", []) or []),
                    "task_label": task_label,
                    "iri": "",
                    "class_label": classes[0] if classes else "",
                    "domain": domain,
                }
            )

# ---------- 加载与转换 ----------
def load_task_ontology(file_path: Path, load_imports: bool = True, onto_paths: list | None = None):
    world = owlready2.World()
    configure_world_paths(world, onto_paths)
    return load_ontology(world, file_path, load_imports=load_imports)


def main():
    parser = argparse.ArgumentParser(description='Generate property constraint tasks (domain and range only).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory.')
    parser.add_argument('--output', type=str, required=True, help='Output root (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of properties.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress owlready2/rdflib warnings output.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')
    args = parser.parse_args()

    configure_logging(args.log)
    suppress_library_noise(args.no_warnings)
    random.seed(args.seed)

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = CONFIG['EXTENSIONS']
    files, input_root = discover_ontology_files(input_path, exts)
    for fp in files:
        try:
            process_for_constraint_task(
                fp,
                input_root,
                output_root,
                concept_scope=args.concept_scope,
                load_imports=not args.no_imports,
                onto_paths=args.onto_path,
            )
        except Exception as e:
            logging.error(f'Error: {e}')

if __name__ == '__main__':
    main()
