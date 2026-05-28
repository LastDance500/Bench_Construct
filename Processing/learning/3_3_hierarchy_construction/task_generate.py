import json
import csv
import os
import random
import logging
import argparse
import warnings
import sys
import re
from pathlib import Path
from collections import deque
import owlready2
from owlready2 import *
from owlready2 import set_log_level, onto_path


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    configure_world_paths,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    get_comment,
    get_definition,
    get_label,
    load_ontology,
    save_empty_marker,
    save_json,
    suppress_library_noise,
)

# ---------- configuration ----------
CONFIG = {
    'EXTENSIONS': ('.owl', '.rdf', '.rdfs', '.ttl', '.xml', '.n3'),
    'MAX_SUBGRAPH_SIZE': 15,  # 最大子图大小（上限）
    'MIN_SUBGRAPH_SIZE': 8,   # 最小子图大小（下限）
    'DEPTH_OPTIONS': [2, 3, 4, 5, 6, 7, 8],
    'MAX_SUBGRAPH_RETRIES': 20,
    'NUM_CLASS_SETS_MAX': 100,
    'CLASSES_PER_SET_MAX': 10,  # 最大类集大小（上限）
    'MIN_CLASSES_PER_SET': 5    # 最小类集大小（下限）
}


# ---------- 兼容性补丁 ----------
if not hasattr(owlready2.World, '_get_obj_triples'):
    def _stub_get_obj_triples(self, *args, **kwargs):
        return []
    owlready2.World._get_obj_triples = _stub_get_obj_triples
if not hasattr(owlready2.World, '_get_obj_triples_cspo_cspo'):
    owlready2.World._get_obj_triples_cspo_cspo = owlready2.World._get_obj_triples

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def render_data_range(value):
    if value is str:
        return "string"
    if value is int:
        return "integer"
    if value is float:
        return "float"
    if value is bool:
        return "boolean"
    return get_label(value)


def normalize_label_text(text: str) -> str:
    label = str(text or "").split(":", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", label.replace("_", " ").replace("-", " ").lower()).strip()


def is_placeholder_label(text: str) -> bool:
    normalized = normalize_label_text(text)
    return not normalized or normalized in {"no scale name found", "unnamed", "none", "null"}


def valid_hierarchy_task(data: dict) -> bool:
    triples = data.get("triples") or []
    if not triples:
        return False
    labels = [normalize_label_text(label) for label in data.get("classes", [])]
    if any(is_placeholder_label(label) for label in labels):
        return False
    return len(labels) == len(set(labels))


def get_clean_property_sets(onto):
    object_property_names = {prop.name for prop in onto.object_properties()}
    data_property_names = {prop.name for prop in onto.data_properties()}
    mixed = object_property_names & data_property_names
    obj_props = [prop for prop in onto.object_properties() if prop.name not in mixed]
    data_props = [prop for prop in onto.data_properties() if prop.name not in mixed]
    return obj_props, data_props

def select_related_classes(all_classes, classes_per_set):
    if not all_classes:
        return []
    max_c = min(classes_per_set, len(all_classes))
    min_c = min(CONFIG['MIN_CLASSES_PER_SET'], max_c)
    # 随机起点
    start = random.choice(all_classes)
    related = {start}
    queue = deque([start])
    # BFS 扩展
    while queue and len(related) < max_c:
        cls = queue.popleft()
        for sup in cls.is_a:
            if isinstance(sup, ThingClass) and sup != Thing and sup not in related:
                related.add(sup); queue.append(sup)
        for sub in cls.subclasses():
            if sub not in related:
                related.add(sub); queue.append(sub)
    # 如果不足下限，随机补足
    if len(related) < min_c:
        remaining = [c for c in all_classes if c not in related]
        if remaining:
            related.update(random.sample(remaining, min(min_c - len(related), len(remaining))))
    return random.sample(list(related), min(max_c, len(related)))

# ---------- subgraph extraction ----------
def get_subgraph_around_classes(onto, input_classes, depth=2):
    obj_props, data_props = get_clean_property_sets(onto)
    ann_props  = list(onto.annotation_properties())
    prop_domains = {p: list(p.domain) for p in obj_props + data_props}
    prop_ranges  = {p: list(p.range)  for p in obj_props + data_props}

    for attempt in range(CONFIG['MAX_SUBGRAPH_RETRIES'] + 1):
        related, rels, data_triples, annotations = set(), set(), set(), {}
        visited = set()
        queue = deque([(c, 0) for c in input_classes if isinstance(c, ThingClass)])
        target_depth = depth
        while queue and len(related) < CONFIG['MAX_SUBGRAPH_SIZE']:
            current, d = queue.popleft()
            if current in visited or not isinstance(current, ThingClass):
                continue
            visited.add(current)
            related.add(current)
            # 注释
            com = get_comment(current)
            if com:
                annotations[current] = com
            # 深度扩展
            if d < target_depth:
                for sup in current.is_a:
                    if isinstance(sup, ThingClass) and sup != Thing:
                        queue.append((sup, d+1))
                for sub in current.subclasses():
                    queue.append((sub, d+1))
            # 对象属性
            for p in obj_props:
                if current in prop_domains.get(p, []):
                    for rng in prop_ranges.get(p, []):
                        if isinstance(rng, ThingClass):
                            rels.add((current, p, rng))
                            if d+1 <= target_depth:
                                queue.append((rng, d+1))
            # 数据属性
            for p in data_props:
                if current in prop_domains.get(p, []):
                    for rng in prop_ranges.get(p, []):
                        data_triples.add((current, p, rng))
            # 注释属性
            for p in ann_props:
                try:
                    vals = p[current]
                    if vals:
                        annotations[current] = vals[0]
                except:
                    pass
        # 检查下限
        if (all(c in related for c in input_classes) and
            len(related) >= CONFIG['MIN_SUBGRAPH_SIZE']):
            return related, rels, data_triples, annotations
        target_depth += 1
        logging.debug("Subgraph too small at depth %s; retrying depth %s", target_depth - 1, target_depth)
    logging.info("No usable subgraph after %s retries", CONFIG['MAX_SUBGRAPH_RETRIES'])
    return set(), set(), set(), {}

# ---------- hierarchy and relations ----------
def generate_hierarchy_triples(onto, input_classes):
    classes, rels, data_triples, annotations = get_subgraph_around_classes(onto, input_classes)
    logging.info(f"Subgraph classes: {[get_label(c) for c in classes]}")
    isolated = set(classes)
    triples = set()
    # 子类/超类
    for c in classes:
        for sup in c.is_a:
            if isinstance(sup, ThingClass) and sup != Thing and sup in classes:
                triples.add((c, 'subClassOf', sup)); isolated.discard(c); isolated.discard(sup)
        for sub in c.subclasses():
            if sub in classes:
                triples.add((sub, 'subClassOf', c)); isolated.discard(sub); isolated.discard(c)
    # prepare output
    classes_with_defs = []
    for c in sorted(classes, key=get_label):
        d = get_definition(c)
        classes_with_defs.append(f"{get_label(c)}: {d}" if d != 'No definition provided.' else get_label(c))
    triple_texts = []
    for s, pr, o in sorted(triples, key=lambda x: (get_label(x[0]), x[1], render_data_range(x[2]))):
        subj = get_label(s)
        obj  = get_label(o) if isinstance(o, ThingClass) else render_data_range(o)
        triple_texts.append({'triple': (subj, pr, obj), 'text': f"{subj} {pr} {obj}."})
    return {'classes': classes_with_defs,
            'triples': triple_texts,
            'annotations': {get_label(c): annotations[c] for c in annotations},
            'isolated_classes': [get_label(c) for c in isolated]}

# ---------- task description ----------
def describe_hierarchy_task(classes, annotations, isolated_classes=None):
    lines = [
        "Given the following set of classes, construct the class hierarchy.",
    ]
    for c in sorted(classes):
        lines.append(f"- **{c}**")
    if isolated_classes:
        lines.append("\n### Note")
        lines.append(f"No subclass/superclass for: {', '.join(isolated_classes)}")
    lines.append("\n### Task")
    lines.append("Generate only subClassOf triples. Do not include object-property or data-property relationships.")
    return "\n".join(lines)

# ---------- main flow ----------
def process_for_hierarchy_task(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    num_class_sets_max=CONFIG['NUM_CLASS_SETS_MAX'],
    classes_per_set_max=CONFIG['CLASSES_PER_SET_MAX'],
    concept_scope: str = 'all',
    load_imports: bool = True,
    onto_paths: list | None = None,
    suppress_noise: bool = False
):
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root, sanitize_parts=False)
    out_path = out_dir / f"hierarchy_{safe_stem}.json"
    csv_path = out_dir / f"hierarchy_{safe_stem}.csv"
    empty_path = empty_marker_path(out_dir, "hierarchy", safe_stem)
    if out_path.exists() and csv_path.exists():
        logging.info("Skip existing: %s", out_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    onto = load_task_ontology(file_path, load_imports=load_imports, onto_paths=onto_paths)
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    if load_imports:
        for imp in onto.imported_ontologies:
            try:
                imp.load()
            except Exception:
                pass
    tasks, all_classes = [], [c for c in onto.classes() if isinstance(c, ThingClass) and c != Thing]
    # concept scope filter on classes
    if concept_scope != 'all':
        def is_native(c):
            return getattr(getattr(c, 'namespace', None), 'ontology', None) is onto
        def is_imported(c):
            o = getattr(getattr(c, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            all_classes = [c for c in all_classes if is_native(c)]
        else:
            all_classes = [c for c in all_classes if is_imported(c)]
    total = len(all_classes)
    if total < CONFIG['MIN_CLASSES_PER_SET']:
        logging.info("Not enough classes (%s) to generate tasks", total)
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="not_enough_classes_for_hierarchy_task",
            extra={"classes": total},
        )
        return
    per = min(classes_per_set_max, total)
    sets = min(num_class_sets_max, total // max(1, CONFIG['MIN_CLASSES_PER_SET']))
    logging.info(f"Generating {sets} tasks with up to {per} classes each")
    for _ in range(sets):
        sel = select_related_classes(all_classes, per)
        if len(sel) < CONFIG['MIN_CLASSES_PER_SET']:
            continue
        data = generate_hierarchy_triples(onto, sel)
        if not valid_hierarchy_task(data):
            continue
        desc = describe_hierarchy_task(data['classes'], data['annotations'], data['isolated_classes'])
        tasks.append({'task_description': desc, 'classes': data['classes'], 'triples': data['triples']})
    if not tasks:
        logging.info("No tasks generated; skip save")
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_hierarchy_tasks",
            extra={"classes": total},
        )
        return
    save_json(tasks, out_path, description="tasks")
    try:
        domain = "/".join(file_path.relative_to(input_root).parts[:-1]) or file_path.parent.name
    except Exception:
        domain = file_path.parent.name
    write_task_csv(tasks, csv_path, task_label="3_3", domain=domain)


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

# ---------- OWL 加载 ----------
def load_task_ontology(file_path: Path, load_imports: bool = True, onto_paths: list | None = None):
    world = owlready2.World()
    configure_world_paths(world, onto_paths)
    return load_ontology(world, file_path, load_imports=load_imports)

def main():
    parser = argparse.ArgumentParser(description='Generate class hierarchy construction tasks.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory.')
    parser.add_argument('--output', type=str, required=True, help='Output root (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of classes.')
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
    failed = []
    for fp in files:
        try:
            process_for_hierarchy_task(
                fp,
                input_root,
                output_root,
                concept_scope=args.concept_scope,
                load_imports=not args.no_imports,
                onto_paths=args.onto_path,
                suppress_noise=args.no_warnings,
            )
        except Exception as e:
            logging.error(f"Failed {fp}: {e}")
            failed.append(str(fp))
    if failed:
        logging.info(f"Failed files {len(failed)}: {failed}")

if __name__ == '__main__':
    main()
