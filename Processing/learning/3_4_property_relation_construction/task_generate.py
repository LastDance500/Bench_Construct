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
from collections import deque, defaultdict

from owlready2 import *


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import build_mirrored_output_dir, configure_logging, configure_world_paths, discover_ontology_files, empty_marker_path, get_comment, get_label, load_ontology, save_empty_marker, save_json, suppress_library_noise

# ---------- configuration ----------
CONFIG = {
    'EXTENSIONS': ('.owl', '.rdf', '.rdfs', '.ttl', '.xml', '.n3'),
    'MAX_SUBGRAPH_SIZE': 10,  # 最大子图大小（上限）
    'MIN_SUBGRAPH_SIZE': 5,   # 最小子图大小（下限）
    'DEPTH_OPTIONS': [2, 3, 4],
    'MAX_SUBGRAPH_RETRIES': 10,
    'NUM_CLASS_SETS_MAX': 100,
    'CLASSES_PER_SET_MAX': 8,   # 最大类集大小（上限）
    'MIN_CLASSES_PER_SET': 4,   # 最小类集大小（下限）
    # 结构精简相关
    'MAX_OBJ_TRIPLES_PER_CLASS': 3,
    'MAX_DATA_TRIPLES_PER_CLASS': 2,
    'MAX_TRIPLES_TOTAL': 24,
    'MAX_CLASSES_LISTED': 8
}

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def render_data_range(value):
    if value is None:
        return ""
    if value is str:
        return "string"
    if value is int:
        return "integer"
    if value is float:
        return "float"
    if value is bool:
        return "boolean"
    return get_label(value)


def is_usable_data_range(value) -> bool:
    label = render_data_range(value)
    normalized = normalize_label_text(label)
    return bool(normalized) and normalized not in {"none", "null", "thing", "object"}


def normalize_label_text(text: str) -> str:
    label = str(text or "").split(":", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", label.replace("_", " ").replace("-", " ").lower()).strip()


def valid_property_task(data: dict) -> bool:
    triples = data.get("triples") or []
    if not triples:
        return False
    labels = [normalize_label_text(label) for label in data.get("classes", [])]
    return bool(labels) and len(labels) == len(set(labels))


def safe_class_properties(cls):
    try:
        return [prop for prop in cls.get_class_properties() if hasattr(prop, "domain")]
    except Exception as exc:
        logging.debug("Skipping class property score for %s: %s", get_label(cls), exc)
        return []


def get_clean_property_sets(onto):
    object_property_names = {prop.name for prop in onto.object_properties()}
    data_property_names = {prop.name for prop in onto.data_properties()}
    mixed = object_property_names & data_property_names
    obj_props = [prop for prop in onto.object_properties() if prop.name not in mixed]
    data_props = [prop for prop in onto.data_properties() if prop.name not in mixed]
    return obj_props, data_props


def select_related_classes(all_classes, classes_per_set):
    """选择一组有层次关系的类，保证达到最小和不超过最大"""
    if not all_classes:
        return []
    max_c = min(classes_per_set, len(all_classes))
    min_c = min(CONFIG['MIN_CLASSES_PER_SET'], max_c)

    def class_score(cls):
        score = len(list(cls.subclasses()))
        score += len([sup for sup in cls.is_a if isinstance(sup, ThingClass) and sup != Thing])
        for prop in safe_class_properties(cls):
            try:
                if cls in getattr(prop, 'domain', []):
                    score += 1
            except Exception:
                continue
        return score

    sorted_classes = sorted(all_classes, key=class_score, reverse=True)
    top_k = max(max_c * 2, 10)
    top_classes = sorted_classes[:min(top_k, len(sorted_classes))]

    start = random.choice(top_classes)
    related = {start}
    queue = deque([start])
    while queue and len(related) < max_c:
        cls = queue.popleft()
        neighbors = [sup for sup in cls.is_a if isinstance(sup, ThingClass) and sup != Thing]
        neighbors += list(cls.subclasses())
        neighbors = sorted(neighbors, key=class_score, reverse=True)
        for nb in neighbors:
            if nb not in related:
                related.add(nb)
                queue.append(nb)
                if len(related) >= max_c:
                    break
    if len(related) < min_c:
        remaining = [c for c in all_classes if c not in related]
        if remaining:
            needed = min(min_c - len(related), len(remaining))
            related.update(random.sample(remaining, needed))
    return random.sample(list(related), min(max_c, len(related)))


# ---------- subgraph extraction ----------
def get_subgraph_around_classes(onto, input_classes, depth=2):
    """提取子图，包含对象和数据属性，且满足最小子图大小"""
    target_depth = min(depth, max(CONFIG['DEPTH_OPTIONS']))

    obj_props, data_props = get_clean_property_sets(onto)
    ann_props  = list(onto.annotation_properties())
    prop_domains = {p: list(p.domain) if p.domain else [] for p in obj_props + data_props}
    prop_ranges  = {p: list(p.range)  if p.range  else [] for p in obj_props + data_props}

    for attempt in range(CONFIG['MAX_SUBGRAPH_RETRIES']):
        visited = set()
        related = set()
        obj_rels = set()
        data_rels = set()
        annotations = {}
        queue = deque([(c, 0) for c in input_classes if isinstance(c, ThingClass)])

        for c in input_classes:
            if isinstance(c, ThingClass):
                sups = [s for s in c.is_a if isinstance(s, ThingClass) and s != Thing]
                subs = list(c.subclasses())
                if sups:
                    queue.append((sups[0], 0))
                elif subs:
                    queue.append((subs[0], 0))

        while queue and len(related) < CONFIG['MAX_SUBGRAPH_SIZE']:
            current, d = queue.popleft()
            if current in visited or not isinstance(current, ThingClass):
                continue
            visited.add(current)
            related.add(current)

            com = get_comment(current)
            if com:
                annotations[current] = com

            if d < target_depth:
                for sup in current.is_a:
                    if isinstance(sup, ThingClass) and sup != Thing:
                        queue.append((sup, d+1))
                for sub in current.subclasses():
                    queue.append((sub, d+1))

            for p in obj_props:
                if current in prop_domains.get(p, []):
                    for rng in prop_ranges.get(p, []):
                        if isinstance(rng, ThingClass):
                            obj_rels.add((current, p, rng))
                            if d+1 <= target_depth:
                                queue.append((rng, d+1))

            for p in data_props:
                if current in prop_domains.get(p, []):
                    for rng in prop_ranges.get(p, []):
                        if is_usable_data_range(rng):
                            data_rels.add((current, p, rng))

            for p in ann_props:
                try:
                    vals = p[current]
                    if vals and current not in annotations:
                        annotations[current] = vals[0]
                except:
                    pass

        if (all(c in related for c in input_classes)
                and len(related) >= CONFIG['MIN_SUBGRAPH_SIZE']
                and (obj_rels or data_rels or any(c.is_a or c.subclasses() for c in related))):
            logging.info(f"Subgraph: classes={len(related)}, obj_rels={len(obj_rels)}, data_rels={len(data_rels)}")
            return related, obj_rels, data_rels, annotations, obj_props, data_props

        prev = target_depth
        idx = CONFIG['DEPTH_OPTIONS'].index(prev)
        if idx < len(CONFIG['DEPTH_OPTIONS'])-1:
            target_depth = CONFIG['DEPTH_OPTIONS'][idx+1]
        logging.debug("Depth %s too small; retrying with %s", prev, target_depth)

    logging.info("No usable property subgraph after %s retries", CONFIG['MAX_SUBGRAPH_RETRIES'])
    return set(), set(), set(), {}, [], []


# ---------- property-only triple generation ----------
def generate_property_triples(onto, input_classes):
    """
    仅生成属性（对象/数据）三元组的任务，不包含 subClassOf。
    triples 内部以属性实体存储，渲染时输出标签，同时在元信息中附带 IRI，确保唯一性。
    """
    for attempt in range(CONFIG['MAX_SUBGRAPH_RETRIES'] + 1):
        classes, obj_rels, data_rels, annotations, obj_props, data_props = \
            get_subgraph_around_classes(onto, input_classes)
        if not classes:
            logging.debug("No valid subgraph at attempt %s; retrying", attempt + 1)
            continue

        triples = set()  # (subject_class, predicate_entity_or_keyword, object_class_or_datatype)
        prop_chars = {}  # {property_entity: [characteristics]}
        obj_count_by_subject = defaultdict(int)
        data_count_by_subject = defaultdict(int)

        # do not include subClassOf, only object/data properties

        # 对象属性三元组
        for s, p, o in obj_rels:
            if s in classes and o in classes:
                # 每个类的对象属性三元组数量上限
                if obj_count_by_subject[s] >= CONFIG['MAX_OBJ_TRIPLES_PER_CLASS']:
                    continue
                triples.add((s, p, o))
                obj_count_by_subject[s] += 1
                chars = []
                if isinstance(p, FunctionalProperty): chars.append('functional')
                if isinstance(p, SymmetricProperty):  chars.append('symmetric')
                if isinstance(p, TransitiveProperty): chars.append('transitive')
                if chars:
                    prop_chars.setdefault(p, []).extend(chars)

        # 数据属性三元组
        for s, p, lit in data_rels:
            if s in classes and is_usable_data_range(lit):
                # 每个类的数据属性三元组数量上限
                if data_count_by_subject[s] >= CONFIG['MAX_DATA_TRIPLES_PER_CLASS']:
                    continue
                triples.add((s, p, lit))
                data_count_by_subject[s] += 1
                if isinstance(p, FunctionalProperty):
                    prop_chars.setdefault(p, []).append('functional')

        if not triples:
            logging.debug("No property triples found at attempt %s; retrying", attempt + 1)
            continue

        # 按稳定排序裁剪总三元组数量，避免任务过长
        if len(triples) > CONFIG['MAX_TRIPLES_TOTAL']:
            def _sort_key(t):
                s, pr, o = t
                pr_label = pr if isinstance(pr, str) else get_label(pr)
                return (get_label(s), pr_label, get_label(o) if isinstance(o, ThingClass) else render_data_range(o))
            triples = set(list(sorted(triples, key=_sort_key))[:CONFIG['MAX_TRIPLES_TOTAL']])

        # 计算实际使用到的属性
        used_obj_props  = {get_label(pr) for s, pr, o in triples if isinstance(o, ThingClass)}
        used_data_props = {get_label(pr) for s, pr, o in triples if not isinstance(o, ThingClass)}

        triple_texts = []
        def triple_sort_key(t):
            s, pr, o = t
            pr_label = pr if isinstance(pr, str) else get_label(pr)
            return (get_label(s), pr_label, get_label(o) if isinstance(o, ThingClass) else render_data_range(o))

        for s, pr, o in sorted(triples, key=triple_sort_key):
            subj_label = get_label(s)
            pred_label = pr if isinstance(pr, str) else get_label(pr)
            obj_label  = get_label(o) if isinstance(o, ThingClass) else render_data_range(o)
            triple_texts.append({
                'triple': (subj_label, pred_label, obj_label),
                'text': f"{subj_label} {pred_label} {obj_label}.",
                'characteristics': prop_chars.get(pr, []),
                'meta': {
                    'subject_iri': str(s.iri) if hasattr(s, 'iri') else None,
                    'predicate_iri': (None if isinstance(pr, str) else str(pr.iri) if hasattr(pr, 'iri') else None),
                    'object_iri': (str(o.iri) if isinstance(o, ThingClass) and hasattr(o, 'iri') else None),
                    'predicate_type': ('object' if isinstance(o, ThingClass) else 'data')
                }
            })

        logging.info(f"Generated valid task with {len(classes)} classes and {len(triple_texts)} property triples")
        return {
            'classes': sorted(get_label(c) for c in classes),
            'triples': triple_texts,
            'annotations': {get_label(c): annotations[c] for c in annotations},
            'isolated_classes': [],
            'object_properties': sorted(used_obj_props),
            'data_properties':   sorted(used_data_props)
        }

    logging.info("No property-only task generated after max retries")
    return {
        'classes': [], 'triples': [], 'annotations': {},
        'isolated_classes': [], 'object_properties': [], 'data_properties': []
    }


# ---------- task description ----------
def describe_property_task(classes, annotations, isolated_classes, object_props, data_props):
    lines = [
        "## Property-only Construction Task",
        "Given the following set of classes, construct only object- and data-property relationships (no subClassOf).",
        "### Classes"
    ]
    if not classes:
        lines.append("- No classes available.")
    else:
        max_show = CONFIG['MAX_CLASSES_LISTED']
        shown = classes[:max_show]
        for c in shown:
            line = f"- **{c}**"
            if c in annotations:
                line += f": {annotations[c]}"
            lines.append(line)
        remaining = len(classes) - len(shown)
        if remaining > 0:
            lines.append(f"- ... and {remaining} more")
    lines.append("\n### Object Properties")
    lines += ([f"- {p}" for p in object_props] or ["- None"])
    lines.append("\n### Data Properties")
    lines += ([f"- {p}" for p in data_props] or ["- None"])
    if isolated_classes:
        lines.append("\n### Note")
        lines.append(f"Standalone classes: {', '.join(isolated_classes)}")
    lines.append("\n### Task")
    lines.append("Generate triples for object- and data-properties only. Do not include subClassOf or property characteristics.")
    return "\n".join(lines)


# ---------- main flow ----------
def process_for_property_task(
        file_path: Path,
        input_root: Path,
        output_root: Path,
        num_class_sets_max=CONFIG['NUM_CLASS_SETS_MAX'],
        classes_per_set_max=CONFIG['CLASSES_PER_SET_MAX'],
        concept_scope: str = 'all',
        load_imports: bool = True,
        onto_paths: list | None = None,
):
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root, sanitize_parts=False)
    out_path = out_dir / f"property_{safe_stem}.json"
    csv_path = out_dir / f"property_{safe_stem}.csv"
    empty_path = empty_marker_path(out_dir, "property", safe_stem)
    if out_path.exists() and csv_path.exists():
        logging.info("Skip existing: %s", out_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    onto = load_task_ontology(file_path, load_imports=load_imports, onto_paths=onto_paths)
    if not onto:
        logging.error(f"Failed to load ontology: {file_path}")
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="ontology_load_failed_for_property_task",
        )
        return
    if load_imports:
        for imp in onto.imported_ontologies:
            try:
                imp.load()
            except Exception:
                pass

    all_classes = [c for c in onto.classes() if isinstance(c, ThingClass) and c != Thing]
    # concept scope on classes
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
            reason="not_enough_classes_for_property_task",
            extra={"classes": total},
        )
        return

    per = min(classes_per_set_max, total)
    num_sets = min(num_class_sets_max, max(1, total // max(1, CONFIG['MIN_CLASSES_PER_SET'])))
    logging.info(f"Generating {num_sets} tasks with up to {per} classes each for {file_path}")

    tasks = []
    for _ in range(num_sets):
        sel = select_related_classes(all_classes, per)
        if len(sel) < CONFIG['MIN_CLASSES_PER_SET']:
            continue
        data = generate_property_triples(onto, sel)
        if not data['classes'] or not valid_property_task(data):
            continue
        desc = describe_property_task(
            data['classes'], data['annotations'], data['isolated_classes'],
            data['object_properties'], data['data_properties']
        )
        tasks.append({
            'task_description': desc,
            'classes': data['classes'],
            'object_properties': data['object_properties'],
            'data_properties': data['data_properties'],
            'triples': data['triples']
        })

    if not tasks:
        logging.info("No valid tasks generated; skipping save")
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_property_relation_tasks",
            extra={"classes": total},
        )
        return

    save_json(tasks, out_path, description="property tasks")
    try:
        domain = "/".join(file_path.relative_to(input_root).parts[:-1]) or file_path.parent.name
    except Exception:
        domain = file_path.parent.name
    write_task_csv(tasks, csv_path, task_label="3_4", domain=domain)


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


def save_failure_marker(file_path: Path, input_root: Path, output_root: Path, reason: str, error: Exception | str | None = None) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root, sanitize_parts=False)
    empty_path = empty_marker_path(out_dir, "property", safe_stem)
    save_empty_marker(
        empty_path,
        source_file=file_path,
        reason=reason,
        extra={"error": str(error) if error is not None else ""},
    )


# ---------- OWL 加载 ----------
def load_task_ontology(file_path: Path, load_imports: bool = True, onto_paths: list | None = None):
    world = World()
    configure_world_paths(world, onto_paths)
    return load_ontology(world, file_path, load_imports=load_imports)


def main():
    parser = argparse.ArgumentParser(description='Generate property-only construction tasks (no subclassOf).')
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
            process_for_property_task(
                fp,
                input_root,
                output_root,
                concept_scope=args.concept_scope,
                load_imports=not args.no_imports,
                onto_paths=args.onto_path,
            )
        except Exception as e:
            logging.error(f"Failed {fp}: {e}")
            save_failure_marker(
                fp,
                input_root,
                output_root,
                reason="property_relation_generation_failed",
                error=e,
            )
            failed.append(str(fp))
    if failed:
        logging.info(f"Failed files: {failed}")

if __name__ == '__main__':
    main()
