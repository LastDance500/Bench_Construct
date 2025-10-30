import json
import os
import random
import logging
import uuid
import argparse
import warnings
from pathlib import Path
from collections import deque, defaultdict

import rdflib
from owlready2 import *

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

# ---------- helpers ----------
def get_label(entity):
    """获取实体或属性的标签或名称"""
    if not entity:
        return "Unnamed"
    labels = getattr(entity, 'label', []) or []
    if labels:
        return labels[0]
    name = getattr(entity, 'name', None)
    if name:
        return name
    return str(entity).split('.')[-1]


def get_comment(entity):
    """获取实体的注释"""
    if not entity:
        return None
    comments = getattr(entity, 'comment', []) or []
    return comments[0] if comments else None


def select_related_classes(all_classes, classes_per_set):
    """选择一组有层次关系的类，保证达到最小和不超过最大"""
    if not all_classes:
        return []
    max_c = min(classes_per_set, len(all_classes))
    min_c = min(CONFIG['MIN_CLASSES_PER_SET'], max_c)

    def class_score(cls):
        score = len(list(cls.subclasses()))
        score += len([sup for sup in cls.is_a if isinstance(sup, ThingClass) and sup != Thing])
        score += len([p for p in cls.get_class_properties() if hasattr(p, 'domain') and cls in getattr(p, 'domain', [])])
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

    obj_props = list(onto.object_properties())
    data_props = list(onto.data_properties())
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
        logging.warning(f"Depth {prev} too small, retry with {target_depth}")

    logging.error(f"Failed after {CONFIG['MAX_SUBGRAPH_RETRIES']} retries")
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
            logging.warning(f"No valid subgraph at attempt {attempt + 1}; retrying")
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
            if s in classes:
                # 每个类的数据属性三元组数量上限
                if data_count_by_subject[s] >= CONFIG['MAX_DATA_TRIPLES_PER_CLASS']:
                    continue
                triples.add((s, p, lit))
                data_count_by_subject[s] += 1
                if isinstance(p, FunctionalProperty):
                    prop_chars.setdefault(p, []).append('functional')

        if not triples:
            logging.warning(f"No property triples found at attempt {attempt + 1}; retrying")
            continue

        # 按稳定排序裁剪总三元组数量，避免任务过长
        if len(triples) > CONFIG['MAX_TRIPLES_TOTAL']:
            def _sort_key(t):
                s, pr, o = t
                pr_label = pr if isinstance(pr, str) else get_label(pr)
                return (get_label(s), pr_label, get_label(o) if isinstance(o, ThingClass) else str(o))
            triples = set(list(sorted(triples, key=_sort_key))[:CONFIG['MAX_TRIPLES_TOTAL']])

        # 计算实际使用到的属性
        used_obj_props  = {get_label(pr) for s, pr, o in triples if isinstance(o, ThingClass)}
        used_data_props = {get_label(pr) for s, pr, o in triples if not isinstance(o, ThingClass)}

        triple_texts = []
        def triple_sort_key(t):
            s, pr, o = t
            pr_label = pr if isinstance(pr, str) else get_label(pr)
            return (get_label(s), pr_label, get_label(o) if isinstance(o, ThingClass) else str(o))

        for s, pr, o in sorted(triples, key=triple_sort_key):
            subj_label = get_label(s)
            pred_label = pr if isinstance(pr, str) else get_label(pr)
            obj_label  = get_label(o) if isinstance(o, ThingClass) else str(o)
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

    logging.error("Failed to generate property-only task after max retries")
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
    lines.append("Generate triples for object- and data-properties only, including characteristics. Do not include subClassOf.")
    return "\n".join(lines)


# ---------- main flow ----------
def process_for_property_task(
        file_path: Path,
        input_root: Path,
        output_root: Path,
        num_class_sets_max=CONFIG['NUM_CLASS_SETS_MAX'],
        classes_per_set_max=CONFIG['CLASSES_PER_SET_MAX'],
        concept_scope: str = 'all'
):
    onto = load_ontology_with_fallback(str(file_path))
    if not onto:
        logging.error(f"Failed to load ontology: {file_path}")
        return
    for imp in onto.imported_ontologies:
        try: imp.load()
        except: pass

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
        logging.warning(f"Not enough classes ({total}) to generate tasks")
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
        if not data['classes']:
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
        logging.warning("No valid tasks generated; skipping save")
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
    out_path = out_dir / f"property_{safe_stem}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(tasks)} property tasks for {file_path}")


# ---------- OWL 加载 ----------
def rdflib_to_owlready(rdf_graph):
    temp = f"temp_{uuid.uuid4()}.owl"
    rdf_graph.serialize(temp, format='xml')
    onto = get_ontology(f"file://{os.path.abspath(temp)}").load()
    os.remove(temp)
    return onto


def load_ontology_with_fallback(file_path):
    try:
        return get_ontology(f"file://{os.path.abspath(file_path)}").load()
    except Exception as e:
        logging.warning(f"Owlready2 load failed: {e}")
        g = rdflib.Graph()
        for fmt in ['xml', 'turtle', 'n3', 'trig']:
            try:
                g.parse(file_path, format=fmt)
                return rdflib_to_owlready(g)
            except:
                continue
        logging.error(f"All formats failed: {file_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate property-only construction tasks (no subclassOf).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory.')
    parser.add_argument('--output', type=str, required=True, help='Output root (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of classes.')
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
    failed = []
    for fp in files:
        try:
            process_for_property_task(fp, input_root, output_root, concept_scope=args.concept_scope)
        except Exception as e:
            logging.error(f"Failed {fp}: {e}")
            failed.append(str(fp))
    if failed:
        logging.info(f"Failed files: {failed}")

if __name__ == '__main__':
    main()
