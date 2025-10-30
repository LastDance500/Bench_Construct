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
        # IAO_0000115 优先 English
        defs = getattr(entity, "IAO_0000115", None)
        if defs:
            definition = next((d for d in defs if getattr(d, 'lang', None) == 'en'), defs[0])
        # skos:definition
        if not definition:
            skos_defs = getattr(entity, 'definition', None)
            if skos_defs:
                definition = next((d for d in skos_defs if getattr(d, 'lang', None) == 'en'), skos_defs[0])
        # rdfs:comment
        if not definition:
            comment = getattr(entity, 'comment', None)
            if comment:
                if isinstance(comment, list):
                    definition = next((d for d in comment if getattr(d, 'lang', None) == 'en'), comment[0])
                else:
                    definition = comment
        definition = str(definition).strip() if definition and str(definition).strip() else 'No definition provided.'
        definition_cache[key] = definition
        return definition
    except Exception as e:
        logging.warning(f"Error retrieving definition for {get_label(entity)}: {e}")
        return 'No definition provided.'

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
    obj_props = list(onto.object_properties())
    data_props = list(onto.data_properties())
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
        logging.warning(f"Subgraph too small at depth {target_depth-1}, retry depth {target_depth}")
    logging.error(f"Failed to create subgraph after {CONFIG['MAX_SUBGRAPH_RETRIES']} retries")
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
    # 对象属性
    for s, p, o in rels:
        if s in classes and o in classes:
            triples.add((s, get_label(p), o))
    # 数据属性：输出数据属性到其范围（通常为数据类型），与任务描述一致
    for s, p, o in data_triples:
        if s in classes:
            triples.add((s, get_label(p), o))
    # prepare output
    classes_with_defs = []
    for c in sorted(classes, key=get_label):
        d = get_definition(c)
        classes_with_defs.append(f"{get_label(c)}: {d}" if d != 'No definition provided.' else get_label(c))
    triple_texts = []
    for s, pr, o in sorted(triples, key=lambda x: (get_label(x[0]), x[1], str(x[2]))):
        subj = get_label(s)
        obj  = get_label(o) if isinstance(o, ThingClass) else str(o)
        triple_texts.append({'triple': (subj, pr, obj), 'text': f"{subj} {pr} {obj}."})
    return {'classes': classes_with_defs,
            'triples': triple_texts,
            'annotations': {get_label(c): annotations[c] for c in annotations},
            'isolated_classes': [get_label(c) for c in isolated]}

# ---------- task description ----------
def describe_hierarchy_task(classes, annotations, isolated_classes=None):
    lines = [
        "Given the following set of classes, construct both hierarchical and other property relationships.",
    ]
    for c in sorted(classes):
        lines.append(f"- **{c}**")
    if isolated_classes:
        lines.append("\n### Note")
        lines.append(f"No subclass/superclass for: {', '.join(isolated_classes)}")
    lines.append("\n### Task")
    lines.append("Generate triples:\n- subClassOf relationships\n- object-property relationships\n- data-property relationships")
    return "\n".join(lines)

# ---------- main flow ----------
def process_for_hierarchy_task(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    num_class_sets_max=CONFIG['NUM_CLASS_SETS_MAX'],
    classes_per_set_max=CONFIG['CLASSES_PER_SET_MAX'],
    concept_scope: str = 'all'
):
    onto = load_ontology_with_fallback(str(file_path))
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    for imp in onto.imported_ontologies:
        try: imp.load()
        except: pass
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
        logging.warning(f"不足类({total})生成任务")
        return
    per = min(classes_per_set_max, total)
    sets = min(num_class_sets_max, total // max(1, CONFIG['MIN_CLASSES_PER_SET']))
    logging.info(f"Generating {sets} tasks with up to {per} classes each")
    for _ in range(sets):
        sel = select_related_classes(all_classes, per)
        if len(sel) < CONFIG['MIN_CLASSES_PER_SET']:
            continue
        data = generate_hierarchy_triples(onto, sel)
        desc = describe_hierarchy_task(data['classes'], data['annotations'], data['isolated_classes'])
        tasks.append({'task_description': desc, 'classes': data['classes'], 'triples': data['triples']})
    if not tasks:
        logging.warning("No tasks generated; skip save")
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
    out_path = out_dir / f"hierarchy_{safe_stem}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(tasks)} tasks: {out_path}")

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
        logging.warning(f"Owlready2 failed: {e}")
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
    parser = argparse.ArgumentParser(description='Generate hierarchy construction tasks (classes + properties).')
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
            process_for_hierarchy_task(fp, input_root, output_root, concept_scope=args.concept_scope)
        except Exception as e:
            logging.error(f"Failed {fp}: {e}")
            failed.append(str(fp))
    if failed:
        logging.info(f"Failed files {len(failed)}: {failed}")

if __name__ == '__main__':
    main()
