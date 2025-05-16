import json
import os
import random
import logging
import uuid
from collections import deque
import rdflib
import owlready2
from owlready2 import *

# ---------- 常量管理 ----------
CONFIG = {
    'BASE_DIR': '../../../data',
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

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- 全局缓存 ----------
definition_cache = {}

# ---------- 工具函数 ----------
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

# ---------- 子图提取 ----------
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
        # 确保包含输入类
        for c in input_classes:
            if isinstance(c, ThingClass):
                queue.append((c, 0))
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
        logging.warning(f"深度 {target_depth-1} 子图过小，重试深度 {target_depth}")
    logging.error(f"{CONFIG['MAX_SUBGRAPH_RETRIES']} 次后仍无法生成子图")
    return set(), set(), set(), {}

# ---------- 层次及关系提取 ----------
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
    # 数据属性（仅保留宾也是类的情况）
    for s, p, o in data_triples:
        if isinstance(o, ThingClass) and s in classes and o in classes:
            triples.add((s, get_label(p), o))
    # 准备输出
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

# ---------- 文本描述 ----------
def describe_hierarchy_task(classes, annotations, isolated_classes=None):
    lines = [
        "## Hierarchy and Relation Construction Task",
        "Given the following set of classes, construct both hierarchical and other property relationships.",
        "### Classes"
    ]
    for c in sorted(classes):
        lines.append(f"- **{c}**")
    if isolated_classes:
        lines.append("\n### Note")
        lines.append(f"No subclass/superclass for: {', '.join(isolated_classes)}")
    lines.append("\n### Task")
    lines.append("Generate triples:\n- subClassOf relationships\n- object-property relationships\n- data-property relationships")
    return "\n".join(lines)

# ---------- 主流程 ----------
def process_for_hierarchy_task(
    file_path,
    num_class_sets_max=CONFIG['NUM_CLASS_SETS_MAX'],
    classes_per_set_max=CONFIG['CLASSES_PER_SET_MAX']
):
    onto = load_ontology_with_fallback(file_path)
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    for imp in onto.imported_ontologies:
        try: imp.load()
        except: pass
    tasks, all_classes = [], [c for c in onto.classes() if isinstance(c, ThingClass) and c != Thing]
    total = len(all_classes)
    if total < CONFIG['MIN_CLASSES_PER_SET']:
        logging.warning(f"不足类({total})生成任务")
        return
    per = min(classes_per_set_max, total)
    sets = min(num_class_sets_max, total // max(1, CONFIG['MIN_CLASSES_PER_SET']))
    logging.info(f"生成 {sets} 个任务，每个最多 {per} 类")
    for _ in range(sets):
        sel = select_related_classes(all_classes, per)
        if len(sel) < CONFIG['MIN_CLASSES_PER_SET']:
            continue
        data = generate_hierarchy_triples(onto, sel)
        desc = describe_hierarchy_task(data['classes'], data['annotations'], data['isolated_classes'])
        tasks.append({'task_description': desc, 'classes': data['classes'], 'triples': data['triples']})
    if not tasks:
        logging.warning("未生成任何任务，跳过保存")
        return
    out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_3_2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hierarchy_{os.path.basename(file_path)}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    logging.info(f"保存 {len(tasks)} 个任务: {out_path}")

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
        logging.warning(f"Owlready2 失败: {e}")
        g = rdflib.Graph()
        for fmt in ['xml', 'turtle', 'n3', 'trig']:
            try:
                g.parse(file_path, format=fmt)
                return rdflib_to_owlready(g)
            except:
                continue
        logging.error(f"所有格式解析失败: {file_path}")
        return None

if __name__ == '__main__':
    random.seed(42)
    files = []
    for root, _, fs in os.walk(CONFIG['BASE_DIR']):
        for fn in fs:
            if fn.lower().endswith(CONFIG['EXTENSIONS']):
                files.append(os.path.join(root, fn))
    failed = []
    for fp in files:
        try:
            process_for_hierarchy_task(fp)
        except Exception as e:
            logging.error(f"失败 {fp}: {e}")
            failed.append(fp)
    if failed:
        logging.info(f"失败文件数 {len(failed)}: {failed}")
