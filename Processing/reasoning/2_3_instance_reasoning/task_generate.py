import json
import os
import random
import logging
from owlready2 import World, ThingClass, owl
from collections import deque

# ---------- 配置 ----------
MAX_QUESTIONS = None
BASE_DIR      = '../../../data'
EXTENSIONS    = ('.owl', '.rdf', '.rdfs', '.ttl')

# ---------- 缓存 ----------
label_cache = {}
depth_cache = {}

# ---------- 获取标签 ----------
def get_label(entity):
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else entity.name
    label_cache[key] = label
    return label

# ---------- 计算类深度 ----------
def compute_depth(entity):
    if entity in depth_cache:
        return depth_cache[entity]
    queue = deque([(entity, 0)])
    visited = {entity}
    while queue:
        current, dist = queue.popleft()
        if current == owl.Thing:
            depth_cache[entity] = dist
            return dist
        for parent in (p for p in current.is_a if isinstance(p, ThingClass)):
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, dist + 1))
    depth_cache[entity] = float('inf')
    return depth_cache[entity]

# ---------- 提取显式实例-类三元组 ----------
def extract_explicit_type_triples(onto):
    triples = set()
    for inst in onto.individuals():
        for cls in inst.is_a:
            if isinstance(cls, ThingClass):
                triples.add((inst, cls))
    return triples

# ---------- 推理实例-类继承 ----------
def infer_type_triples(explicit, all_individuals):
    inferred = set()
    # 对每个显式类型，向上添加所有祖先类
    for inst, cls in explicit:
        for anc in cls.ancestors():
            if isinstance(anc, ThingClass) and anc != cls and anc != owl.Thing:
                inferred.add((inst, anc))
    return inferred

# ---------- 生成题干 ----------
def make_prompt(inst_label):
    return f"Which of the following classes does '{inst_label}' belong to?"

# ---------- 干扰项生成 ----------
def get_class_distractors(inst, correct_cls, all_classes, num_choices=4):
    # 干扰项：与正确类无继承关系，不在其祖先或子类中
    correct_anc = set(correct_cls.ancestors())
    correct_desc = set(correct_cls.subclasses())
    candidates = []
    for c in all_classes:
        if c == correct_cls:
            continue
        if c in correct_anc or c in correct_desc:
            continue
        candidates.append(c)
    random.shuffle(candidates)
    distractors = candidates[:num_choices-1]
    # 若不足，再随机补足
    if len(distractors) < num_choices-1:
        extras = [c for c in all_classes if c not in distractors and c != correct_cls]
        distractors += random.sample(extras, num_choices-1-len(distractors))
    return distractors

# ---------- 题目生成器 ----------
class TypeQuestionGenerator:
    def __init__(self, implicit, explicit, all_classes):
        # 按实例分组
        self.by_inst = {}
        for inst, cls in implicit:
            self.by_inst.setdefault(inst, []).append(cls)
        self.explicit = explicit
        self.all_classes = all_classes

    def generate_one(self, inst, cls, num_choices=4):
        # 生成选项
        distractors = get_class_distractors(inst, cls, self.all_classes, num_choices)
        options = [cls] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = []
        correct = None
        for i, c in enumerate(options):
            opts.append({'option_letter': letters[i], 'label': get_label(c)})
            if c == cls:
                correct = letters[i]

        # 计算额外 meta 信息
        lbl = get_label(cls)
        d = compute_depth(cls)
        parents = [p for p in cls.is_a if isinstance(p, ThingClass)]
        parent_count = len(parents)
        sibling_count = sum(len(parent.subclasses()) - 1 for parent in parents)
        subclass_count = len(list(cls.subclasses()))

        prompt = make_prompt(get_label(inst))
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'instance_iri':   str(inst.iri),
                'class_iri':      str(cls.iri),
                'iri':            str(cls.iri),
                'label':          lbl,
                'depth':          d,
                'sibling_count':  sibling_count,
                'subclass_count': subclass_count,
                'parent_count':   parent_count,
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        insts = list(self.by_inst.keys())
        random.shuffle(insts)
        for inst in insts:
            for cls in self.by_inst[inst]:
                q = self.generate_one(inst, cls)
                questions.append(q)
                if max_q and len(questions) >= max_q:
                    return questions
        return questions

# ---------- 保存 ----------
def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(questions)} questions to {save_path}")

# ---------- 主流程 ----------
def process_owl_file(file_path, max_q=None):
    world = World()
    onto = world.get_ontology(f"file://{os.path.abspath(file_path)}").load()
    for imp in onto.imported_ontologies:
        try:
            imp.load()
        except:
            pass

    # 显式类型
    explicit = extract_explicit_type_triples(onto)
    logging.info(f"{file_path} - explicit types: {len(explicit)}")

    # 推理类型继承
    inferred = infer_type_triples(explicit, list(onto.individuals()))
    implicit = inferred - explicit
    logging.info(f"{file_path} - inferred types: {len(implicit)}")
    if not implicit:
        return

    all_classes = list(onto.classes())
    gen = TypeQuestionGenerator(implicit, explicit, all_classes)
    questions = gen.generate_all(max_q)

    out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_2_3')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'inst2class_{os.path.basename(file_path)}.json')
    save_questions(questions, save_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    files = []
    for root, _, fnames in os.walk(BASE_DIR):
        for f in fnames:
            if f.lower().endswith(EXTENSIONS):
                files.append(os.path.join(root, f))
    for fp in files:
        try:
            process_owl_file(fp, MAX_QUESTIONS)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")
