import json
import os
import random
import logging
from owlready2 import World, ThingClass, owl

# 全局缓存
label_cache = {}

def get_label(entity):
    """
    取 rdfs:label 或 skos:prefLabel，fallback 到 entity.name
    """
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, "label", []) or getattr(entity, "prefLabel", [])
    label = labs[0] if labs else entity.name
    label_cache[key] = label
    return label

def compute_depth(entity, memo=None):
    if memo is None:
        memo = {}
    if entity in memo:
        return memo[entity]
    if entity == owl.Thing:
        memo[entity] = 0
        return 0
    parents = [p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing]
    depth = 1 if not parents else max(compute_depth(p, memo) for p in parents) + 1
    memo[entity] = depth
    return depth

def get_siblings(entity):
    sibs = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass) and p != owl.Thing:
            sibs.update(c for c in p.subclasses())
    sibs.discard(entity)
    return sibs

def compute_global_metrics(classes):
    max_depth = max_sib = max_sub = max_par = 0
    memo = {}
    for e in classes:
        d   = compute_depth(e, memo)
        s   = len(get_siblings(e))
        sub = len(list(e.subclasses()))
        par = len([p for p in e.is_a if isinstance(p, ThingClass) and p != owl.Thing])
        max_depth = max(max_depth, d)
        max_sib   = max(max_sib, s)
        max_sub   = max(max_sub, sub)
        max_par   = max(max_par, par)
    return dict(
        max_depth=max_depth,
        max_sibling_count=max_sib,
        max_subclass_count=max_sub,
        max_parent_count=max_par
    )

def compute_selection_weight(entity, gm):
    d   = compute_depth(entity)
    s   = len(get_siblings(entity))
    sub = len(list(entity.subclasses()))
    par = len([p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing])
    nd   = d   / gm["max_depth"]            if gm["max_depth"]            else 0
    ns   = s   / gm["max_sibling_count"]    if gm["max_sibling_count"]    else 0
    nsub = sub / gm["max_subclass_count"]   if gm["max_subclass_count"]   else 0
    npar = par / gm["max_parent_count"]     if gm["max_parent_count"]     else 0
    return nd * (ns + 1) / (nsub + npar + 1)

class OntologyLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.world     = World()
        self.onto      = None

    def load(self):
        # 加载本体文件（带 imports）
        iri = f"file://{os.path.abspath(self.file_path)}"
        onto = self.world.get_ontology(iri)
        try:
            onto.load()
        except Exception:
            logging.warning(f"加载用户本体带 imports 失败，尝试本地-only：{self.file_path}")
            onto.load(only_local=True)
        self.onto = onto
        return onto

    def preload_entities(self):
        # 预触发类与实例的属性读取
        for cls in self.onto.classes():
            _ = getattr(cls, "label", None)
            _ = getattr(cls, "prefLabel", None)
        for inst in self.onto.individuals():
            _ = getattr(inst, "label", None)
            _ = getattr(inst, "prefLabel", None)

    def get_all_classes(self):
        return [cls for cls in self.onto.classes() if cls != owl.Thing]

    def get_all_instances(self):
        return list(self.onto.individuals())

class ClassInstanceQuestionGenerator:
    def __init__(self, instances, classes):
        self.instances = instances
        self.classes   = classes
        # 全局指标，用于后续可选的加权筛选
        self.gm        = compute_global_metrics(classes)

    def get_candidate_distractors(self, target_class):
        # 排除 target_class 的所有祖先和后代
        ancestors   = set(target_class.ancestors()) - {target_class, owl.Thing}
        descendants = set(target_class.descendants()) - {target_class}
        excluded    = ancestors | descendants | {target_class, owl.Thing}
        return [c for c in self.classes if c not in excluded]

    def generate_question_for_instance(self, inst):
        inst_label = get_label(inst)
        # 选择一个直接类型作为答案
        types = [t for t in inst.is_a if isinstance(t, ThingClass) and t != owl.Thing]
        if not types:
            return None
        target = random.choice(types)
        target_label = get_label(target)

        # 生成干扰项
        candidates = self.get_candidate_distractors(target)
        random.shuffle(candidates)
        distractors = candidates[:3]
        # 若不足 3 个，再从全局补足
        if len(distractors) < 3:
            others = [c for c in self.classes if c != target and c not in distractors]
            random.shuffle(others)
            for c in others:
                distractors.append(c)
                if len(distractors) >= 3:
                    break

        options = [target] + distractors[:3]
        random.shuffle(options)

        # 构建选项结构
        letters = ['A','B','C','D']
        opts    = []
        correct = None
        for i, c in enumerate(options):
            opts.append({
                "option_letter": letters[i],
                "label":         get_label(c)
            })
            if c == target:
                correct = letters[i]

        # 计算目标类的元数据
        depth          = compute_depth(target)
        sibling_count  = len(get_siblings(target))
        subclass_count = len(list(target.subclasses()))
        parent_count   = len([p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing])

        return {
            "prompt": f"Which of the following classes does '{inst_label}' belong to?",
            "options": opts,
            "correct_answer": correct,
            "meta": {
                "instance_iri":   str(inst.iri),
                "instance_label": inst_label,
                "class_iri":      str(target.iri),
                "class_label":    target_label,
                "depth":          depth,
                "sibling_count":  sibling_count,
                "subclass_count": subclass_count,
                "parent_count":   parent_count
            }
        }

    def generate_all_questions(self):
        questions = []
        skipped   = 0
        for inst in self.instances:
            try:
                q = self.generate_question_for_instance(inst)
                if q:
                    questions.append(q)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        return questions, skipped

def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

def process_owl_file(file_path):
    base      = os.path.splitext(os.path.basename(file_path))[0]
    out_dir   = os.path.dirname(file_path).replace("data", "bench/bench_1_4")
    save_path = os.path.join(out_dir, f"class2inst_{base}.json")
    if os.path.exists(save_path):
        logging.info(f"Skip existing: {save_path}")
        return

    loader    = OntologyLoader(file_path)
    onto      = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return

    loader.preload_entities()
    instances = loader.get_all_instances()
    classes   = loader.get_all_classes()

    gen, sk   = ClassInstanceQuestionGenerator(instances, classes), 0
    qs, sk    = gen.generate_all_questions()
    logging.info(f"Generated {len(qs)} questions (skipped {sk}).")
    if qs:
        save_questions(qs, save_path)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("class2inst.log", "w", "utf-8")
        ]
    )
    random.seed(42)
    base_dir = "../../../data"
    exts     = (".owl", ".rdf", ".rdfs", ".ttl")
    files    = []
    for root, _, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.lower().endswith(exts):
                files.append(os.path.join(root, fname))

    logging.info(f"Found {len(files)} files.")
    for fp in files:
        try:
            process_owl_file(fp)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == "__main__":
    main()
