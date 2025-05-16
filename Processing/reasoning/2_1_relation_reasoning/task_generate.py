import json
import os
import random
import logging
import re
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from owlready2 import World, ThingClass, owl, Restriction, sync_reasoner
import signal

# ---------- 配置 ----------
MAX_QUESTIONS = 30000  # 最大问题数量
REASONING_CLASS_THRESHOLD = 5000  # 跳过外部推理的类数量阈值
MAX_CLASSES = 1000  # 最大采样类数量
MAX_TRIPLES = 10000  # 最大采样三元组数量
BASE_DIR = '../../../data'
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')
REASONING_TIMEOUT = 300  # 推理超时时间（秒）

# ---------- 全局缓存 ----------
label_cache = {}
depth_cache = {}
ancestors_cache = defaultdict(set)

# ---------- 获取标签 ----------
def get_label(entity):
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else entity.name
    label_cache[key] = label
    return label

# ---------- 预计算类元数据 ----------
def precompute_class_metadata(onto, classes):
    depths = {owl.Thing: 0}
    ancestors_cache.clear()
    queue = deque([(owl.Thing, 0)])
    visited = set()
    while queue:
        cls, depth = queue.popleft()
        if cls in visited:
            continue
        visited.add(cls)
        depths[cls] = depth
        ancestors_cache[cls].add(cls)
        for parent in (p for p in cls.is_a if isinstance(p, ThingClass)):
            ancestors_cache[cls].update(ancestors_cache[parent])
        for sub in cls.subclasses():
            if sub in classes or sub == owl.Thing:
                queue.append((sub, depth + 1))
    return depths, ancestors_cache

# ---------- 获取同胞类 ----------
def get_siblings(entity, classes):
    sibs = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass) and p != owl.Thing:
            sibs.update(c for c in p.subclasses() if c in classes)
    sibs.discard(entity)
    return sibs

# ---------- 显式三元组提取 ----------
def extract_explicit_class_triples(onto, relations, classes):
    triples = set()
    for cls in classes:
        if 'subclassOf' in relations:
            for parent in cls.is_a:
                if isinstance(parent, ThingClass) and parent != owl.Thing and parent in classes:
                    triples.add((cls, 'subclassOf', parent))
        if 'equivalentTo' in relations:
            for eq in cls.equivalent_to:
                if isinstance(eq, ThingClass) and eq in classes:
                    triples.add((cls, 'equivalentTo', eq))
        for restriction in cls.is_a:
            if isinstance(restriction, Restriction):
                prop = restriction.property
                name = getattr(prop, 'python_name', None)
                if not name or name not in relations:
                    continue
                filler = getattr(restriction, 'value', None) or \
                         getattr(restriction, 'some_values_from', None) or \
                         getattr(restriction, 'all_values_from', None)
                if isinstance(filler, ThingClass) and filler in classes:
                    triples.add((cls, name, filler))
    return triples

# ---------- 自然语言 Prompt 生成 ----------
def humanize_relation(rel_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', rel_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.replace('_', ' ').lower()

def make_prompt(subj_label, rel_name, inferred=False):
    if rel_name == 'subclassOf':
        question = f"Which of the following classes is the superclass of '{subj_label}'?"
    elif rel_name == 'equivalentTo':
        question = f"Which of the following classes is equivalent to '{subj_label}'?"
    else:
        human = humanize_relation(rel_name)
        question = f"Which of the following classes {human} '{subj_label}'?"
    return ("After reasoning, " + question) if inferred else question

# ---------- 题目生成 ----------
class RelationQuestionGenerator:
    def __init__(self, triples, all_classes, inferred=False):
        self.triples = triples
        self.all_classes = all_classes
        self.inferred = inferred
        self.by_subject = defaultdict(list)
        for subj, rel, obj in triples:
            self.by_subject[subj].append((subj, rel, obj))

    def _select_triple(self, subj):
        return random.choice(self.by_subject[subj])

    def _get_distractors(self, subj, rel, obj, num_choices):
        # 对于 subclassOf，排除所有真正的超类；对于 equivalentTo，排除所有等价类；
        # 其他关系，也排除所有真实填充值
        if rel == 'subclassOf':
            disallowed = ancestors_cache[subj]
        elif rel == 'equivalentTo':
            disallowed = set(subj.equivalent_to)
        else:
            disallowed = {f for (s, r, f) in self.triples if s == subj and r == rel}

        # 随机候选
        candidates = random.sample(self.all_classes, min(100, len(self.all_classes)))
        distractors = []
        for c in candidates:
            if c == obj:
                continue
            if c in disallowed:
                continue
            distractors.append(c)
            if len(distractors) >= num_choices - 1:
                break

        # 如果不足，再从剩下的里补
        if len(distractors) < num_choices - 1:
            extra = [c for c in self.all_classes if c not in distractors and c not in disallowed and c != obj]
            distractors += random.sample(extra, min(num_choices - 1 - len(distractors), len(extra)))

        return distractors[:num_choices - 1]

    def generate_one(self, subj, rel, obj, num_choices=4):
        depth = depth_cache.get(subj, 0)
        sibling_count = len(get_siblings(subj, self.all_classes))
        subclass_count = len([c for c in subj.subclasses() if c in self.all_classes])
        parent_count = len([p for p in subj.is_a if isinstance(p, ThingClass) and p != owl.Thing])

        distractors = self._get_distractors(subj, rel, obj, num_choices)
        options = [obj] + distractors
        random.shuffle(options)

        letters = ['A', 'B', 'C', 'D']
        opts = []
        correct = None
        for i, choice in enumerate(options):
            label = get_label(choice)
            opts.append({'option_letter': letters[i], 'label': label})
            if choice == obj:
                correct = letters[i]

        prompt = make_prompt(get_label(subj), rel, self.inferred)
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'iri': str(subj.iri),
                'label': get_label(subj),
                'depth': depth,
                'sibling_count': sibling_count,
                'subclass_count': subclass_count,
                'parent_count': parent_count,
                'relation': rel,
                'object_iri': str(obj.iri),
                'inferred': self.inferred
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        subjects = list(self.by_subject.keys())
        random.shuffle(subjects)
        for subj in subjects:
            subj, rel, obj = self._select_triple(subj)
            try:
                q = self.generate_one(subj, rel, obj)
                questions.append(q)
                if max_q and len(questions) >= max_q:
                    break
            except Exception as e:
                logging.warning(f"Error generating question: {e}")
        return questions

# ---------- 保存 ----------
def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(questions)} questions to {save_path}")

# ---------- 主流程 ----------
def process_owl_file_with_reasoning(file_path, max_q=None):
    logging.info(f"Processing {file_path}")
    world = World()
    onto = world.get_ontology(f"file://{os.path.abspath(file_path)}").load()

    # 加载 imports
    for imp in onto.imported_ontologies:
        try:
            imp.load()
        except Exception:
            pass

    # 采样类
    all_classes = list(onto.classes())
    if len(all_classes) > MAX_CLASSES:
        all_classes = random.sample(all_classes, MAX_CLASSES)
    else:
        all_classes = list(onto.classes())

    # 预计算元数据
    global depth_cache
    depth_cache, _ = precompute_class_metadata(onto, set(all_classes))

    relations = ['subclassOf', 'equivalentTo'] + [prop.python_name for prop in onto.object_properties()]
    explicit = extract_explicit_class_triples(onto, relations, all_classes)

    # 推理或手动计算
    num_classes = len(all_classes)
    inferred = set()
    if num_classes <= REASONING_CLASS_THRESHOLD:
        def timeout_handler(signum, frame):
            raise TimeoutError("Reasoning timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(REASONING_TIMEOUT)
        try:
            with onto:
                sync_reasoner(infer_property_values=True)
            for cls in all_classes:
                for anc in cls.ancestors():
                    if isinstance(anc, ThingClass) and anc != cls and anc != owl.Thing and anc in all_classes:
                        inferred.add((cls, 'subclassOf', anc))
                for eq in cls.equivalent_to:
                    if isinstance(eq, ThingClass) and eq in all_classes:
                        inferred.add((cls, 'equivalentTo', eq))
        except TimeoutError:
            logging.warning(f"{file_path} - Reasoning timed out, using manual inference")
        except Exception as e:
            logging.warning(f"{file_path} - Reasoning failed: {e}")
        finally:
            signal.alarm(0)
    else:
        logging.info(f"{file_path} - Skipping reasoning (classes: {num_classes})")
        for cls in all_classes:
            for anc in ancestors_cache[cls]:
                if anc != cls and anc != owl.Thing and anc in all_classes:
                    inferred.add((cls, 'subclassOf', anc))
            for eq in cls.equivalent_to:
                if isinstance(eq, ThingClass) and eq in all_classes:
                    inferred.add((cls, 'equivalentTo', eq))

    # 处理限制
    for cls in all_classes:
        for restriction in cls.is_a:
            if isinstance(restriction, Restriction):
                prop = restriction.property
                name = getattr(prop, 'python_name', None)
                if not name or name not in relations:
                    continue
                filler = getattr(restriction, 'value', None) or \
                         getattr(restriction, 'some_values_from', None) or \
                         getattr(restriction, 'all_values_from', None)
                if isinstance(filler, ThingClass) and filler in all_classes:
                    inferred.add((cls, name, filler))

    implicit = random.sample(list(inferred - explicit), min(MAX_TRIPLES, len(inferred - explicit)))
    if not implicit:
        logging.info(f"{file_path} - No implicit triples, skipping")
        return

    # 生成问题
    gen = RelationQuestionGenerator(implicit, all_classes, inferred=True)
    questions = gen.generate_all(max_q)

    # 保存结果
    out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_2_1')
    save_path = os.path.join(out_dir, f'relations_inferred_{os.path.basename(file_path)}.json')
    save_questions(questions, save_path)

    # 清空缓存
    label_cache.clear()
    depth_cache.clear()
    ancestors_cache.clear()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    random.seed(42)
    files = [os.path.join(root, fname) for root, _, filenames in os.walk(BASE_DIR)
             for fname in filenames if fname.lower().endswith(EXTENSIONS)]
    logging.info(f"Found {len(files)} files to process")
    with ProcessPoolExecutor() as executor:
        executor.map(process_owl_file_with_reasoning, files, [MAX_QUESTIONS] * len(files))

if __name__ == '__main__':
    main()
