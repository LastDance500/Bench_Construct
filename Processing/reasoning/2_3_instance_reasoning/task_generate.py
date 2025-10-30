import json
import os
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Optional
from owlready2 import World, ThingClass, owl, onto_path, set_log_level
from collections import deque

# Defaults
EXTENSIONS    = ('.owl', '.rdf', '.rdfs', '.ttl')

# ---------- 缓存 ----------
label_cache = {}
depth_cache = {}

# ---------- label helper ----------
def get_label(entity):
    key = str(entity.iri)
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else entity.name
    label_cache[key] = label
    return label

# ---------- class depth ----------
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

# ---------- extract explicit instance->class triples ----------
def extract_explicit_type_triples(onto):
    triples = set()
    for inst in onto.individuals():
        for cls in inst.is_a:
            if isinstance(cls, ThingClass):
                triples.add((inst, cls))
    return triples

# ---------- infer instance->ancestor class triples ----------
def infer_type_triples(explicit):
    inferred = set()
    # 对每个显式类型，向上添加所有祖先类（传递闭包）
    for inst, cls in explicit:
        for anc in cls.ancestors():
            if isinstance(anc, ThingClass) and anc != cls and anc != owl.Thing:
                inferred.add((inst, anc))
    return inferred

# ---------- prompt ----------
def make_prompt(inst_label):
    return f"Which of the following classes does '{inst_label}' belong to?"

# ---------- distractors ----------
def get_class_distractors(inst, correct_cls, all_classes, true_classes_for_inst, num_choices=4):
    # 干扰项：
    # 1) 与正确类无任何继承/等价关系（排除所有祖先与所有后代、等价类）
    # 2) 不是该实例的真实类型（显式或推理得到的所有类型）
    # 3) 选项标签与正确答案及已选干扰项标签不重复，避免标签歧义

    # 祖先与所有后代（传递）
    correct_anc = set(correct_cls.ancestors())
    # owlready2 的 descendants() 为传递后代
    try:
        correct_desc_all = set(correct_cls.descendants())
    except Exception:
        # 兼容性兜底：若不可用，递归展开
        frontier = list(correct_cls.subclasses())
        seen = set(frontier)
        while frontier:
            cur = frontier.pop()
            for sub in cur.subclasses():
                if sub not in seen:
                    seen.add(sub)
                    frontier.append(sub)
        correct_desc_all = seen

    # 等价类
    equiv = set(e for e in getattr(correct_cls, 'equivalent_to', []) if isinstance(e, ThingClass))

    forbidden = correct_anc | correct_desc_all | equiv | set(true_classes_for_inst) | {correct_cls, owl.Thing}

    # 候选初筛
    candidates = [c for c in all_classes if isinstance(c, ThingClass) and c not in forbidden]
    random.shuffle(candidates)

    # 通过标签去重，避免产生标签相同导致的歧义
    selected = []
    used_labels = {get_label(correct_cls)}
    for c in candidates:
        lab = get_label(c)
        if lab in used_labels:
            continue
        selected.append(c)
        used_labels.add(lab)
        if len(selected) >= num_choices - 1:
            break

    # 不足则返回空，跳过该题，确保选项质量与唯一性
    if len(selected) < num_choices - 1:
        return None

    return selected

# ---------- question generator ----------
class TypeQuestionGenerator:
    def __init__(self, implicit, explicit, all_classes):
        # 按实例分组
        self.by_inst = {}
        for inst, cls in implicit:
            self.by_inst.setdefault(inst, []).append(cls)
        self.explicit = explicit
        self.all_classes = all_classes

        # 为每个实例构建其真实类型全集（显式 + 推理）用于过滤干扰项
        self.true_classes_by_inst = {}
        all_true_pairs = set(implicit) | set(explicit)
        for inst, cls in all_true_pairs:
            self.true_classes_by_inst.setdefault(inst, set()).add(cls)

    def generate_one(self, inst, cls, num_choices=4):
        # Generate options
        true_classes = self.true_classes_by_inst.get(inst, set())
        distractors = get_class_distractors(inst, cls, self.all_classes, true_classes, num_choices)
        if not distractors:
            return None
        options = [cls] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = []
        correct = None
        for i, c in enumerate(options):
            opts.append({'option_letter': letters[i], 'label': get_label(c)})
            if c == cls:
                correct = letters[i]

        # Metadata
        lbl = get_label(cls)
        d = compute_depth(cls)
        parents = [p for p in cls.is_a if isinstance(p, ThingClass)]
        parent_count = len(parents)
        sibling_count = sum(max(len(list(parent.subclasses())) - 1, 0) for parent in parents)
        subclass_count = len(list(cls.subclasses()))

        prompt = make_prompt(get_label(inst))
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'subject_iri':         str(inst.iri),
                'subject_label':       get_label(inst),
                'subject_kind':        'instance',
                'relation':            'instance_of_inferred',
                'object_iri':          str(cls.iri),
                'object_label':        lbl,
                'object_kind':         'class',
                'class_context_iri':   str(cls.iri),
                'class_context_label': lbl,
                'depth':               d,
                'sibling_count':       sibling_count,
                'subclass_count':      subclass_count,
                'parent_count':        parent_count,
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        insts = list(self.by_inst.keys())
        random.shuffle(insts)
        for inst in insts:
            for cls in self.by_inst[inst]:
                q = self.generate_one(inst, cls)
                if q is None:
                    # 跳过无法保证唯一性或干扰项不足的题目
                    continue
                questions.append(q)
                if max_q and len(questions) >= max_q:
                    return questions
        return questions

def slugify_for_windows(name: str) -> str:
    safe = []
    prev_us = False
    for ch in name:
        if ch.isalnum() or ch in ("-", "."):
            safe.append(ch)
            prev_us = False
        else:
            if not prev_us:
                safe.append("_")
            prev_us = True
    s = "".join(safe).strip("_")
    return s or "unnamed"


def save_questions(questions, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(questions)} questions to {save_path}")

# ---------- 主流程 ----------
def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    max_questions: Optional[int],
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    concept_scope: str,
) -> None:
    world = World()
    if onto_paths:
        for p in onto_paths:
            try:
                pp = str(Path(p).resolve())
                if pp not in world._ontology_path:
                    world._ontology_path.append(pp)
                if pp not in onto_path:
                    onto_path.append(pp)
            except Exception:
                pass
    iri = f"file://{file_path.resolve()}"
    onto = world.get_ontology(iri)
    try:
        if load_imports:
            onto.load()
        else:
            onto.load(only_local=True)
    except Exception as e:
        if load_imports:
            logging.warning(f"Failed loading with imports; retrying local-only: {file_path} ({e})")
            try:
                onto.load(only_local=True)
            except Exception as e2:
                logging.error(f"Failed local-only: {file_path} ({e2})")
                return
        else:
            logging.error(f"Failed local-only: {file_path} ({e})")
            return

    explicit = extract_explicit_type_triples(onto)
    inferred = infer_type_triples(explicit)
    implicit = inferred - explicit
    if not implicit:
        return

    # concept-scope on subject instances
    if concept_scope != 'all':
        def is_native(inst):
            return getattr(getattr(inst, 'namespace', None), 'ontology', None) is onto
        def is_imported(inst):
            o = getattr(getattr(inst, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            implicit = [(i, c) for (i, c) in implicit if is_native(i)]
        else:
            implicit = [(i, c) for (i, c) in implicit if is_imported(i)]
        if not implicit:
            return

    all_classes = list(onto.classes())
    gen = TypeQuestionGenerator(implicit, explicit, all_classes)
    questions = gen.generate_all(max_questions)

    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    save_path = out_dir / f'inst2class_inferred_{safe_stem}.json'
    if questions:
        save_questions(questions, save_path)

def main():
    parser = argparse.ArgumentParser(description='Generate instance reasoning MCQs (instance -> inferred classes).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of instances: all/native/imported.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('process_2_3.log', 'w', 'utf-8')])
    if args.no_warnings:
        try:
            set_log_level(0)
        except Exception:
            pass
        warnings.filterwarnings('ignore')
        for name in ('owlready2','rdflib'):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in EXTENSIONS:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, fnames in os.walk(str(input_path)):
            for f in fnames:
                if f.lower().endswith(EXTENSIONS):
                    files.append(Path(root) / f)
    max_q = None if args.max_questions == 0 else args.max_questions
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                max_questions=max_q,
                load_imports=not args.no_imports,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                concept_scope=args.concept_scope,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()
