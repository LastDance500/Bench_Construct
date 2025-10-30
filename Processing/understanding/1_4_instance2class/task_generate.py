import json
import os
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import World, ThingClass, owl, onto_path, set_log_level

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
    def __init__(self, file_path: Path, load_imports: bool = True, onto_paths: Optional[List[Path]] = None):
        self.file_path = Path(file_path)
        self.world     = World()
        self.onto      = None
        self.load_imports = load_imports
        if onto_paths:
            for p in onto_paths:
                try:
                    pp = str(Path(p).resolve())
                    if pp not in self.world._ontology_path:
                        self.world._ontology_path.append(pp)
                    if pp not in onto_path:
                        onto_path.append(pp)
                except Exception:
                    pass

    def load(self):
        # Load ontology
        iri = f"file://{self.file_path.resolve()}"
        onto = self.world.get_ontology(iri)
        try:
            if self.load_imports:
                onto.load()
            else:
                onto.load(only_local=True)
        except Exception as e:
            if self.load_imports:
                logging.warning(f"Failed to load with imports; retrying local-only: {self.file_path} ({e})")
                try:
                    onto.load(only_local=True)
                except Exception as e2:
                    logging.error(f"Failed local-only: {self.file_path} ({e2})")
                    return None
            else:
                logging.error(f"Failed local-only: {self.file_path} ({e})")
                return None
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
                "subject_iri":         str(inst.iri),
                "subject_label":       inst_label,
                "subject_kind":        "instance",
                "relation":            "instance_of",
                "object_iri":          str(target.iri),
                "object_label":        target_label,
                "object_kind":         "class",
                "class_context_iri":   str(target.iri),
                "class_context_label": target_label,
                "depth":               depth,
                "sibling_count":       sibling_count,
                "subclass_count":      subclass_count,
                "parent_count":        parent_count
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


def process_owl_file(file_path: Path, input_root: Path, output_root: Path, load_imports: bool, onto_paths: Optional[List[Path]], suppress_warnings: bool, concept_scope: str = 'all') -> None:
    loader    = OntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    onto      = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return

    loader.preload_entities()
    instances = loader.get_all_instances()
    # Filter instances by origin if needed
    if concept_scope != 'all':
        def is_native(inst) -> bool:
            return getattr(getattr(inst, 'namespace', None), 'ontology', None) is onto
        def is_imported(inst) -> bool:
            o = getattr(getattr(inst, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            instances = [i for i in instances if is_native(i)]
        else:
            instances = [i for i in instances if is_imported(i)]
    classes   = loader.get_all_classes()

    gen, sk   = ClassInstanceQuestionGenerator(instances, classes), 0
    qs, sk    = gen.generate_all_questions()
    logging.info(f"Generated {len(qs)} questions (skipped {sk}).")
    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir   = output_root.joinpath(*safe_parts, safe_stem)
    save_path = out_dir / f"class2inst_{safe_stem}.json"
    if qs:
        save_questions(qs, save_path)

def main():
    parser = argparse.ArgumentParser(description='Generate instance-to-class MCQs from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory with Windows-safe mirrored structure.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all', 'native', 'imported'], default='all', help='Filter by origin of instances: all/native/imported.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('process_1_4.log', 'w', 'utf-8')],
    )
    if args.no_warnings:
        try:
            set_log_level(0)
        except Exception:
            pass
        warnings.filterwarnings('ignore')
        for name in ('owlready2', 'rdflib'):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass
    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')
    files: List[Path] = []
    if input_path.is_file() and input_path.suffix.lower() in exts:
        files = [input_path]
        input_root = input_path.parent
    else:
        input_root = input_path
        for root, _, filenames in os.walk(str(input_path)):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    files.append(Path(root) / fname)

    logging.info(f"Found {len(files)} files.")
    for fp in files:
        try:
            process_owl_file(
                file_path=fp,
                input_root=input_root,
                output_root=output_root,
                load_imports=not args.no_imports,
                onto_paths=[Path(p) for p in args.onto_path] if args.onto_path else None,
                suppress_warnings=args.no_warnings,
                concept_scope=args.concept_scope,
            )
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == "__main__":
    main()
