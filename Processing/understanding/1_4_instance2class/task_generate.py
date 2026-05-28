import random
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from owlready2 import ThingClass, owl


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    BaseOntologyLoader,
    build_mirrored_output_dir,
    class_depth,
    class_stats,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    get_definition,
    get_label,
    global_class_metrics,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    selection_weight,
    siblings as class_siblings,
    suppress_library_noise,
)

# 全局缓存
label_cache = {}
ancestor_cache = {}
descendant_cache = {}
equivalent_cache = {}
MAX_DISTRACTOR_SCAN = 1000

PROPP_EXCLUDED_LABELS = {
    'eTrap motif',
    'eTRAP motif',
    'eTRAP added motif',
    'Linking from AaTh-Numbers to ATU-Numbers',
    'Linking back to the ATU source',
}

def cached_label(entity):
    key = str(getattr(entity, 'iri', str(entity)))
    if key not in label_cache:
        label_cache[key] = get_label(entity)
    return label_cache[key]


def normalize_label(text: str) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").lower().split())


def label_tokens(text: str) -> set[str]:
    return {token for token in normalize_label(text).split() if len(token) >= 3}


def labels_too_similar(left: str, right: str) -> bool:
    left_norm = normalize_label(left)
    right_norm = normalize_label(right)
    if left_norm == right_norm:
        return True
    left_tokens = label_tokens(left)
    right_tokens = label_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) > 0.82


def is_valid_named_entity(entity) -> bool:
    label = cached_label(entity)
    if not label or label == "Unnamed":
        return False
    normalized = label.lower()
    return not any(term.lower() in normalized for term in PROPP_EXCLUDED_LABELS)

def compute_depth(entity, memo=None):
    return class_depth(entity, memo=memo)

def get_siblings(entity):
    return class_siblings(entity)


def safe_descendants(entity):
    if entity in descendant_cache:
        return descendant_cache[entity]
    try:
        descendants = set(entity.descendants()) - {entity}
    except Exception:
        seen = set()
        frontier = list(entity.subclasses())
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(list(current.subclasses()))
        descendants = seen
    descendant_cache[entity] = descendants
    return descendants


def equivalent_classes(entity):
    if entity not in equivalent_cache:
        equivalent_cache[entity] = {e for e in getattr(entity, "equivalent_to", []) if isinstance(e, ThingClass)}
    return equivalent_cache[entity]


def cached_ancestors(entity):
    if entity not in ancestor_cache:
        ancestor_cache[entity] = {c for c in entity.ancestors() if isinstance(c, ThingClass)}
    return ancestor_cache[entity]


def true_type_closure(inst) -> set:
    true_types = set()
    for cls in getattr(inst, "is_a", []) or []:
        if not isinstance(cls, ThingClass) or cls == owl.Thing:
            continue
        true_types.add(cls)
        true_types.update(c for c in cached_ancestors(cls) if c != owl.Thing)
        true_types.update(equivalent_classes(cls))
    return true_types


def structurally_related(left, right) -> bool:
    if left == right:
        return True
    if left in equivalent_classes(right) or right in equivalent_classes(left):
        return True
    left_related = cached_ancestors(left) | safe_descendants(left)
    right_related = cached_ancestors(right) | safe_descendants(right)
    return right in left_related or left in right_related


def shared_ancestor_depth(left, right) -> int:
    shared = (cached_ancestors(left) & cached_ancestors(right)) - {owl.Thing}
    if not shared:
        return -1
    return max(compute_depth(ancestor) for ancestor in shared)


def is_valid_distractor_candidate(candidate, target_class, forbidden) -> bool:
    return (
        isinstance(candidate, ThingClass)
        and candidate not in forbidden
        and is_valid_named_entity(candidate)
        and not structurally_related(candidate, target_class)
        and shared_ancestor_depth(candidate, target_class) >= 0
    )

def compute_global_metrics(classes):
    return global_class_metrics(classes)

def compute_selection_weight(entity, gm):
    return selection_weight(entity, gm)

class OntologyLoader(BaseOntologyLoader):

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
        self.classes   = [c for c in classes if is_valid_named_entity(c)]
        # 全局指标，用于后续可选的加权筛选
        self.gm        = compute_global_metrics(self.classes)

    def get_candidate_distractors(self, target_class, true_classes):
        forbidden = (
            set(true_classes)
            | cached_ancestors(target_class)
            | safe_descendants(target_class)
            | equivalent_classes(target_class)
            | {target_class, owl.Thing}
        )

        target_depth = compute_depth(target_class)
        candidates = []
        seen = set()

        def add_candidate(candidate):
            if candidate in seen or not is_valid_distractor_candidate(candidate, target_class, forbidden):
                return
            seen.add(candidate)
            candidates.append(candidate)

        for sibling in get_siblings(target_class):
            add_candidate(sibling)

        local_pool = []
        ancestors = list(cached_ancestors(target_class) - {target_class, owl.Thing})
        random.shuffle(ancestors)
        for ancestor in ancestors[:50]:
            for branch in ancestor.subclasses():
                if branch == target_class:
                    continue
                local_pool.append(branch)
                for child in branch.subclasses():
                    if abs(compute_depth(child) - target_depth) <= 1:
                        local_pool.append(child)

        local_pool = list(set(local_pool))
        random.shuffle(local_pool)
        for candidate in local_pool:
            add_candidate(candidate)
            if len(candidates) >= 20:
                break

        if len(candidates) < 3:
            pool = list(self.classes)
            random.shuffle(pool)
            for idx, candidate in enumerate(pool):
                if idx >= MAX_DISTRACTOR_SCAN:
                    break
                if (
                    candidate in seen
                    or candidate in forbidden
                    or abs(compute_depth(candidate) - target_depth) > 1
                    or shared_ancestor_depth(candidate, target_class) < 0
                ):
                    continue
                add_candidate(candidate)
                if len(candidates) >= 20:
                    break
        return candidates

    def validate_unique_answer(self, inst, target, options) -> bool:
        true_classes = true_type_closure(inst)
        if target not in true_classes:
            return False
        true_options = [option for option in options if option in true_classes]
        if true_options != [target]:
            return False
        for option in options:
            if option == target:
                continue
            if structurally_related(option, target):
                return False
        labels = [cached_label(option) for option in options]
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                if labels_too_similar(left, right):
                    return False
        return True

    def generate_question_for_instance(self, inst):
        inst_label = cached_label(inst)
        # 选择一个直接类型作为答案
        if not is_valid_named_entity(inst):
            return None
        types = [
            t for t in inst.is_a
            if isinstance(t, ThingClass) and t != owl.Thing and is_valid_named_entity(t) and compute_depth(t) >= 2
        ]
        if not types:
            return None
        target = sorted(types, key=compute_depth, reverse=True)[0]
        target_label = cached_label(target)

        # 生成干扰项
        true_classes = true_type_closure(inst)
        candidates = self.get_candidate_distractors(target, true_classes)
        random.shuffle(candidates)
        distractors = []
        seen_labels = {target_label}
        for candidate in candidates:
            label = cached_label(candidate)
            if any(labels_too_similar(label, used) for used in seen_labels):
                continue
            distractors.append(candidate)
            seen_labels.add(label)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            return None

        options = [target] + distractors[:3]
        if not self.validate_unique_answer(inst, target, options):
            return None
        random.shuffle(options)

        # 构建选项结构
        letters = ['A','B','C','D']
        opts    = []
        correct = None
        for i, c in enumerate(options):
            label = cached_label(c)
            if label == "Unnamed":
                return None
            opts.append({
                "option_letter": letters[i],
                "label":         label
            })
            if c == target:
                correct = letters[i]

        # 计算目标类的元数据
        stats = class_stats(target)
        depth = stats.depth
        sibling_count = stats.sibling_count
        subclass_count = stats.subclass_count
        parent_count = stats.parent_count

        prompt = f"Which of the following classes does '{inst_label}' belong to?"
        inst_context = get_definition(inst)
        if inst_context and inst_context != "No definition provided.":
            prompt = f"{prompt} Context: {inst_context}"

        return {
            "prompt": prompt,
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

    def generate_all_questions(self, max_questions=None):
        questions = []
        skipped   = 0
        for inst in self.instances:
            try:
                q = self.generate_question_for_instance(inst)
                if q:
                    questions.append(q)
                    if max_questions and len(questions) >= max_questions:
                        return questions, skipped
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        return questions, skipped

def save_questions(questions, save_path):
    save_json(questions, Path(save_path), description="questions")


def process_owl_file(file_path: Path, input_root: Path, output_root: Path, load_imports: bool, onto_paths: Optional[List[Path]], suppress_warnings: bool, concept_scope: str = 'all', max_questions: Optional[int] = None) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f"class2inst_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "class2inst", safe_stem)
    if save_path.exists():
        logging.info("Skip existing: %s", save_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

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
    qs, sk    = gen.generate_all_questions(max_questions=max_questions)
    logging.info(f"Generated {len(qs)} questions (skipped {sk}).")
    if qs:
        save_questions(qs, save_path)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_instance_class_questions",
            extra={"instances": len(instances), "classes": len(classes), "skipped": sk},
        )

def main():
    parser = argparse.ArgumentParser(description='Generate instance-to-class MCQs from ontologies.')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory with Windows-safe mirrored structure.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all', 'native', 'imported'], default='all', help='Filter by origin of instances: all/native/imported.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    configure_logging(args.log, "process_1_4.log")
    suppress_library_noise(args.no_warnings)
    random.seed(args.seed)
    max_q = None if args.max_questions == 0 else args.max_questions
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    exts = ('.owl', '.rdf', '.rdfs', '.ttl')
    files, input_root = discover_ontology_files(input_path, exts)

    logging.info(f"Found {len(files)} files.")
    onto_paths = resolve_onto_paths(args.onto_path)
    for fp in files:
        try:
            with file_timeout(args.file_timeout_seconds):
                process_owl_file(
                    file_path=fp,
                    input_root=input_root,
                    output_root=output_root,
                    load_imports=not args.no_imports,
                    onto_paths=onto_paths,
                    suppress_warnings=args.no_warnings,
                    concept_scope=args.concept_scope,
                    max_questions=max_q,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == "__main__":
    main()
