import json
import os
import random
import logging
import argparse
import warnings
import sys
from pathlib import Path
from typing import List, Optional
from owlready2 import World, ThingClass, owl, onto_path, set_log_level
from collections import deque


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    build_mirrored_output_dir,
    class_depth,
    class_stats,
    configure_world_paths,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    get_definition,
    get_label,
    load_ontology,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)

# Defaults
EXTENSIONS    = ('.owl', '.rdf', '.rdfs', '.ttl')

depth_cache = {}
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


def is_valid_named_entity(entity) -> bool:
    label = get_label(entity)
    if not label or label == 'Unnamed':
        return False
    normalized = label.lower()
    return not any(term.lower() in normalized for term in PROPP_EXCLUDED_LABELS)


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


def safe_descendants(cls):
    if cls in descendant_cache:
        return descendant_cache[cls]
    try:
        descendants = set(cls.descendants()) - {cls}
    except Exception:
        frontier = list(cls.subclasses())
        seen = set(frontier)
        while frontier:
            cur = frontier.pop()
            for sub in cur.subclasses():
                if sub not in seen:
                    seen.add(sub)
                    frontier.append(sub)
        descendants = seen
    descendant_cache[cls] = descendants
    return descendants


def equivalent_classes(cls):
    if cls not in equivalent_cache:
        equivalent_cache[cls] = {e for e in getattr(cls, 'equivalent_to', []) if isinstance(e, ThingClass)}
    return equivalent_cache[cls]


def cached_ancestors(cls):
    if cls not in ancestor_cache:
        ancestor_cache[cls] = {c for c in cls.ancestors() if isinstance(c, ThingClass)}
    return ancestor_cache[cls]


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


def is_valid_distractor_candidate(candidate, correct_cls, forbidden) -> bool:
    return (
        isinstance(candidate, ThingClass)
        and candidate not in forbidden
        and is_valid_named_entity(candidate)
        and not structurally_related(candidate, correct_cls)
        and shared_ancestor_depth(candidate, correct_cls) >= 0
    )

# ---------- class depth ----------
def compute_depth(entity):
    if entity not in depth_cache:
        depth_cache[entity] = class_depth(entity)
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
        for anc in cached_ancestors(cls):
            if anc != cls and anc != owl.Thing:
                inferred.add((inst, anc))
    return inferred

# ---------- prompt ----------
def make_prompt(inst_label):
    return f"After reasoning over the ontology, which class is inferred for the instance '{inst_label}'?"

# ---------- distractors ----------
def get_class_distractors(inst, correct_cls, all_classes, true_classes_for_inst, num_choices=4):
    # 干扰项：
    # 1) 与正确类无任何继承/等价关系（排除所有祖先与所有后代、等价类）
    # 2) 不是该实例的真实类型（显式或推理得到的所有类型）
    # 3) 选项标签与正确答案及已选干扰项标签不重复，避免标签歧义

    correct_anc = cached_ancestors(correct_cls)
    correct_desc_all = safe_descendants(correct_cls)
    equiv = equivalent_classes(correct_cls)

    forbidden = correct_anc | correct_desc_all | equiv | set(true_classes_for_inst) | {correct_cls, owl.Thing}

    correct_depth = compute_depth(correct_cls)
    candidates = []
    seen = set()

    def add_candidate(candidate):
        if candidate in seen or not is_valid_distractor_candidate(candidate, correct_cls, forbidden):
            return
        seen.add(candidate)
        candidates.append(candidate)

    for parent in correct_cls.is_a:
        if isinstance(parent, ThingClass):
            for sibling in parent.subclasses():
                if sibling != correct_cls:
                    add_candidate(sibling)

    if len(candidates) < num_choices - 1:
        local_pool = []
        ancestors = list(cached_ancestors(correct_cls) - {correct_cls, owl.Thing})
        random.shuffle(ancestors)
        for ancestor in ancestors[:50]:
            for branch in ancestor.subclasses():
                if branch == correct_cls:
                    continue
                local_pool.append(branch)
                for child in branch.subclasses():
                    if abs(compute_depth(child) - correct_depth) <= 1:
                        local_pool.append(child)
        local_pool = list(set(local_pool))
        random.shuffle(local_pool)
        for candidate in local_pool:
            add_candidate(candidate)
            if len(candidates) >= max(num_choices * 5, 20):
                break

    if len(candidates) < num_choices - 1:
        pool = list(all_classes)
        random.shuffle(pool)
        for idx, candidate in enumerate(pool):
            if idx >= MAX_DISTRACTOR_SCAN:
                break
            if (
                candidate in seen
                or candidate in forbidden
                or abs(compute_depth(candidate) - correct_depth) > 1
                or shared_ancestor_depth(candidate, correct_cls) < 0
            ):
                continue
            add_candidate(candidate)
            if len(candidates) >= max(num_choices * 5, 20):
                break
    random.shuffle(candidates)

    # 通过标签去重，避免产生标签相同导致的歧义
    selected = []
    used_labels = {get_label(correct_cls)}
    for c in candidates:
        lab = get_label(c)
        if any(labels_too_similar(lab, used) for used in used_labels):
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
        if not (is_valid_named_entity(inst) and is_valid_named_entity(cls)):
            return None
        # Generate options
        true_classes = self.true_classes_by_inst.get(inst, set())
        distractors = get_class_distractors(inst, cls, self.all_classes, true_classes, num_choices)
        if not distractors:
            return None
        options = [cls] + distractors
        if not self.validate_unique_answer(inst, cls, options):
            return None
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
        stats = class_stats(cls, self.all_classes)
        d = stats.depth
        parent_count = stats.parent_count
        sibling_count = stats.sibling_count
        subclass_count = stats.subclass_count

        prompt = make_prompt(get_label(inst))
        inst_context = get_definition(inst)
        if inst_context and inst_context != "No definition provided.":
            prompt = f"{prompt} Context: {inst_context}"
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

    def validate_unique_answer(self, inst, cls, options):
        true_classes = self.true_classes_by_inst.get(inst, set())
        if cls not in true_classes:
            return False
        true_options = [option for option in options if option in true_classes]
        if true_options != [cls]:
            return False
        labels = [get_label(option) for option in options]
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                if labels_too_similar(left, right):
                    return False
        for option in options:
            if option == cls:
                continue
            if structurally_related(option, cls):
                return False
        return True

    def generate_all(self, max_q=None):
        questions = []
        insts = list(self.by_inst.keys())
        random.shuffle(insts)
        for inst in insts:
            candidates = sorted(
                [cls for cls in self.by_inst[inst] if compute_depth(cls) >= 2 and is_valid_named_entity(cls)],
                key=compute_depth,
                reverse=True,
            )
            if not candidates:
                continue
            for cls in candidates[:1]:
                q = self.generate_one(inst, cls)
                if q is None:
                    # 跳过无法保证唯一性或干扰项不足的题目
                    continue
                questions.append(q)
                if max_q and len(questions) >= max_q:
                    return questions
        return questions

def save_questions(questions, save_path: Path):
    save_json(questions, save_path, description="questions")

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
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f'inst2class_inferred_{safe_stem}.json'
    empty_path = empty_marker_path(out_dir, "inst2class_inferred", safe_stem)
    if save_path.exists():
        logging.info("Skip existing: %s", save_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    world = World()
    configure_world_paths(world, onto_paths)
    onto = load_ontology(world, file_path, load_imports=load_imports)
    if onto is None:
        return

    explicit = extract_explicit_type_triples(onto)
    inferred = infer_type_triples(explicit)
    implicit = inferred - explicit
    if not implicit:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_implicit_instance_type_triples",
            extra={"explicit_type_triples": len(explicit), "inferred_type_triples": len(inferred)},
        )
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
            save_empty_marker(
                empty_path,
                source_file=file_path,
                reason="no_implicit_instance_type_triples_after_scope_filter",
                extra={"explicit_type_triples": len(explicit), "inferred_type_triples": len(inferred)},
            )
            return

    all_classes = [c for c in onto.classes() if is_valid_named_entity(c)]
    gen = TypeQuestionGenerator(implicit, explicit, all_classes)
    questions = gen.generate_all(max_questions)

    if questions:
        save_questions(questions, save_path)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_inferred_instance_questions",
            extra={"implicit_type_triples": len(implicit)},
        )

def main():
    parser = argparse.ArgumentParser(description='Generate instance reasoning MCQs (instance -> inferred classes).')
    parser.add_argument('--input', type=str, required=True, help='Input ontology file or directory (.owl/.rdf/.rdfs/.ttl).')
    parser.add_argument('--output', type=str, required=True, help='Output root directory (Windows-safe mirrored).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-questions', type=int, default=0, help='Max questions per ontology (0 means all).')
    parser.add_argument('--file-timeout-seconds', type=int, default=0, help='Skip a single ontology file if processing exceeds this many seconds (0 means no timeout).')
    parser.add_argument('--no-imports', action='store_true', help='Do not load imports (local-only).')
    parser.add_argument('--onto-path', action='append', default=None, help='Local directories to resolve owl:imports (can repeat).')
    parser.add_argument('--concept-scope', type=str, choices=['all','native','imported'], default='all', help='Filter by origin of instances: all/native/imported.')
    parser.add_argument('--no-warnings', action='store_true', help='Suppress warnings and library noise.')
    parser.add_argument('--log', type=str, default='info', help='Logging level: debug, info, warning, error.')

    args = parser.parse_args()

    configure_logging(args.log, "process_2_3.log")
    suppress_library_noise(args.no_warnings)

    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    files, input_root = discover_ontology_files(input_path, EXTENSIONS)
    max_q = None if args.max_questions == 0 else args.max_questions
    onto_paths = resolve_onto_paths(args.onto_path)
    for fp in files:
        try:
            with file_timeout(args.file_timeout_seconds):
                process_owl_file(
                    file_path=fp,
                    input_root=input_root,
                    output_root=output_root,
                    max_questions=max_q,
                    load_imports=not args.no_imports,
                    onto_paths=onto_paths,
                    concept_scope=args.concept_scope,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")

if __name__ == '__main__':
    main()
