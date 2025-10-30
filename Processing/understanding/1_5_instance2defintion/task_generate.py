import json
import os
import random
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Optional
from rdflib import URIRef, Literal
from owlready2 import World, ThingClass, onto_path, set_log_level

# 全局缓存
instance_definition_cache = {}
instance_label_cache = {}


def has_annotation_definition(entity):
    """Return True if entity has explicit definition/comment annotations (not counting label/prefLabel)."""
    # 复用 get_definition 的缓存与逻辑，避免重复扫描与 RDFLib 语言标签问题
    return get_definition(entity) != "No definition provided."


def get_definition(entity):
    """Return definition/comment if present (English preferred), else 'No definition provided.'"""
    key = str(entity.iri)
    if key in instance_definition_cache:
        return instance_definition_cache[key]

    defs = []
    defs.extend(getattr(entity, "IAO_0000115", []) or [])
    defs.extend(getattr(entity, "definition", []) or [])
    cmts = getattr(entity, "comment", []) or []
    if isinstance(cmts, str):
        defs.append(cmts)
    else:
        defs.extend(cmts)

    # 扫描其他 annotation properties（仅通过 owlready2，避免 RDFLib 语言标签校验报错）
    world_obj = getattr(entity, "world",
                        getattr(entity.namespace, "world", None))
    if world_obj:
        for ap in world_obj.annotation_properties():
            ap_local = str(ap.iri).split('#')[-1].lower()
            if "definition" in ap_local or "comment" in ap_local:
                vals = getattr(entity, ap.python_name, []) or []
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                defs.extend(vals)

    if not defs:
        definition = "No definition provided."
    else:
        # English 优先
        definition = next((str(d) for d in defs if getattr(d, "lang", None) == 'en'), None)
        if not definition:
            definition = str(defs[0])

    instance_definition_cache[key] = definition
    return definition


def get_label(entity):
    """Get rdfs:label or prefLabel; fallback to name."""
    key = str(entity.iri)
    if key in instance_label_cache:
        return instance_label_cache[key]
    labs = getattr(entity, "label", []) or getattr(entity, "prefLabel", [])
    label = labs[0] if labs else entity.name
    instance_label_cache[key] = label
    return label


class OntologyLoader:
    def __init__(self, file_path: Path, load_imports: bool = True, onto_paths: Optional[List[Path]] = None):
        self.file_path = Path(file_path)
        self.world = World()
        self.onto = None
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
        # Load common annotation ontologies
        for ont in (
            "http://purl.obolibrary.org/obo/iao.owl",
            "http://www.w3.org/2004/02/skos/core#"
        ):
            try:
                self.world.get_ontology(ont).load()
            except Exception as e:
                logging.debug(f"Failed loading annotation ontology {ont}: {e}")
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
                    logging.error(f"Failed to load local-only: {self.file_path} ({e2})")
                    return None
            else:
                logging.error(f"Failed to load local-only: {self.file_path} ({e})")
                return None
        self.onto = onto
        return onto

    def preload_entities(self):
        # 预加载实例注解属性，提升后续访问性能
        for inst in self.onto.individuals():
            _ = getattr(inst, "IAO_0000115", None)
            _ = getattr(inst, "definition", None)
            _ = getattr(inst, "comment", None)
            _ = getattr(inst, "label", None)
            _ = getattr(inst, "prefLabel", None)

    def get_all_instances_with_definition(self):
        # Keep only instances with explicit annotation definitions
        return [
            inst for inst in self.onto.individuals()
            if has_annotation_definition(inst)
        ]


class QuestionGenerator:
    def __init__(self, instances):
        self.instances = instances

    def get_candidate_distractors(self, target):
        # 从同类实例中选干扰项
        cand = set()
        for cls in target.is_a:
            if isinstance(cls, ThingClass):
                cand |= {
                    i for i in cls.instances()
                    if i != target and has_annotation_definition(i)
                }
        # 不足 3 个时随机补充
        if len(cand) < 3:
            others = [i for i in self.instances if i != target]
            random.shuffle(others)
            for o in others:
                cand.add(o)
                if len(cand) >= 3:
                    break
        cand.discard(target)
        return list(cand)

    def generate_question_for_target(self, target):
        # 正确答案的定义与标签
        definition = get_definition(target)
        label = get_label(target)

        # 构造选项（定义文本）
        options = [{"definition": definition, "is_correct": True}]
        distractors = random.sample(self.get_candidate_distractors(target), 3)
        for d in distractors:
            options.append({
                "definition": get_definition(d),
                "is_correct": False
            })

        # 随机打乱
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts, correct = [], None
        for idx, opt in enumerate(options):
            opts.append({
                "option_letter": letters[idx],
                "definition": opt["definition"]
            })
            if opt["is_correct"]:
                correct = letters[idx]

        # choose primary type (if any) as class context for stats
        direct_types = [c for c in target.is_a if isinstance(c, ThingClass)]
        primary = direct_types[0] if direct_types else None
        depth = sibling_count = subclass_count = parent_count = None
        class_context_iri = class_context_label = None
        if primary is not None:
            class_context_iri = str(primary.iri)
            class_context_label = get_label(primary)
            depth = 0
            try:
                depth = 0 if primary == None else 0
                depth = 0 if primary is None else 0
            except Exception:
                pass
            try:
                from owlready2 import owl as _owl, ThingClass as _ThingClass
                def _compute_depth(e, memo=None):
                    if memo is None:
                        memo = {}
                    if e in memo:
                        return memo[e]
                    if e == _owl.Thing:
                        memo[e] = 0
                        return 0
                    parents = [p for p in e.is_a if isinstance(p, _ThingClass) and p != _owl.Thing]
                    d = 1 if not parents else max(_compute_depth(p, memo) for p in parents) + 1
                    memo[e] = d
                    return d
                depth = _compute_depth(primary)
                sibling_count  = len({c for p in primary.is_a if isinstance(p, _ThingClass) and p != _owl.Thing for c in p.subclasses()} - {primary})
                subclass_count = len(list(primary.subclasses()))
                parent_count   = len([p for p in primary.is_a if isinstance(p, _ThingClass) and p != _owl.Thing])
            except Exception:
                pass

        return {
            "prompt": f"Which of the following definitions best describes the instance '{label}'?",
            "options": opts,
            "correct_answer": correct,
            "meta": {
                "subject_iri":          str(target.iri),
                "subject_label":        label,
                "subject_kind":         "instance",
                "relation":             "instance_definition",
                "object_iri":           None,
                "object_label":         None,
                "object_kind":          None,
                "class_context_iri":    class_context_iri,
                "class_context_label":  class_context_label,
                "depth":                depth,
                "sibling_count":        sibling_count,
                "subclass_count":       subclass_count,
                "parent_count":         parent_count,
                "types": [str(c.iri) for c in direct_types],
            }
        }

    def generate_all_questions(self):
        questions, skipped = [], 0
        for inst in self.instances:
            try:
                questions.append(self.generate_question_for_target(inst))
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
    loader = OntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    onto = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    loader.preload_entities()
    insts = loader.get_all_instances_with_definition()
    # Filter instances by origin if needed
    if concept_scope != 'all':
        def is_native(inst) -> bool:
            return getattr(getattr(inst, 'namespace', None), 'ontology', None) is onto
        def is_imported(inst) -> bool:
            o = getattr(getattr(inst, 'namespace', None), 'ontology', None)
            return (o is not None) and (o is not onto)
        if concept_scope == 'native':
            insts = [i for i in insts if is_native(i)]
        else:
            insts = [i for i in insts if is_imported(i)]
    gen = QuestionGenerator(insts)
    questions, skipped = gen.generate_all_questions()
    logging.info(f"Generated {len(questions)} questions (skipped {skipped}).")
    try:
        rel = file_path.relative_to(input_root)
    except Exception:
        rel = file_path.name
    rel_parts = list(Path(rel).parts)
    safe_parts = [slugify_for_windows(p) for p in rel_parts[:-1]]
    safe_stem = slugify_for_windows(Path(rel_parts[-1]).stem if rel_parts else file_path.stem)
    out_dir = output_root.joinpath(*safe_parts, safe_stem)
    save_path = out_dir / f"instance2definition_{safe_stem}.json"
    if questions:
        save_questions(questions, save_path)


def main():
    parser = argparse.ArgumentParser(description='Generate instance-to-definition MCQs from ontologies (instances with explicit definitions).')
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
        handlers=[logging.StreamHandler(), logging.FileHandler('process_1_5.log', 'w', 'utf-8')],
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
