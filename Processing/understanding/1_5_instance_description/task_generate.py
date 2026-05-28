import json
import os
import random
import logging
import argparse
import warnings
import sys
import re
from pathlib import Path
from typing import List, Optional
from rdflib import URIRef, Literal
from owlready2 import World, ThingClass, onto_path, set_log_level


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (
    BaseOntologyLoader,
    build_mirrored_output_dir,
    class_stats,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    get_definition as shared_get_definition,
    get_label as shared_get_label,
    resolve_onto_paths,
    save_empty_marker,
    save_json,
    slugify_for_windows,
    suppress_library_noise,
)

# 全局缓存
instance_definition_cache = {}
instance_label_cache = {}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").replace("-", " ")).strip().lower()


def normalize_option_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def text_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_text(text)) if len(token) >= 3}


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def is_low_quality_definition(text: str) -> bool:
    normalized = normalize_text(text)
    return (
        not normalized
        or normalized in {"none", "no definition", "no definition provided.", "the instance", "the instance."}
        or normalized.startswith("of ")
        or normalized.startswith("note ")
        or normalized.startswith("note:")
        or normalized.startswith("_:")
        or "http://" in normalized
        or "https://" in normalized
        or "improperly formatted iri" in normalized
        or "error" in normalized
        or "current axiom" in normalized
        or "will have to be" in normalized
        or "what about" in normalized
        or "added in the ontology" in normalized
        or "play with inconsistencies" in normalized
        or len(normalized.split()) < 4
        or len(normalized) < 24
    )


def has_annotation_definition(entity):
    """Return True if entity has explicit definition/comment annotations (not counting label/prefLabel)."""
    # 复用 get_definition 的缓存与逻辑，避免重复扫描与 RDFLib 语言标签问题
    return not is_low_quality_definition(get_definition(entity))


def get_definition(entity):
    key = str(entity.iri)
    if key in instance_definition_cache:
        return instance_definition_cache[key]
    definition = shared_get_definition(entity)
    instance_definition_cache[key] = definition
    return definition


def get_label(entity):
    key = str(entity.iri)
    if key in instance_label_cache:
        return instance_label_cache[key]
    label = shared_get_label(entity)
    instance_label_cache[key] = label
    return label


class OntologyLoader(BaseOntologyLoader):
    def annotation_ontology_iris(self) -> tuple[str, ...]:
        return (
            "http://purl.obolibrary.org/obo/iao.owl",
            "http://www.w3.org/2004/02/skos/core#",
        )

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
    def __init__(self, instances, max_questions: Optional[int] = None, mask_instance: bool = True):
        self.instances = instances
        self.max_questions = max_questions
        self.mask_instance = mask_instance

    def _collect_aliases(self, entity) -> List[str]:
        aliases = set()
        for attr in ("label", "prefLabel", "altLabel", "alternativeLabel", "hasExactSynonym", "hasRelatedSynonym", "hasBroadSynonym", "hasNarrowSynonym"):
            vals = getattr(entity, attr, []) or []
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            for value in vals:
                try:
                    aliases.add(str(value))
                except Exception:
                    pass
        try:
            iri_s = str(entity.iri)
            aliases.add(iri_s.split("#")[-1] if "#" in iri_s else iri_s.rsplit("/", 1)[-1])
        except Exception:
            pass
        variants = {alias.replace("_", " ").replace("-", " ") for alias in aliases}
        aliases |= variants
        return [alias for alias in aliases if len(alias.strip()) >= 2]

    def _mask_definition_text(self, text: str, entity) -> str:
        if not self.mask_instance or not text:
            return text
        masked = text
        for alias in sorted(self._collect_aliases(entity), key=len, reverse=True):
            escaped = re.escape(alias)
            pattern = re.compile(rf"(?i)(?<!\w){escaped}(?!\w)")
            masked = pattern.sub("the instance", masked)
        masked = re.sub(r"(?i)\b(a|an)\s+the instance\b", "the instance", masked)
        masked = re.sub(r"(?i)\bthe\s+the instance\b", "the instance", masked)
        masked = re.sub(r"\s+", " ", masked).strip()
        return masked

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
        raw_definition = get_definition(target)
        label = get_label(target)
        definition = self._mask_definition_text(raw_definition, target)
        if is_low_quality_definition(definition):
            raise ValueError("low quality masked definition")
        answer_leak_risk = lexical_overlap(label, definition) > 0.45
        if answer_leak_risk:
            raise ValueError("definition leaks target label")

        # 构造选项（定义文本）
        used_definitions = {normalize_option_text(definition)}
        options = [{"definition": definition, "is_correct": True}]
        candidate_distractors = self.get_candidate_distractors(target)
        if len(candidate_distractors) < 3:
            raise ValueError("not enough distractors")
        random.shuffle(candidate_distractors)
        for d in candidate_distractors:
            masked = self._mask_definition_text(get_definition(d), d)
            normalized_masked = normalize_option_text(masked)
            if not normalized_masked or normalized_masked in used_definitions:
                continue
            if is_low_quality_definition(masked):
                continue
            if lexical_overlap(label, masked) > 0.45:
                continue
            options.append({
                "definition": masked,
                "is_correct": False
            })
            used_definitions.add(normalized_masked)
            if len(options) >= 4:
                break
        if len(options) < 4:
            raise ValueError("not enough non-leaking distractors")
        if len({normalize_option_text(option["definition"]) for option in options}) != 4:
            raise ValueError("duplicate option definitions")

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
            stats = class_stats(primary)
            depth = stats.depth
            sibling_count = stats.sibling_count
            subclass_count = stats.subclass_count
            parent_count = stats.parent_count

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
                "answer_leak_risk":     False,
                "leak_type":            [],
                "definition_masked":    self.mask_instance,
            }
        }

    def generate_all_questions(self):
        questions, skipped = [], 0
        for inst in self.instances:
            if self.max_questions and len(questions) >= self.max_questions:
                break
            try:
                questions.append(self.generate_question_for_target(inst))
            except Exception:
                skipped += 1
        return questions, skipped


def save_questions(questions, save_path):
    save_json(questions, Path(save_path), description="questions")


def process_owl_file(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    load_imports: bool,
    onto_paths: Optional[List[Path]],
    suppress_warnings: bool,
    concept_scope: str = 'all',
    max_questions: Optional[int] = None,
) -> None:
    out_dir, safe_stem = build_mirrored_output_dir(file_path, input_root, output_root)
    save_path = out_dir / f"instance2definition_{safe_stem}.json"
    empty_path = empty_marker_path(out_dir, "instance2definition", safe_stem)
    if save_path.exists():
        logging.info("Skip existing: %s", save_path)
        return
    if empty_path.exists():
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

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
    gen = QuestionGenerator(insts, max_questions=max_questions, mask_instance=True)
    questions, skipped = gen.generate_all_questions()
    logging.info(f"Generated {len(questions)} questions (skipped {skipped}).")
    if questions:
        save_questions(questions, save_path)
    else:
        save_empty_marker(
            empty_path,
            source_file=file_path,
            reason="no_valid_instance_description_questions",
            extra={"instances": len(insts), "skipped": skipped},
        )


def main():
    parser = argparse.ArgumentParser(description='Generate instance-to-definition MCQs from ontologies (instances with explicit definitions).')
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

    configure_logging(args.log, "process_1_5.log")
    suppress_library_noise(args.no_warnings)
    random.seed(args.seed)
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
                    max_questions=args.max_questions or None,
                )
        except FileProcessingTimeout as e:
            logging.error("Timeout processing %s: %s", fp, e)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")


if __name__ == "__main__":
    main()
