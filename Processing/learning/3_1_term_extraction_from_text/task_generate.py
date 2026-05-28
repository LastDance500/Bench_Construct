from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from owlready2 import ThingClass, owl


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import (  # noqa: E402
    BaseOntologyLoader,
    build_mirrored_output_dir,
    class_stats,
    configure_logging,
    direct_parents,
    direct_subclasses,
    discover_ontology_files,
    get_definition,
    get_label,
    resolve_onto_paths,
    save_json,
    suppress_library_noise,
)


TASK_ID = "L1"
TASK_LABEL = "3_1"
TASK_NAME = "Ontology Term Extraction from Text"
GENERATION_METHOD = "template_verbalization"
PARAPHRASE_GENERATION_METHOD = "template_verbalization_with_llm_paraphrase"

GENERIC_CLASS_LABELS = {
    "thing",
    "entity",
    "object",
    "resource",
    "member",
    "concept",
}
GENERIC_PROPERTY_LABELS = {
    "label",
    "comment",
    "seealso",
    "see also",
    "isdefinedby",
    "is defined by",
    "related",
    "type",
}
DISTRACTORS = (
    "This description may appear in a domain model used for data organization.",
    "The surrounding workflow may involve records, processes, and administrative information.",
    "The example is intended to support consistent annotation and reuse.",
)
TERM_OVERLAP_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "of",
    "on",
    "or",
    "than",
    "the",
    "to",
    "with",
}

PARAPHRASE_ARTIFACT_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"\bcheck word count\b",
        r"\bword count\b",
        r"^\s*:\s*[\"“]",
        r"\bhere is (?:the|a)\b",
        r"\bi rewrote\b",
        r"\brewritten paragraph\b",
        r"\bparaphrased paragraph\b",
        r"\brevised paragraph\b",
        r"\brequired labels\b",
        r"\bdraft text\b",
        r"\banswer format\b",
        r"\brelation\.reuse\b",
        r"\bthis domain models use\b",
        r"\btype thereof\b",
    )
]

REPEATED_COORDINATION_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9() ,.'-]{2,60}?)\s+and\s+\1\b",
    re.I,
)


@dataclass(frozen=True)
class EntityInfo:
    entity: object
    iri: str
    label: str
    norm: str


@dataclass(frozen=True)
class PropertyInfo(EntityInfo):
    domains: tuple[ThingClass, ...]
    ranges: tuple[ThingClass, ...]
    property_type: str


def split_camel_case(text: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text


def clean_label(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        return None
    text = text.split("#")[-1].rsplit("/", 1)[-1]
    text = split_camel_case(text.replace("_", " ").replace("-", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_term(text: object) -> str:
    value = clean_label(text) or ""
    value = value.lower()
    value = re.sub(r"['\"`]", "", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def usable_label(label: Optional[str], generic_labels: set[str]) -> bool:
    if not label:
        return False
    norm = normalize_term(label)
    if len(norm) < 3:
        return False
    if norm in generic_labels:
        return False
    if norm.startswith("obsolete ") or " obsolete " in f" {norm} ":
        return False
    return True


def contains_normalized_label(text: str, label: str) -> bool:
    norm_text = f" {normalize_term(text)} "
    norm_label = normalize_term(label)
    if not norm_label:
        return False
    return f" {norm_label} " in norm_text


def text_quality_warnings(text: str) -> list[str]:
    warnings = []
    for pattern in PARAPHRASE_ARTIFACT_PATTERNS:
        if pattern.search(text):
            warnings.append(f"paraphrase artifact: {pattern.pattern}")
            break

    repeated = REPEATED_COORDINATION_PATTERN.search(text)
    if repeated:
        phrase = re.sub(r"\s+", " ", repeated.group(1)).strip()
        if len(phrase.split()) <= 8:
            warnings.append(f"repeated coordinated phrase: {phrase}")

    return warnings


def is_native(entity, ontology) -> bool:
    return getattr(getattr(entity, "namespace", None), "ontology", None) is ontology


def unique_entity_infos(entities: Iterable[object], generic_labels: set[str]) -> list[EntityInfo]:
    by_norm: dict[str, EntityInfo] = {}
    duplicates: set[str] = set()
    for entity in entities:
        if entity == owl.Thing:
            continue
        label = clean_label(get_label(entity))
        if not usable_label(label, generic_labels):
            continue
        norm = normalize_term(label)
        if norm in by_norm:
            duplicates.add(norm)
            continue
        by_norm[norm] = EntityInfo(entity=entity, iri=str(entity.iri), label=label, norm=norm)
    for norm in duplicates:
        by_norm.pop(norm, None)
    return list(by_norm.values())


def class_candidates(ontology, concept_scope: str) -> list[EntityInfo]:
    classes = list(ontology.classes())
    if concept_scope == "native":
        classes = [cls for cls in classes if is_native(cls, ontology)]
    elif concept_scope == "imported":
        classes = [cls for cls in classes if not is_native(cls, ontology)]
    return unique_entity_infos(classes, GENERIC_CLASS_LABELS)


def _as_class_list(values: Iterable[object], class_by_entity: dict[ThingClass, EntityInfo]) -> tuple[ThingClass, ...]:
    classes = []
    for value in values or []:
        if isinstance(value, ThingClass) and value in class_by_entity:
            classes.append(value)
    return tuple(classes)


def property_candidates(ontology, class_by_entity: dict[ThingClass, EntityInfo]) -> list[PropertyInfo]:
    properties = []
    for property_type, props in (
        ("object_property", list(ontology.object_properties())),
        ("data_property", list(ontology.data_properties())),
    ):
        for prop in props:
            label = clean_label(get_label(prop))
            if not usable_label(label, GENERIC_PROPERTY_LABELS):
                continue
            domains = _as_class_list(getattr(prop, "domain", []) or [], class_by_entity)
            ranges = _as_class_list(getattr(prop, "range", []) or [], class_by_entity)
            properties.append(
                PropertyInfo(
                    entity=prop,
                    iri=str(prop.iri),
                    label=label,
                    norm=normalize_term(label),
                    domains=domains,
                    ranges=ranges,
                    property_type=property_type,
                )
            )
    unique: dict[str, PropertyInfo] = {}
    duplicates: set[str] = set()
    for prop in properties:
        if prop.norm in unique:
            duplicates.add(prop.norm)
        else:
            unique[prop.norm] = prop
    for norm in duplicates:
        unique.pop(norm, None)
    return list(unique.values())


def definition_sentence(info: EntityInfo) -> Optional[str]:
    definition = get_definition(info.entity)
    if not definition or definition == "No definition provided.":
        return None
    definition = re.sub(r"\s+", " ", str(definition)).strip()
    if re.match(r"(?i)^(note|todo|fixme|editor note)\b", definition):
        return None
    if len(definition.split()) > 32:
        definition = " ".join(definition.split()[:32]).rstrip(" ,;") + "."
    elif definition[-1] not in ".!?":
        definition = definition.rstrip(" ,;") + "."
    return f"In this ontology, {info.label} refers to {definition}"


def hierarchy_sentence(child: EntityInfo, parent: EntityInfo) -> str:
    return random.choice(
        (
            f"{child.label} is a type of {parent.label}.",
            f"{child.label} is modeled as a subclass of {parent.label}.",
            f"The ontology organizes {child.label} under {parent.label}.",
        )
    )


def property_sentence(
    prop: PropertyInfo,
    class_by_entity: dict[ThingClass, EntityInfo],
    fallback_domain: EntityInfo,
    use_declared_domain: bool = True,
) -> Optional[str]:
    domain = None
    if use_declared_domain:
        domain = next((class_by_entity.get(cls) for cls in prop.domains if cls in class_by_entity), None)
    range_info = next((class_by_entity.get(cls) for cls in prop.ranges if cls in class_by_entity), None)
    domain = domain or fallback_domain
    if not use_declared_domain:
        return random.choice(
            (
                f"The same ontology fragment also includes the property term '{prop.label}'.",
                f"The relation term '{prop.label}' is included as a property in this ontology fragment.",
            )
        )
    if range_info:
        if normalize_term(domain.label) == normalize_term(range_info.label):
            return random.choice(
                (
                    f"The property '{prop.label}' is used as a relation for {domain.label}.",
                    f"The ontology uses the property '{prop.label}' when describing {domain.label}.",
                )
            )
        return random.choice(
            (
                f"The property '{prop.label}' connects {domain.label} with {range_info.label}.",
                f"{domain.label} is connected to {range_info.label} through the relation {prop.label}.",
            )
        )
    return random.choice(
        (
            f"The property '{prop.label}' is used as a relation for {domain.label}.",
            f"The ontology uses the property '{prop.label}' when describing {domain.label}.",
        )
    )


def content_tokens(label: str) -> set[str]:
    return {token for token in normalize_term(label).split() if token not in TERM_OVERLAP_STOP_TOKENS and len(token) > 2}


def property_overlaps_selected_classes(prop: PropertyInfo, selected_classes: Iterable[EntityInfo]) -> bool:
    prop_tokens = content_tokens(prop.label)
    if not prop_tokens:
        return False
    class_tokens = set()
    for info in selected_classes:
        class_tokens.update(content_tokens(info.label))
    return bool(prop_tokens & class_tokens)


def build_fragment(
    focal: EntityInfo,
    class_by_entity: dict[ThingClass, EntityInfo],
    properties: list[PropertyInfo],
    max_properties: int,
) -> Optional[dict]:
    focal_cls = focal.entity
    if not isinstance(focal_cls, ThingClass):
        return None

    selected_classes: dict[ThingClass, EntityInfo] = {focal_cls: focal}
    subclass_edges: list[tuple[str, str]] = []

    parents = [cls for cls in direct_parents(focal_cls, class_by_entity.keys()) if cls in class_by_entity]
    if parents:
        parent = random.choice(parents)
        selected_classes[parent] = class_by_entity[parent]
        subclass_edges.append((focal.iri, str(parent.iri)))

    children = [cls for cls in direct_subclasses(focal_cls, class_by_entity.keys()) if cls in class_by_entity]
    random.shuffle(children)
    for child in children[:2]:
        selected_classes[child] = class_by_entity[child]
        subclass_edges.append((str(child.iri), focal.iri))

    connected_props = [
        prop
        for prop in properties
        if focal_cls in prop.domains or focal_cls in prop.ranges
        or any(cls in selected_classes for cls in prop.domains + prop.ranges)
    ]
    random.shuffle(connected_props)
    selected_props: list[PropertyInfo] = []
    property_edges: list[tuple[str, str, str]] = []
    for prop in connected_props:
        if len(selected_props) >= max_properties:
            break
        domain = next((cls for cls in prop.domains if cls in class_by_entity), None) or focal_cls
        range_cls = next((cls for cls in prop.ranges if cls in class_by_entity), None)
        selected_classes[domain] = class_by_entity[domain]
        if range_cls:
            selected_classes[range_cls] = class_by_entity[range_cls]
            property_edges.append((str(domain.iri), prop.iri, str(range_cls.iri)))
        else:
            property_edges.append((str(domain.iri), prop.iri, "literal"))
        selected_props.append(prop)

    if len(selected_props) < max_properties:
        remaining_props = [prop for prop in properties if prop not in selected_props]
        random.shuffle(remaining_props)
        for prop in remaining_props:
            if not property_overlaps_selected_classes(prop, selected_classes.values()):
                continue
            selected_props.append(prop)
            property_edges.append((focal.iri, prop.iri, "unspecified"))
            if len(selected_props) >= max_properties:
                break

    if len(selected_classes) < 3 and parents:
        for parent in parents:
            selected_classes[parent] = class_by_entity[parent]
            if len(selected_classes) >= 3:
                break
    if len(selected_classes) < 3:
        return None

    ordered_classes = list(selected_classes.values())[:8]
    ordered_props = selected_props[:max_properties]

    sentences = []
    focal_def = definition_sentence(focal)
    if focal_def:
        sentences.append(focal_def)
    for child_iri, parent_iri in subclass_edges[:3]:
        child = next((info for info in ordered_classes if info.iri == child_iri), None)
        parent = next((info for info in ordered_classes if info.iri == parent_iri), None)
        if child and parent:
            sentences.append(hierarchy_sentence(child, parent))
    unspecified_property_iris = {edge[1] for edge in property_edges if edge[2] == "unspecified"}
    for prop in ordered_props:
        sentence = property_sentence(
            prop,
            class_by_entity,
            fallback_domain=focal,
            use_declared_domain=prop.iri not in unspecified_property_iris,
        )
        if sentence:
            sentences.append(sentence)
    sentences.append(random.choice(DISTRACTORS))

    text = " ".join(sentences)
    return {
        "text": text,
        "classes": ordered_classes,
        "properties": ordered_props,
        "subclass_edges": subclass_edges,
        "property_edges": property_edges,
    }


def build_label_index(labels: Iterable[str]) -> list[tuple[str, str]]:
    by_norm = {}
    for label in labels:
        norm = normalize_term(label)
        if norm:
            by_norm.setdefault(norm, label)
    return [(label, norm) for norm, label in by_norm.items()]


def diversity_key(info: EntityInfo, class_pool: set[ThingClass]) -> tuple[str, int]:
    parents = direct_parents(info.entity, class_pool)
    parent_label = min((normalize_term(get_label(parent)) for parent in parents), default="root")
    depth = class_stats(info.entity).depth or 0
    return parent_label or "root", depth


def diverse_class_order(classes: list[EntityInfo], class_pool: set[ThingClass]) -> list[EntityInfo]:
    """Round-robin by parent branch and depth so early samples cover more ontology regions."""
    buckets: dict[tuple[str, int], list[EntityInfo]] = defaultdict(list)
    for info in classes:
        parent_label, depth = diversity_key(info, class_pool)
        depth_band = min(depth // 2, 8)
        buckets[(parent_label, depth_band)].append(info)
    for bucket in buckets.values():
        random.shuffle(bucket)

    ordered = []
    keys = list(buckets)
    random.shuffle(keys)
    while keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return ordered


def validate_sample(
    sample: dict,
    all_labels: Iterable[str] | Iterable[tuple[str, str]],
    strict_ambiguity_check: bool,
) -> tuple[bool, list[str]]:
    warnings = []
    text = sample["text"]
    class_labels = [info.label for info in sample["classes"]]
    property_labels = [info.label for info in sample["properties"]]
    gold_labels = class_labels + property_labels

    quality_warnings = text_quality_warnings(text)
    if quality_warnings:
        return False, quality_warnings

    for label in gold_labels:
        if not contains_normalized_label(text, label):
            return False, [f"gold label missing from text: {label}"]
    if len({normalize_term(label) for label in gold_labels}) != len(gold_labels):
        return False, ["duplicate normalized gold label"]
    if len(class_labels) < 3:
        return False, ["too few class labels"]
    if len(gold_labels) > 10:
        return False, ["too many gold labels"]
    word_count = len(text.split())
    if word_count < 40 or word_count > 120:
        return False, [f"text length out of bounds: {word_count}"]

    gold_norms = {normalize_term(label) for label in gold_labels}
    ambiguous = ambiguous_labels(text, all_labels, gold_norms)
    if ambiguous:
        warnings.append(f"ambiguous ontology labels in text: {ambiguous[:5]}")
        if strict_ambiguity_check:
            return False, warnings
    return True, warnings


def ambiguous_labels(text: str, all_labels: Iterable[str] | Iterable[tuple[str, str]], gold_norms: set[str]) -> list[str]:
    norm_text = normalize_term(text)
    padded_text = f" {norm_text} "
    gold_spans = []
    for norm in gold_norms:
        if not norm:
            continue
        pattern = re.compile(rf"(?<!\S){re.escape(norm)}(?!\S)")
        gold_spans.extend(match.span() for match in pattern.finditer(norm_text))

    ambiguous = []
    for label_item in all_labels:
        if isinstance(label_item, tuple):
            label, norm_label = label_item
        else:
            label = label_item
            norm_label = normalize_term(label)
        if not norm_label or norm_label in gold_norms:
            continue
        if len(norm_label.split()) < 2:
            continue
        if f" {norm_label} " not in padded_text:
            continue
        pattern = re.compile(rf"(?<!\S){re.escape(norm_label)}(?!\S)")
        for match in pattern.finditer(norm_text):
            start, end = match.span()
            inside_gold = any(gold_start <= start and end <= gold_end for gold_start, gold_end in gold_spans)
            if not inside_gold:
                ambiguous.append(label)
                break
    return ambiguous


def paraphrase_with_openai(
    text: str,
    class_labels: list[str],
    property_labels: list[str],
    provider: str,
    model: str,
    timeout: float,
    validation_feedback: Optional[str] = None,
) -> Optional[str]:
    try:
        from openai import OpenAI
    except Exception as exc:
        logging.warning("OpenAI package unavailable, using template text: %s", exc)
        return None

    provider = provider.lower()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        default_headers = None
        key_name = "DEEPSEEK_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        default_headers = None
        key_name = "OPENAI_API_KEY"

    if not api_key:
        logging.warning("%s is not set, using template text", key_name)
        return None

    labels = class_labels + property_labels
    prompt = (
        "Rewrite the ontology-derived text into one natural, concise domain paragraph.\n"
        "Rules:\n"
        "1. Preserve every required ontology label exactly as written, including capitalization and spaces.\n"
        "2. Do not add new ontology terms, class names, or property names.\n"
        "3. Do not infer new domain/range facts beyond the draft.\n"
        "4. Aim for 55 to 90 words; it must be at least 40 words and at most 120 words.\n"
        "5. If the paragraph is too short, add one generic background sentence using only non-ontology words such as records, workflows, annotation, reuse, or data organization.\n"
        "6. For each required property label, state it as an ontology property or relation term in the paragraph; do not silently replace it with a paraphrase.\n"
        "7. Return only the rewritten paragraph, with no bullet points, headings, quotes, word counts, or explanation.\n\n"
        f"Required labels: {json.dumps(labels, ensure_ascii=False)}\n\n"
        f"Draft text:\n{text}"
    )
    if validation_feedback:
        prompt += f"\n\nPrevious attempt failed validation. Avoid this problem: {validation_feedback}"
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful ontology benchmark text editor. You preserve labels exactly and do not invent facts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        rewritten = response.choices[0].message.content or ""
    except Exception as exc:
        logging.warning("OpenAI paraphrase failed, using template text: %s", exc)
        return None

    rewritten = rewritten.strip().strip('"').strip("'")
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten or None


def maybe_paraphrase_fragment(
    fragment: dict,
    all_labels: Iterable[str],
    strict_ambiguity_check: bool,
    provider: str,
    model: str,
    timeout: float,
    retries: int,
    failure_policy: str,
) -> Optional[tuple[dict, dict]]:
    metadata = {
        "draft_text": fragment["text"],
        "final_text": fragment["text"],
        "paraphrase_provider": provider,
        "paraphrase_model": None,
        "paraphrase_applied": False,
        "paraphrase_attempts": 0,
        "paraphrase_validation_warnings": [],
    }
    if provider == "none":
        return fragment, metadata

    class_labels = [info.label for info in fragment["classes"]]
    property_labels = [info.label for info in fragment["properties"]]
    attempts = max(1, retries + 1)
    last_warnings: list[str] = []

    if provider not in {"deepseek", "openai"}:
        logging.warning("Unknown paraphrase provider '%s', using template text", provider)
        attempts = 0

    for attempt in range(attempts):
        rewritten = paraphrase_with_openai(
            fragment["text"],
            class_labels,
            property_labels,
            provider=provider,
            model=model,
            timeout=timeout,
            validation_feedback="; ".join(last_warnings) if last_warnings else None,
        )
        metadata["paraphrase_attempts"] = attempt + 1
        if not rewritten:
            last_warnings = ["empty paraphrase response"]
            continue

        candidate = dict(fragment)
        candidate["text"] = rewritten
        valid, warnings = validate_sample(candidate, all_labels, strict_ambiguity_check=True)
        if not valid:
            last_warnings = warnings
            logging.info("Rejected paraphrase attempt %d/%d: %s", attempt + 1, attempts, "; ".join(warnings))
            continue

        metadata.update(
            {
                "final_text": rewritten,
                "paraphrase_model": model,
                "paraphrase_applied": True,
                "paraphrase_validation_warnings": [],
            }
        )
        return candidate, metadata

    metadata["paraphrase_validation_warnings"] = last_warnings
    valid_draft, draft_warnings = validate_sample(fragment, all_labels, strict_ambiguity_check=True)
    if failure_policy == "fallback":
        return fragment, metadata
    if failure_policy == "clean_fallback" and valid_draft:
        metadata["paraphrase_validation_warnings"] = []
        return fragment, metadata
    logging.info(
        "Dropped fragment after paraphrase failure; policy=%s, last_warnings=%s, draft_warnings=%s",
        failure_policy,
        last_warnings,
        draft_warnings,
    )
    return None


def answer_text(classes: list[EntityInfo], properties: list[PropertyInfo]) -> str:
    class_part = "; ".join(info.label for info in classes)
    property_part = "; ".join(info.label for info in properties) if properties else "None"
    return f"Classes: {class_part}\nProperties: {property_part}"


def build_question(text: str) -> str:
    return (
        "## Ontology Term Extraction Task\n"
        "Given a short domain text, identify the ontology-relevant terms that should be modeled as classes "
        "and the terms that should be modeled as properties.\n\n"
        "### Text\n"
        f"{text}\n\n"
        "### Question\n"
        "Which terms in the text should be extracted as ontology classes, and which terms should be extracted "
        "as ontology properties?\n\n"
        "### Answer Format\n"
        "Classes: term1; term2; ...\n"
        "Properties: term1; term2; ..."
    )


def json_entity(info: EntityInfo) -> dict:
    return {"iri": info.iri, "label": info.label, "aliases": []}


def build_sample(
    ontology_id: str,
    domain: str,
    index: int,
    fragment: dict,
    generation_metadata: Optional[dict] = None,
) -> dict:
    focal = fragment["classes"][0]
    stats = class_stats(focal.entity)
    generation_metadata = generation_metadata or {}
    paraphrase_applied = bool(generation_metadata.get("paraphrase_applied", False))
    task_description = build_question(fragment["text"])
    return {
        "id": f"L1_{ontology_id}_{index:04d}",
        "task_id": TASK_ID,
        "task_label": TASK_LABEL,
        "capability": "Learning",
        "task_name": TASK_NAME,
        "ontology_id": ontology_id,
        "ontology_name": ontology_id,
        "domain": domain,
        "task_description": task_description,
        "question": task_description,
        "answer": answer_text(fragment["classes"], fragment["properties"]),
        "text": fragment["text"],
        "gold_classes": [json_entity(info) for info in fragment["classes"]],
        "gold_properties": [json_entity(info) for info in fragment["properties"]],
        "metadata": {
            "focal_class": focal.iri,
            "focal_class_label": focal.label,
            "depth": stats.depth,
            "sibling_count": stats.sibling_count,
            "subclass_count": stats.subclass_count,
            "parent_count": stats.parent_count,
            "source_fragment": {
                "classes": [info.iri for info in fragment["classes"]],
                "properties": [info.iri for info in fragment["properties"]],
                "subclass_edges": fragment["subclass_edges"],
                "property_edges": fragment["property_edges"],
            },
            "generation_method": PARAPHRASE_GENERATION_METHOD if paraphrase_applied else GENERATION_METHOD,
            "uses_llm": paraphrase_applied,
            "llm_role": "surface_paraphrase" if paraphrase_applied else None,
            **generation_metadata,
        },
    }


def write_csv(samples: list[dict], path: Path) -> None:
    fieldnames = [
        "id",
        "question",
        "definition",
        "options",
        "task_label",
        "task_id",
        "task_name",
        "domain",
        "label",
        "iri",
        "depth",
        "gold_classes",
        "gold_properties",
        "generation_method",
        "uses_llm",
        "paraphrase_model",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            meta = sample["metadata"]
            writer.writerow(
                {
                    "id": sample["id"],
                    "question": sample.get("task_description", sample.get("question", "")),
                    "definition": sample["answer"],
                    "options": "",
                    "task_label": TASK_LABEL,
                    "task_id": TASK_ID,
                    "task_name": TASK_NAME,
                    "domain": sample["domain"],
                    "label": meta["focal_class_label"],
                    "iri": meta["focal_class"],
                    "depth": meta["depth"],
                    "gold_classes": json.dumps(sample["gold_classes"], ensure_ascii=False),
                    "gold_properties": json.dumps(sample["gold_properties"], ensure_ascii=False),
                    "generation_method": meta.get("generation_method", GENERATION_METHOD),
                    "uses_llm": meta.get("uses_llm", False),
                    "paraphrase_model": meta.get("paraphrase_model", ""),
                }
            )


def process_ontology(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    load_imports: bool,
    onto_paths: Optional[list[Path]],
    concept_scope: str,
    max_questions: int,
    max_properties: int,
    strict_ambiguity_check: bool,
    paraphrase_provider: str,
    paraphrase_model: str,
    paraphrase_timeout: float,
    paraphrase_retries: int,
    paraphrase_failure_policy: str,
    paraphrase_workers: int,
    retry_empty: bool,
) -> None:
    out_dir, ontology_id = build_mirrored_output_dir(file_path, input_root, output_root)
    json_path = out_dir / f"term_extraction_{ontology_id}.json"
    csv_path = out_dir / f"term_extraction_{ontology_id}.csv"
    empty_path = out_dir / f"no_l1_samples_{ontology_id}.json"
    if json_path.exists() and csv_path.exists():
        logging.info("Skip existing: %s", json_path)
        return
    if empty_path.exists() and not retry_empty:
        logging.info("Skip empty ontology marker: %s", empty_path)
        return

    loader = BaseOntologyLoader(file_path, load_imports=load_imports, onto_paths=onto_paths)
    ontology = loader.load()
    if ontology is None:
        logging.error("Load failed: %s", file_path)
        return

    classes = class_candidates(ontology, concept_scope)
    class_by_entity = {info.entity: info for info in classes if isinstance(info.entity, ThingClass)}
    properties = property_candidates(ontology, class_by_entity)
    all_labels = [info.label for info in classes] + [prop.label for prop in properties]
    all_label_index = build_label_index(all_labels)
    logging.info("Found %d classes and %d properties for %s", len(classes), len(properties), file_path)

    classes = diverse_class_order(classes, set(class_by_entity))
    samples: list[dict] = []
    candidates: list[tuple[dict, list[str]]] = []
    warning_count = 0
    dropped_count = 0
    domain = "/".join(Path(file_path).relative_to(input_root).parts[:-1]) or file_path.parent.name

    for focal in classes:
        if len(candidates) >= max_questions:
            break
        fragment = build_fragment(focal, class_by_entity, properties, max_properties=max_properties)
        if not fragment:
            continue
        valid, validation_warnings = validate_sample(fragment, all_label_index, strict_ambiguity_check)
        if not valid:
            logging.debug("Rejected L1 fragment for %s: %s", focal.label, "; ".join(validation_warnings))
            continue
        candidates.append((fragment, validation_warnings))

    def paraphrase_candidate(candidate: tuple[dict, list[str]]) -> tuple[Optional[tuple[dict, dict]], list[str]]:
        fragment, validation_warnings = candidate
        return (
            maybe_paraphrase_fragment(
                fragment,
                all_label_index,
                strict_ambiguity_check=strict_ambiguity_check,
                provider=paraphrase_provider,
                model=paraphrase_model,
                timeout=paraphrase_timeout,
                retries=paraphrase_retries,
                failure_policy=paraphrase_failure_policy,
            ),
            validation_warnings,
        )

    workers = max(1, paraphrase_workers)
    if paraphrase_provider != "none" and workers > 1 and candidates:
        logging.info("Paraphrasing %d L1 fragments with %d workers", len(candidates), workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            paraphrase_results = list(executor.map(paraphrase_candidate, candidates))
    else:
        paraphrase_results = [paraphrase_candidate(candidate) for candidate in candidates]

    for paraphrased, validation_warnings in paraphrase_results:
        if paraphrased is None:
            dropped_count += 1
            continue
        fragment, generation_metadata = paraphrased
        if generation_metadata.get("paraphrase_validation_warnings"):
            warning_count += len(generation_metadata["paraphrase_validation_warnings"])
        elif paraphrase_provider == "none":
            warning_count += len(validation_warnings)
        sample = build_sample(ontology_id, domain, len(samples) + 1, fragment, generation_metadata)
        samples.append(sample)
        if len(samples) >= max_questions:
            break

    if not samples:
        logging.info("No L1 samples generated for %s", file_path)
        save_json(
            {
                "ontology_id": ontology_id,
                "source_file": str(file_path),
                "reason": "no_valid_l1_samples",
                "classes_found": len(classes),
                "properties_found": len(properties),
            },
            empty_path,
            description="empty L1 term extraction marker",
        )
        return
    save_json(samples, json_path, description="L1 term extraction samples")
    write_csv(samples, csv_path)
    logging.info(
        "Saved %d L1 samples to %s (%d validation warnings, %d dropped after paraphrase validation)",
        len(samples),
        out_dir,
        warning_count,
        dropped_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L1 ontology term extraction samples from ontology fragments.")
    parser.add_argument("--input", type=str, required=True, help="Input ontology file or directory.")
    parser.add_argument("--output", type=str, required=True, help="Output root directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-questions", type=int, default=100, help="Max samples per ontology.")
    parser.add_argument("--max-properties", type=int, default=3, help="Max properties per sample.")
    parser.add_argument("--no-imports", action="store_true", help="Do not load imports.")
    parser.add_argument("--onto-path", action="append", default=None, help="Local directories to resolve owl:imports.")
    parser.add_argument("--concept-scope", choices=["all", "native", "imported"], default="native")
    parser.add_argument("--strict-ambiguity-check", action="store_true", help="Reject text containing non-gold ontology labels.")
    parser.add_argument("--retry-empty", action="store_true", help="Retry ontologies previously marked as having no valid L1 samples.")
    parser.add_argument(
        "--paraphrase-provider",
        choices=["none", "deepseek", "openai"],
        default="none",
        help="Optional surface paraphrase provider.",
    )
    parser.add_argument(
        "--paraphrase-model",
        default=os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL", "deepseek-chat"),
        help="Model used for surface paraphrasing.",
    )
    parser.add_argument("--paraphrase-timeout", type=float, default=30.0, help="OpenAI request timeout in seconds.")
    parser.add_argument("--paraphrase-retries", type=int, default=1, help="Retries for failed paraphrase validation.")
    parser.add_argument(
        "--paraphrase-workers",
        type=int,
        default=int(os.getenv("PARAPHRASE_WORKERS", "1")),
        help="Number of concurrent paraphrase requests per ontology.",
    )
    parser.add_argument(
        "--paraphrase-failure-policy",
        choices=["clean_fallback", "fallback", "drop"],
        default="clean_fallback",
        help="What to do after paraphrase retries fail. clean_fallback keeps only warning-free template text.",
    )
    parser.add_argument("--no-warnings", action="store_true", help="Suppress warnings and library noise.")
    parser.add_argument("--log", type=str, default="info", help="Logging level.")
    args = parser.parse_args()

    configure_logging(args.log, "process_3_1.log")
    suppress_library_noise(args.no_warnings)
    random.seed(args.seed)

    files, input_root = discover_ontology_files(Path(args.input))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    onto_paths = resolve_onto_paths(args.onto_path)
    logging.info("Found %d ontology files.", len(files))
    for file_path in files:
        try:
            process_ontology(
                file_path=file_path,
                input_root=input_root,
                output_root=output_root,
                load_imports=not args.no_imports,
                onto_paths=onto_paths,
                concept_scope=args.concept_scope,
                max_questions=args.max_questions,
                max_properties=args.max_properties,
                strict_ambiguity_check=args.strict_ambiguity_check,
                paraphrase_provider=args.paraphrase_provider,
                paraphrase_model=args.paraphrase_model,
                paraphrase_timeout=args.paraphrase_timeout,
                paraphrase_retries=args.paraphrase_retries,
                paraphrase_failure_policy=args.paraphrase_failure_policy,
                paraphrase_workers=args.paraphrase_workers,
                retry_empty=args.retry_empty,
            )
        except Exception as exc:
            logging.error("%s failed: %s", file_path, exc)


if __name__ == "__main__":
    main()
