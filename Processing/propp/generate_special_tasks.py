import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from owlready2 import ThingClass, World, owl


PROCESSING_ROOT = Path(__file__).resolve().parents[1]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import configure_logging, configure_world_paths, get_definition, get_label, load_ontology, save_json


DOMAIN = "arts_media_entertainment/propp_ontology"
PROPP_EXCLUDED_LABELS = {
    "eTrap motif",
    "eTRAP motif",
    "eTRAP added motif",
    "Linking from AaTh-Numbers to ATU-Numbers",
    "Linking back to the ATU source",
}

FUNCTION_RULES = [
    (
        "W",
        "Wedding Resolution",
        "The tale ends with a wedding or marriage-related resolution.",
        [
            "The tale ends with the villain's first attack.",
            "The tale centers on a donor testing the hero but has no resolution.",
            "The tale ends only with the hero's departure from home.",
        ],
    ),
    (
        "β",
        "Absentation",
        "The tale includes an absentation or a family member leaving the household.",
        [
            "The tale begins with a wedding resolution.",
            "The tale contains only magical combat and no family disruption.",
            "The tale explicitly ends with the hero's punishment.",
        ],
    ),
    (
        "↑",
        "Departure",
        "The hero or another key figure departs on a journey.",
        [
            "The tale contains no movement away from home.",
            "The tale is limited to a wedding feast at the start.",
            "The tale is only about a magical helper being tested.",
        ],
    ),
    (
        "↓",
        "Return",
        "Someone returns after a journey or quest.",
        [
            "No one returns after leaving.",
            "The tale contains only an initial lack with no later resolution.",
            "The plot never moves beyond the home setting.",
        ],
    ),
    (
        "A",
        "Villainy or Lack",
        "The sequence explicitly contains Propp's A function, marking villainy or lack.",
        [
            "The sequence only describes the final wedding.",
            "The sequence contains only an interdiction and no A function.",
            "The sequence begins with the hero's triumphant return and nothing else.",
        ],
    ),
    (
        "C",
        "Interdiction",
        "The sequence contains an interdiction: someone is warned, forbidden, or instructed not to do something.",
        [
            "The sequence only marks the final wedding resolution.",
            "The sequence contains no warning, prohibition, or instruction.",
            "The sequence only marks the hero's return after the quest.",
        ],
    ),
    (
        "H",
        "Struggle",
        "The tale includes a struggle or confrontation.",
        [
            "The tale contains no direct confrontation.",
            "The sequence is only a list of kinship relations.",
            "The tale ends before any conflict starts.",
        ],
    ),
    (
        "I",
        "Victory",
        "The conflict ends with the villain or opponent being defeated.",
        [
            "The opponent is never defeated.",
            "The tale stops before the conflict is resolved.",
            "The sequence only indicates a departure from home.",
        ],
    ),
]

VERBALISATION_PROMPTS = [
    "An instance in a Proppian tale is verbalised as: '{verbalisation}'. Which class best matches this evidence?",
    "The evidence for a tale element says: '{verbalisation}'. Which class does this verbalisation support?",
    "Given the verbalisation '{verbalisation}', which ontology class is the most specific match?",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def clean_literal(value) -> str:
    text = normalize_whitespace(value)
    return text.strip('"')


def valid_label(text: str) -> bool:
    text = clean_literal(text)
    if not text or text == "Unnamed":
        return False
    normalized = text.lower()
    return not any(term.lower() in normalized for term in PROPP_EXCLUDED_LABELS)


def is_mostly_non_latin(text: str) -> bool:
    chars = [ch for ch in str(text or "") if ch.isalpha()]
    if not chars:
        return False
    latin = sum(1 for ch in chars if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    return latin / len(chars) < 0.45


def is_quality_verbalisation(text: str) -> bool:
    text = clean_literal(text)
    words = re.findall(r"[A-Za-z][A-Za-z'’.-]*", text)
    return (
        20 <= len(text) <= 240
        and len(words) >= 5
        and not is_mostly_non_latin(text)
    )


def compute_depth(entity, memo=None):
    if memo is None:
        memo = {}
    if entity in memo:
        return memo[entity]
    if entity == owl.Thing:
        memo[entity] = 0
        return 0
    parents = [p for p in getattr(entity, "is_a", []) if isinstance(p, ThingClass) and p != owl.Thing]
    depth = 1 if not parents else max(compute_depth(p, memo) for p in parents) + 1
    memo[entity] = depth
    return depth


def render_options(option_texts):
    letters = ["A", "B", "C", "D"]
    return [{"option_letter": letter, "label": text} for letter, text in zip(letters, option_texts)]


def append_csv_row(rows, question, option_rows, answer, task_label, label, iri, depth):
    rows.append(
        {
            "question": question,
            "options": "\n\n".join(f"{opt['option_letter']}. {opt['label']}" for opt in option_rows),
            "answer": answer,
            "task_label": task_label,
            "label": label,
            "iri": iri,
            "depth": depth,
            "domain": DOMAIN,
        }
    )


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "options", "answer", "task_label", "label", "iri", "depth", "domain"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_verbalisation_text(inst) -> str | None:
    for attr in ("Verbalisation", "verbalisation"):
        values = getattr(inst, attr, None)
        if not values:
            continue
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            text = clean_literal(value)
            if text:
                return text
    return None


def get_function_sequence_text(inst) -> str | None:
    for attr in ("FunctionSequence", "functionSequence"):
        value = getattr(inst, attr, None)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        text = clean_literal(value)
        if text:
            return text
    return None


def choose_specific_type(inst):
    candidates = [
        cls
        for cls in getattr(inst, "is_a", [])
        if isinstance(cls, ThingClass) and cls != owl.Thing and valid_label(get_label(cls))
    ]
    candidates = [cls for cls in candidates if compute_depth(cls) >= 2]
    if not candidates:
        return None
    return sorted(candidates, key=lambda cls: (compute_depth(cls), len(get_label(cls))), reverse=True)[0]


def verbalisation_distractors(target_cls, all_classes):
    siblings = []
    for parent in getattr(target_cls, "is_a", []):
        if not isinstance(parent, ThingClass):
            continue
        for sibling in parent.subclasses():
            if sibling == target_cls or not valid_label(get_label(sibling)):
                continue
            siblings.append(sibling)
    random.shuffle(siblings)
    selected = []
    used_labels = {get_label(target_cls)}
    for candidate in siblings + all_classes:
        if candidate == target_cls or candidate in selected:
            continue
        label = get_label(candidate)
        if not valid_label(label) or label in used_labels:
            continue
        selected.append(candidate)
        used_labels.add(label)
        if len(selected) >= 3:
            break
    return selected if len(selected) == 3 else None


def verbalisation_text_distractors(target_inst, target_cls, typed_rows):
    candidates = [
        (inst, cls, text)
        for inst, cls, text in typed_rows
        if inst != target_inst and cls != target_cls and is_quality_verbalisation(text)
    ]
    random.shuffle(candidates)
    selected = []
    used_texts = {get_verbalisation_text(target_inst)}
    for _, _, text in candidates:
        if not text or text in used_texts:
            continue
        selected.append(text)
        used_texts.add(text)
        if len(selected) >= 3:
            break
    return selected if len(selected) == 3 else None


def build_verbalisation_rows(onto):
    all_classes = [
        cls for cls in onto.classes()
        if isinstance(cls, ThingClass) and cls != owl.Thing and valid_label(get_label(cls))
    ]
    questions = []
    csv_rows = []
    typed_rows = []
    for inst in onto.individuals():
        verbalisation = get_verbalisation_text(inst)
        if not verbalisation:
            continue
        if not is_quality_verbalisation(verbalisation):
            continue
        target_cls = choose_specific_type(inst)
        if target_cls is None:
            continue
        typed_rows.append((inst, target_cls, verbalisation))

    for inst, target_cls, verbalisation in typed_rows:
        distractors = verbalisation_distractors(target_cls, all_classes)
        if distractors is None:
            continue
        options = [target_cls] + distractors
        random.shuffle(options)
        correct = "ABCD"[options.index(target_cls)]
        prompt = random.choice(VERBALISATION_PROMPTS).format(verbalisation=verbalisation)
        option_rows = [{"option_letter": "ABCD"[idx], "label": get_label(cls)} for idx, cls in enumerate(options)]
        questions.append(
            {
                "prompt": prompt,
                "options": option_rows,
                "correct_answer": correct,
                "meta": {
                    "subject_iri": str(inst.iri),
                    "subject_label": get_label(inst),
                    "subject_kind": "instance",
                    "relation": "verbalisation_evidence",
                    "object_iri": str(target_cls.iri),
                    "object_label": get_label(target_cls),
                    "object_kind": "class",
                    "class_context_iri": str(target_cls.iri),
                    "class_context_label": get_label(target_cls),
                    "depth": compute_depth(target_cls),
                    "verbalisation": verbalisation,
                },
            }
        )
        append_csv_row(
            csv_rows,
            prompt,
            option_rows,
            correct,
            "propp_v1",
            get_label(target_cls),
            str(inst.iri),
            compute_depth(target_cls),
        )

        text_distractors = verbalisation_text_distractors(inst, target_cls, typed_rows)
        if text_distractors is None:
            continue
        text_options = [verbalisation] + text_distractors
        random.shuffle(text_options)
        text_correct = "ABCD"[text_options.index(verbalisation)]
        text_prompt = f"Which verbalisation provides evidence that an instance belongs to '{get_label(target_cls)}'?"
        text_option_rows = render_options(text_options)
        questions.append(
            {
                "prompt": text_prompt,
                "options": text_option_rows,
                "correct_answer": text_correct,
                "meta": {
                    "subject_iri": str(target_cls.iri),
                    "subject_label": get_label(target_cls),
                    "subject_kind": "class",
                    "relation": "verbalisation_evidence_reverse",
                    "object_iri": str(inst.iri),
                    "object_label": get_label(inst),
                    "object_kind": "instance",
                    "class_context_iri": str(target_cls.iri),
                    "class_context_label": get_label(target_cls),
                    "depth": compute_depth(target_cls),
                    "verbalisation": verbalisation,
                },
            }
        )
        append_csv_row(
            csv_rows,
            text_prompt,
            text_option_rows,
            text_correct,
            "propp_v1_reverse",
            get_label(target_cls),
            str(inst.iri),
            compute_depth(target_cls),
        )
    return questions, csv_rows


def sequence_rules(sequence: str):
    matched = []
    for symbol, title, correct, distractors in FUNCTION_RULES:
        if symbol in sequence:
            matched.append((symbol, title, correct, distractors))
    return matched


def function_option_label(symbol: str, title: str) -> str:
    return symbol


def sequence_rule_distractors(correct_symbol: str):
    candidates = [
        function_option_label(symbol, title)
        for symbol, title, _, _ in FUNCTION_RULES
        if symbol != correct_symbol
    ]
    random.shuffle(candidates)
    return candidates[:3]


def build_function_sequence_rows(onto):
    questions = []
    csv_rows = []
    for inst in onto.individuals():
        sequence = get_function_sequence_text(inst)
        if not sequence:
            continue
        rules = sequence_rules(sequence)
        if not rules:
            continue
        for symbol, title, correct_text, distractors in rules:
            choices = [correct_text] + distractors[:3]
            random.shuffle(choices)
            correct = "ABCD"[choices.index(correct_text)]
            prompt = (
                f"The tale '{get_label(inst)}' has the Proppian function sequence '{sequence}'. "
                f"Which statement is supported by this sequence?"
            )
            option_rows = render_options(choices)
            questions.append(
                {
                    "prompt": prompt,
                    "options": option_rows,
                    "correct_answer": correct,
                    "meta": {
                        "subject_iri": str(inst.iri),
                        "subject_label": get_label(inst),
                        "subject_kind": "instance",
                        "relation": "function_sequence_reasoning",
                        "object_iri": None,
                        "object_label": title,
                        "object_kind": "text",
                        "class_context_iri": None,
                        "class_context_label": title,
                        "depth": None,
                        "function_sequence": sequence,
                        "trigger_symbol": symbol,
                    },
                }
            )
            csv_rows.append(
                {
                    "question": prompt,
                    "options": "\n\n".join(f"{opt['option_letter']}. {opt['label']}" for opt in option_rows),
                    "answer": correct,
                    "task_label": "propp_v2",
                    "label": title,
                    "iri": str(inst.iri),
                    "depth": "",
                    "domain": DOMAIN,
                }
            )

            reverse_correct_text = function_option_label(symbol, title)
            reverse_choices = [reverse_correct_text] + sequence_rule_distractors(symbol)
            if len(reverse_choices) < 4:
                continue
            random.shuffle(reverse_choices)
            reverse_correct = "ABCD"[reverse_choices.index(reverse_correct_text)]
            reverse_prompt = (
                f"The tale '{get_label(inst)}' has the Proppian function sequence '{sequence}'. "
                f"Which function symbol in the sequence supports this statement: {correct_text}"
            )
            reverse_option_rows = render_options(reverse_choices)
            questions.append(
                {
                    "prompt": reverse_prompt,
                    "options": reverse_option_rows,
                    "correct_answer": reverse_correct,
                    "meta": {
                        "subject_iri": str(inst.iri),
                        "subject_label": get_label(inst),
                        "subject_kind": "instance",
                        "relation": "function_sequence_reverse_reasoning",
                        "object_iri": None,
                        "object_label": title,
                        "object_kind": "text",
                        "class_context_iri": None,
                        "class_context_label": title,
                        "depth": None,
                        "function_sequence": sequence,
                        "trigger_symbol": symbol,
                    },
                }
            )
            csv_rows.append(
                {
                    "question": reverse_prompt,
                    "options": "\n\n".join(f"{opt['option_letter']}. {opt['label']}" for opt in reverse_option_rows),
                    "answer": reverse_correct,
                    "task_label": "propp_v2_reverse",
                    "label": title,
                    "iri": str(inst.iri),
                    "depth": "",
                    "domain": DOMAIN,
                }
            )
    return questions, csv_rows


def main():
    parser = argparse.ArgumentParser(description="Generate Propp-specific verbalisation and function-sequence tasks.")
    parser.add_argument("--input", required=True, help="Input Propp ontology file.")
    parser.add_argument("--output", required=True, help="Output directory for JSON and CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log", default="info", help="Logging level.")
    args = parser.parse_args()

    configure_logging(args.log)
    random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    world = World()
    configure_world_paths(world, None)
    onto = load_ontology(world, Path(args.input), load_imports=False)
    if onto is None:
        raise RuntimeError(f"Failed to load ontology: {args.input}")

    verbalisation_questions, verbalisation_rows = build_verbalisation_rows(onto)
    sequence_questions, sequence_rows = build_function_sequence_rows(onto)

    save_json(verbalisation_questions, output_dir / "propp_verbalisation_evidence.json", description="Propp verbalisation tasks")
    save_json(sequence_questions, output_dir / "propp_function_sequence_reasoning.json", description="Propp sequence tasks")
    write_csv(verbalisation_rows, output_dir / "propp_verbalisation_evidence.csv")
    write_csv(sequence_rows, output_dir / "propp_function_sequence_reasoning.csv")


if __name__ == "__main__":
    main()
