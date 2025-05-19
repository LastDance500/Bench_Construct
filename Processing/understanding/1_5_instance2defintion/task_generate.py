import json
import os
import random
import logging
from rdflib import URIRef, Literal
from owlready2 import World, ThingClass

instance_definition_cache = {}
instance_label_cache = {}


def has_annotation_definition(entity):
    annos = []

    annos.extend(getattr(entity, "IAO_0000115", []) or [])
    annos.extend(getattr(entity, "definition", []) or [])
    cmts = getattr(entity, "comment", []) or []
    if isinstance(cmts, str):
        annos.append(cmts)
    else:
        annos.extend(cmts)

    world_obj = getattr(entity, "world",
                        getattr(entity.namespace, "world", None))
    if world_obj:
        for ap in world_obj.annotation_properties():
            ap_local = str(ap.iri).split('#')[-1].lower()
            if "definition" in ap_local or "comment" in ap_local:
                vals = getattr(entity, ap.python_name, []) or []
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                annos.extend(vals)

        try:
            graph = world_obj.as_rdflib_graph()
            subj = URIRef(entity.iri)
            for pred, obj in graph.predicate_objects(subj):
                local = str(pred).split('#')[-1].lower().rsplit('/', 1)[-1]
                if ("definition" in local or "comment" in local) and isinstance(obj, Literal):
                    annos.append(obj)
        except Exception as e:
            logging.warning(f"error：{e}")

    return len(annos) > 0


def get_definition(entity):
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

        try:
            graph = world_obj.as_rdflib_graph()
            subj = URIRef(entity.iri)
            for pred, obj in graph.predicate_objects(subj):
                local = str(pred).split('#')[-1].lower().rsplit('/', 1)[-1]
                if ("definition" in local or "comment" in local) and isinstance(obj, Literal):
                    defs.append(obj)
        except Exception as e:
            pass

    if not defs:
        definition = "No definition provided."
    else:
        definition = next((str(d) for d in defs if getattr(d, "lang", None) == 'en'), None)
        if not definition:
            definition = str(defs[0])

    instance_definition_cache[key] = definition
    return definition


def get_label(entity):
    key = str(entity.iri)
    if key in instance_label_cache:
        return instance_label_cache[key]
    labs = getattr(entity, "label", []) or getattr(entity, "prefLabel", [])
    label = labs[0] if labs else entity.name
    instance_label_cache[key] = label
    return label


class OntologyLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.world = World()
        self.onto = None

    def load(self):
        for ont in (
            "http://purl.obolibrary.org/obo/iao.owl",
            "http://www.w3.org/2004/02/skos/core#"
        ):
            try:
                self.world.get_ontology(ont).load()
            except Exception as e:
                logging.warning(f"load {ont} failed：{e}")
        iri = f"file://{os.path.abspath(self.file_path)}"
        onto = self.world.get_ontology(iri)
        try:
            onto.load()
        except Exception:
            logging.warning(f"local-only：{self.file_path}")
            onto.load(only_local=True)
        self.onto = onto
        return onto

    def preload_entities(self):
        for inst in self.onto.individuals():
            _ = getattr(inst, "IAO_0000115", None)
            _ = getattr(inst, "definition", None)
            _ = getattr(inst, "comment", None)
            _ = getattr(inst, "label", None)
            _ = getattr(inst, "prefLabel", None)

    def get_all_instances_with_definition(self):
        return [
            inst for inst in self.onto.individuals()
            if has_annotation_definition(inst)
        ]


class QuestionGenerator:
    def __init__(self, instances):
        self.instances = instances

    def get_candidate_distractors(self, target):
        cand = set()
        for cls in target.is_a:
            if isinstance(cls, ThingClass):
                cand |= {
                    i for i in cls.instances()
                    if i != target and has_annotation_definition(i)
                }
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
        definition = get_definition(target)
        label = get_label(target)

        options = [{"definition": definition, "is_correct": True}]
        distractors = random.sample(self.get_candidate_distractors(target), 3)
        for d in distractors:
            options.append({
                "definition": get_definition(d),
                "is_correct": False
            })

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

        return {
            "prompt": f"Which of the following definitions best describes the instance '{label}'?",
            "options": opts,
            "correct_answer": correct,
            "meta": {
                "iri": str(target.iri),
                "label": label,
                "types": [
                    str(c.iri)
                    for c in target.is_a
                    if isinstance(c, ThingClass)
                ]
            }
        }

    def generate_all_questions(self):
        questions, skipped = [], 0
        for inst in self.instances:
            try:
                questions.append(self.generate_question_for_target(inst))
            except Exception as e:
                skipped += 1
        return questions, skipped


def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


def process_owl_file(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.dirname(file_path).replace("data", "bench/bench_1_5")
    save_path = os.path.join(out_dir, f"instance2definition_{base}.json")
    if os.path.exists(save_path):
        logging.info(f"Skip existing: {save_path}")
        return
    loader = OntologyLoader(file_path)
    onto = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    loader.preload_entities()
    insts = loader.get_all_instances_with_definition()
    gen = QuestionGenerator(insts)
    questions, skipped = gen.generate_all_questions()
    logging.info(f"Generated {len(questions)} questions (skipped {skipped}).")
    if questions:
        save_questions(questions, save_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("process_instance2definition.log", "w", "utf-8")
        ]
    )
    random.seed(42)
    base_dir = "../../../data"
    exts = (".owl", ".rdf", ".rdfs", ".ttl")
    files = []
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
