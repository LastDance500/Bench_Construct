import json
import os
import random
import logging
import re
from rdflib import URIRef, Literal
from owlready2 import World, ThingClass, owl

definition_cache = {}
label_cache = {}

def get_definition(entity):
    key = str(entity.iri)
    if key in definition_cache:
        return definition_cache[key]

    defs = []
    # 1. IAO_0000115
    defs.extend(getattr(entity, "IAO_0000115", []) or [])
    # 2. skos:definition
    defs.extend(getattr(entity, "definition", []) or [])
    # 3. rdfs:comment
    cmts = getattr(entity, "comment", []) or []
    if isinstance(cmts, str):
        defs.append(cmts)
    else:
        defs.extend(cmts)
    # 4. skos:prefLabel
    defs.extend(getattr(entity, "prefLabel", []) or [])

    world_obj = getattr(entity, "world",
                        getattr(entity.namespace, "world", None))

    if world_obj:
        for ap in world_obj.annotation_properties():
            ap_local = str(ap.iri).split('#')[-1]
            if "definition" in ap_local.lower():
                vals = getattr(entity, ap.python_name, []) or []
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                defs.extend(vals)

        try:
            graph = world_obj.as_rdflib_graph()
            subj  = URIRef(entity.iri)
            for pred, obj in graph.predicate_objects(subj):
                pred_str = str(pred)
                if '#' in pred_str:
                    local = pred_str.split('#')[-1]
                else:
                    local = pred_str.rsplit('/', 1)[-1]
                if "definition" in local.lower():
                    if isinstance(obj, Literal):
                        defs.append(obj)
                    else:
                        defs.append(obj)
        except Exception as e:
            logging.warning(f"扫描 RDF 图提取 definition 时出错：{e}")

    if not defs:
        labs = getattr(entity, "label", []) or []
        defs.extend(labs)

    definition = None
    for d in defs:
        if getattr(d, "lang", None) == 'en':
            definition = str(d)
            break
    if not definition and defs:
        definition = str(defs[0])
    if not definition:
        definition = "No definition provided."

    definition_cache[key] = definition
    return definition

def get_label(entity):
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
    def __init__(self, file_path):
        self.file_path = file_path
        self.world     = World()
        self.onto      = None

    def load(self):
        for ont in (
            "http://purl.obolibrary.org/obo/iao.owl",
            "http://www.w3.org/2004/02/skos/core#"
        ):
            try:
                self.world.get_ontology(ont).load()
            except Exception as e:
                pass

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
        for cls in self.onto.classes():
            _ = getattr(cls, "IAO_0000115", None)
            _ = getattr(cls, "definition", None)
            _ = getattr(cls, "comment", None)
            _ = getattr(cls, "label", None)
            _ = getattr(cls, "prefLabel", None)

    def get_all_classes_with_definition(self):
        return [
            cls for cls in self.onto.classes()
            if cls != owl.Thing and get_definition(cls) != "No definition provided."
        ]

class QuestionGenerator:
    def __init__(self, classes):
        self.classes = classes

    def get_candidate_distractors(self, target):
        cand = set()
        for p in target.is_a:
            if isinstance(p, ThingClass) and p != owl.Thing:
                cand |= {
                    s for s in p.subclasses()
                    if s != target and get_definition(s) != "No definition provided."
                }
        cand |= {
            p for p in target.is_a
            if isinstance(p, ThingClass) and p != owl.Thing and get_definition(p) != "No definition provided."
        }
        cand |= {
            c for c in target.subclasses()
            if c != target and get_definition(c) != "No definition provided."
        }
        if len(cand) < 3:
            others = [c for c in self.classes if c != target]
            random.shuffle(others)
            for o in others:
                if o not in cand:
                    cand.add(o)
                if len(cand) >= 3:
                    break
        cand.discard(target)
        return list(cand)

    def generate_question_for_target(self, target):
        d   = compute_depth(target)
        s   = len(get_siblings(target))
        sub = len(list(target.subclasses()))
        par = len([p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing])

        defs = get_definition(target)
        lbl  = get_label(target)
        options = [{"label": lbl, "definition": defs, "is_correct": True}]
        distractors = random.sample(self.get_candidate_distractors(target), 3)
        for dsc in distractors:
            options.append({
                "label": get_label(dsc),
                "definition": get_definition(dsc),
                "is_correct": False
            })
        random.shuffle(options)

        letters = ['A', 'B', 'C', 'D']
        opts = []
        correct = None
        for i, o in enumerate(options):
            opts.append({
                "option_letter": letters[i],
                "definition": o["definition"]
            })
            if o["is_correct"]:
                correct = letters[i]

        return {
            "prompt": f"Which of the following definitions best describes '{lbl}'?",
            "options": opts,
            "correct_answer": correct,
            "meta": {
                "iri":            str(target.iri),
                "label":          lbl,
                "depth":          d,
                "sibling_count":  s,
                "subclass_count": sub,
                "parent_count":   par
            }
        }

    def generate_all_questions(self):
        if not self.classes:
            return [], 0
        gm = compute_global_metrics(self.classes)
        weights = [(e, compute_selection_weight(e, gm)) for e in self.classes]
        max_w = max(w for _, w in weights) or 1.0
        questions, skipped = [], 0
        for e, w in weights:
            if random.random() < (w / max_w):
                try:
                    questions.append(self.generate_question_for_target(e))
                except Exception:
                    skipped += 1
            else:
                skipped += 1
        return questions, skipped

def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

def process_owl_file(file_path):
    base     = os.path.splitext(os.path.basename(file_path))[0]
    out_dir  = os.path.dirname(file_path).replace("data", "bench/bench_1_1")
    save_path= os.path.join(out_dir, f"class2def_{base}.json")
    if os.path.exists(save_path):
        logging.info(f"Skip existing: {save_path}")
        return
    loader = OntologyLoader(file_path)
    onto   = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    loader.preload_entities()
    classes = loader.get_all_classes_with_definition()
    gen     = QuestionGenerator(classes)
    q, sk   = gen.generate_all_questions()
    logging.info(f"Generated {len(q)} questions (skipped {sk}).")
    if q:
        save_questions(q, save_path)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("process.log", "w", "utf-8")
        ]
    )
    random.seed(42)
    base_dir = "../../../data/Sciences/Computer Science"
    exts     = (".owl", ".rdf", ".rdfs", ".ttl")
    files    = []
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
