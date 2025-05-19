import json
import os
import random
import logging
from owlready2 import World, ThingClass, ObjectPropertyClass, Restriction, owl, sync_reasoner
from collections import OrderedDict

MAX_QUESTIONS = None
BASE_DIR = '../../../data'
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')
NUM_CHOICES = 4
MAX_CACHE_SIZE = 10000
MAX_CHAINS = 1000

label_cache = OrderedDict()

def get_label(entity):
    key = str(getattr(entity, 'iri', str(entity)))
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else getattr(entity, 'name', str(entity))
    label_cache[key] = label
    if len(label_cache) > MAX_CACHE_SIZE:
        label_cache.popitem(last=False)
    return label

def compute_depth(entity):
    if not isinstance(entity, ThingClass):
        return float('inf')
    queue = [(entity, 0)]
    visited = {entity}
    while queue:
        current, dist = queue[0]
        queue = queue[1:]
        if current == owl.Thing:
            return dist
        for parent in (current.is_a or []):
            if not isinstance(parent, ThingClass):
                continue
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, dist + 1))
    return float('inf')

def extract_property_info(onto):
    triples = []
    for prop in onto.object_properties():
        if not isinstance(prop, ObjectPropertyClass):
            continue
        domains = [d for d in (getattr(prop, 'domain', []) or []) if isinstance(d, ThingClass)]
        ranges = [r for r in (getattr(prop, 'range', []) or []) if isinstance(r, ThingClass)]
        if not domains or not ranges:
            continue
        for domain in domains:
            for range_cls in ranges:
                subclasses = [sc for sc in domain.subclasses() if isinstance(sc, ThingClass) and sc != domain]
                for subclass in subclasses:
                    triples.append((prop, domain, range_cls, subclass))
    logging.info(f"Extracted {len(triples)} property-domain-range-subclass triples")
    return triples

def extract_property_chain_info(onto):
    chains = []
    props = list(onto.object_properties())
    for p1 in props:
        if not isinstance(p1, ObjectPropertyClass):
            continue
        d1_list = [d for d in (getattr(p1, 'domain', []) or []) if isinstance(d, ThingClass)]
        r1_list = [r for r in (getattr(p1, 'range', []) or []) if isinstance(r, ThingClass)]
        for p2 in props:
            if not isinstance(p2, ObjectPropertyClass):
                continue
            d2_list = [d for d in (getattr(p2, 'domain', []) or []) if isinstance(d, ThingClass)]
            r2_list = [r for r in (getattr(p2, 'range', []) or []) if isinstance(r, ThingClass)]
            for d1 in d1_list:
                for r1 in r1_list:
                    for d2 in d2_list:
                        if r1 != d2:
                            continue
                        for r2 in r2_list:
                            subclasses = [sc for sc in d1.subclasses() if isinstance(sc, ThingClass) and sc != d1]
                            for subclass in subclasses:
                                chains.append((p1, p2, d1, r1, r2, subclass))
                                if len(chains) >= MAX_CHAINS:
                                    logging.info(f"Reached max chains limit: {MAX_CHAINS}")
                                    return chains
    logging.info(f"Extracted {len(chains)} property chains")
    return chains

def extract_existential_info(onto):
    existentials = []
    for cls in onto.classes():
        for constraint in getattr(cls, 'equivalent_to', []) + getattr(cls, 'is_a', []):
            if isinstance(constraint, Restriction) and constraint.type == owl.some:
                if isinstance(constraint.property, ObjectPropertyClass) and isinstance(constraint.value, ThingClass):
                    existentials.append((cls, constraint.property, constraint.value))
    logging.info(f"Extracted {len(existentials)} existential restrictions")
    return existentials

def get_distractors(correct, all_classes, exclude=None):
    candidates = [c for c in all_classes if c != correct and isinstance(c, ThingClass)]
    if exclude:
        candidates = [c for c in candidates if c not in exclude]
    candidates = sorted(candidates, key=lambda c: abs(compute_depth(c) - compute_depth(correct)))
    return random.sample(candidates[:10], min(NUM_CHOICES - 1, len(candidates)))

class RangeQuestionGenerator:
    def __init__(self, triples, all_classes):
        self.triples = triples
        self.all_classes = all_classes

    def generate_one(self, prop, domain, range_cls, subclass):
        distractors = get_distractors(range_cls, self.all_classes, exclude=[domain] + list(domain.subclasses()))
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [range_cls] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == range_cls)
        prompt = (f"Given that the property '{get_label(prop)}' has a domain of '{get_label(domain)}', "
                  f"and '{get_label(subclass)}' is a subclass of '{get_label(domain)}', "
                  f"what is the range of '{get_label(prop)}' when applied to '{get_label(subclass)}'?")
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'type': 'range',
                'property_iri': str(prop.iri),
                'domain_iri': str(domain.iri),
                'range_iri': str(range_cls.iri),
                'subclass_iri': str(subclass.iri),
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.triples)
        for prop, domain, range_cls, subclass in self.triples[:max_q or len(self.triples)]:
            q = self.generate_one(prop, domain, range_cls, subclass)
            if q:
                questions.append(q)
        return questions

class ChainQuestionGenerator:
    def __init__(self, chains, all_classes):
        self.chains = chains
        self.all_classes = all_classes

    def generate_one(self, p1, p2, d1, mid, final, subclass):
        distractors = get_distractors(final, self.all_classes)
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [final] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == final)
        prompt = (f"Given the property chain '{get_label(p1)} ∘ {get_label(p2)}': "
                  f"{get_label(p1)}: '{get_label(d1)}' → '{get_label(mid)}', "
                  f"{get_label(p2)}: '{get_label(mid)}' → '{get_label(final)}', "
                  f"what class does the chain point to when applied to '{get_label(subclass)}' (a subclass of '{get_label(d1)}')?")
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'type': 'chain',
                'property1_iri': str(p1.iri),
                'property2_iri': str(p2.iri),
                'domain_iri': str(d1.iri),
                'mid_iri': str(mid.iri),
                'final_iri': str(final.iri),
                'subclass_iri': str(subclass.iri),
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.chains)
        for p1, p2, d1, mid, final, subclass in self.chains[:max_q or len(self.chains)]:
            q = self.generate_one(p1, p2, d1, mid, final, subclass)
            if q:
                questions.append(q)
        return questions

class ExistentialQuestionGenerator:
    def __init__(self, existentials, all_classes):
        self.existentials = existentials
        self.all_classes = all_classes

    def generate_one(self, cls, prop, filler):
        distractors = get_distractors(filler, self.all_classes)
        if len(distractors) < NUM_CHOICES - 1:
            return None
        options = [filler] + distractors
        random.shuffle(options)
        letters = ['A', 'B', 'C', 'D']
        opts = [{'option_letter': l, 'label': get_label(c)} for l, c in zip(letters, options)]
        correct = next(l for l, c in zip(letters, options) if c == filler)
        prompt = (f"Given that the class '{get_label(cls)}' is defined as having some '{get_label(prop)}' "
                  f"pointing to instances of '{get_label(filler)}', "
                  f"what class must instances of '{get_label(cls)}' be related to via '{get_label(prop)}'?")
        return {
            'prompt': prompt,
            'options': opts,
            'correct_answer': correct,
            'meta': {
                'type': 'existential',
                'class_iri': str(cls.iri),
                'property_iri': str(prop.iri),
                'filler_iri': str(filler.iri),
            }
        }

    def generate_all(self, max_q=None):
        questions = []
        random.shuffle(self.existentials)
        for cls, prop, filler in self.existentials[:max_q or len(self.existentials)]:
            q = self.generate_one(cls, prop, filler)
            if q:
                questions.append(q)
        return questions

def save_questions(questions, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(questions)} questions to {save_path}")

def process_owl_file(file_path, max_q=None):
    world = World()
    onto = None
    try:
        onto = world.get_ontology(f"file://{os.path.abspath(file_path)}").load()
        if not onto:
            logging.error(f"Failed to load ontology {file_path}")
            return
        logging.info(f"Loaded ontology: {onto}")
    except Exception as e:
        logging.error(f"Failed to load ontology {file_path}: {e}")
        return

    for imp in onto.imported_ontologies or []:
        logging.info(f"Imported ontology: {imp}")
        try:
            imp.load()
        except Exception as e:
            logging.debug(f"Failed to load imported ontology {imp}: {e}")

    triples = extract_property_info(onto)
    chains = extract_property_chain_info(onto)
    existentials = extract_existential_info(onto)
    all_classes = [c for c in onto.classes() if isinstance(c, ThingClass)]
    if not all_classes:
        logging.info(f"No valid classes found for {file_path}")
        return

    q1 = RangeQuestionGenerator(triples, all_classes).generate_all(max_q)
    q2 = ChainQuestionGenerator(chains, all_classes).generate_all(max_q)
    q3 = ExistentialQuestionGenerator(existentials, all_classes).generate_all(max_q)
    questions = q1 + q2 + q3
    random.shuffle(questions)

    out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_2_2')
    save_path = os.path.join(out_dir, f'range_inference_{os.path.basename(file_path)}.json')
    save_questions(questions, save_path)

    try:
        world.close()
    except Exception as e:
        logging.debug(f"Failed to close World: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    random.seed(42)
    files = []
    for root, _, fnames in os.walk(BASE_DIR):
        for f in fnames:
            if f.lower().endswith(EXTENSIONS):
                files.append(os.path.join(root, f))
    for fp in files:
        try:
            process_owl_file(fp, MAX_QUESTIONS)
        except Exception as e:
            logging.error(f"{fp} failed: {e}", exc_info=True)
        finally:
            label_cache.clear()