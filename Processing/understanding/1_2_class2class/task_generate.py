import os
import json
import random
import logging
import re
from rdflib import URIRef, RDF
from owlready2 import World, ThingClass, Restriction, owl
from collections import defaultdict
from itertools import islice

MAX_QUESTIONS = 30000
BASE_DIR = '../../../data'
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')

label_cache = {}


def get_label(entity):
    key = str(entity.iri)
    if key not in label_cache:
        labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
        label_cache[key] = labs[0] if labs else entity.name
    return label_cache[key]


# 将驼峰和下划线转为空格
def humanize_relation(rel_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', rel_name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.replace('_', ' ').lower()


def make_prompt(subj_label, rel_name):
    templates = {
        'subclassOf': f"Which of the following classes is the superclass of '{subj_label}'?",
        'equivalentTo': f"Which of the following classes is equivalent to '{subj_label}'?",
        'disjointWith': f"Which of the following classes is disjoint with '{subj_label}'?",
        'complementOf': f"Which of the following classes is the complement of '{subj_label}'?",
        'unionOf': f"Which of the following classes is a member of the union defining '{subj_label}'?",
        'intersectionOf': f"Which of the following classes is a member of the intersection defining '{subj_label}'?"
    }
    return templates.get(rel_name, f"Which of the following classes {humanize_relation(rel_name)} '{subj_label}'?")


class RelationQuestionGenerator:
    def __init__(self, triples, all_classes, metadata):
        self.triples = triples
        self.all_classes = all_classes
        self.meta = metadata
        self.disjoint_sets = defaultdict(set)
        for c in all_classes:
            for dis in getattr(c, 'disjoint_with', []):
                if isinstance(dis, ThingClass):
                    self.disjoint_sets[c].add(dis)

    def _get_distractors(self, obj, k):
        obj_anc = self.meta[obj]['ancestors']
        distractors = []
        candidates = list(self.disjoint_sets[obj])
        random.shuffle(candidates)
        for c in candidates:
            if c is not obj and len(distractors) < k:
                distractors.append(c)

        if len(distractors) < k:
            remaining = [c for c in self.all_classes if c is not obj and c not in obj_anc and c not in distractors]
            random.shuffle(remaining)
            distractors.extend(remaining[:k - len(distractors)])

        return distractors[:k]

    def generate_all(self, max_q=MAX_QUESTIONS):
        questions = []
        letters = ['A', 'B', 'C', 'D']
        for subj, rel, obj in islice(self.triples, max_q):
            m = self.meta[subj]
            distractors = self._get_distractors(obj, 3)
            options = [obj] + distractors
            random.shuffle(options)
            opts = []
            correct = None
            for i, choice in enumerate(options):
                opts.append({'option_letter': letters[i], 'label': get_label(choice)})
                if choice is obj:
                    correct = letters[i]
            prompt = make_prompt(get_label(subj), rel)
            questions.append({
                'prompt': prompt,
                'options': opts,
                'correct_answer': correct,
                'meta': {
                    'iri': str(subj.iri),
                    'label': get_label(subj),
                    'depth': m['depth'],
                    'sibling_count': m['siblings'],
                    'subclass_count': m['subclasses'],
                    'parent_count': m['parents'],
                    'relation': rel,
                    'object_iri': str(obj.iri)
                }
            })
            if len(questions) % 1000 == 0:
                logging.info(f"Generated {len(questions)} questions")
            if len(questions) >= max_q:
                break
        return questions

def compute_ancestors(parent_map, all_classes):
    ancestors_map = {c: set() for c in all_classes}

    def get_ancestors(c):
        if c in ancestors_map and ancestors_map[c]:
            return ancestors_map[c]
        anc = set()
        for p in parent_map[c]:
            anc.add(p)
            anc.update(get_ancestors(p))
        ancestors_map[c] = anc
        return anc

    for c in all_classes:
        get_ancestors(c)
    return ancestors_map


def extract_and_prepare(onto):
    all_classes = list(onto.classes())
    logging.info(f"Found {len(all_classes)} classes")

    parent_map = {}
    for c in all_classes:
        try:
            parent_map[c] = [p for p in c.is_a if isinstance(p, ThingClass) and p != owl.Thing]
        except Exception as e:
            logging.warning(f"Skipping parents for {c}: {e}")
            parent_map[c] = []

    child_map = defaultdict(list)
    for c, parents in parent_map.items():
        for p in parents:
            child_map[p].append(c)

    ancestors_map = compute_ancestors(parent_map, all_classes)

    metadata = {}
    for c in all_classes:
        if parent_map[c]:
            try:
                depth = max(metadata[p]['depth'] for p in parent_map[c]) + 1
            except Exception:
                depth = 0
        else:
            depth = 0
        siblings = sum(len(child_map[p]) - 1 for p in parent_map[c])
        metadata[c] = {
            'depth': depth,
            'siblings': siblings,
            'subclasses': len(child_map.get(c, [])),
            'parents': len(parent_map[c]),
            'ancestors': ancestors_map[c]
        }

    relations = {'subclassOf', 'equivalentTo', 'disjointWith', 'complementOf', 'unionOf', 'intersectionOf'}
    triples = []

    for c in all_classes:
        try:
            for p in parent_map[c]:
                triples.append((c, 'subclassOf', p))
            for eq in getattr(c, 'equivalent_to', []):
                if isinstance(eq, ThingClass):
                    triples.append((c, 'equivalentTo', eq))
            for dis in getattr(c, 'disjoint_with', []):
                if isinstance(dis, ThingClass):
                    triples.append((c, 'disjointWith', dis))
            for r in c.is_a:
                if isinstance(r, Restriction):
                    name = r.property.python_name
                    if name in relations:
                        val = getattr(r, 'value', None) or getattr(r, 'some_values_from', None) or getattr(r,
                                                                                                           'all_values_from',
                                                                                                           None)
                        if isinstance(val, ThingClass):
                            triples.append((c, name, val))
        except Exception as e:
            logging.warning(f"Failed explicit triples for {c}: {e}")

    try:
        for a, b in onto.disjoint_classes():
            As = a if isinstance(a, (list, tuple, set)) else [a]
            Bs = b if isinstance(b, (list, tuple, set)) else [b]
            for x in As:
                for y in Bs:
                    if isinstance(x, ThingClass) and isinstance(y, ThingClass):
                        triples.append((x, 'disjointWith', y))
    except Exception as e:
        logging.warning(f"Failed global disjoints: {e}")

    graph = onto.world.as_rdflib_graph()
    for c in all_classes:
        try:
            subj = URIRef(c.iri)
            for pred, obj in graph.predicate_objects(subj):
                local = str(pred).rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                if local in ('complementOf', 'unionOf', 'intersectionOf'):
                    node = obj
                    while node and node != RDF.nil:
                        first = graph.value(node, RDF.first)
                        ent = onto.world._entities.get(str(first))
                        if isinstance(ent, ThingClass):
                            triples.append((c, local, ent))
                        node = graph.value(node, RDF.rest)
                        if node == obj:
                            logging.warning(f"Cycle detected in RDF list for {c}")
                            break
        except Exception as e:
            logging.warning(f"Failed RDF list for {c}: {e}")

    for prop in onto.object_properties():
        try:
            name = prop.python_name
            if name in relations:
                for c in all_classes:
                    for o in prop[c]:
                        if isinstance(o, ThingClass):
                            triples.append((c, name, o))
        except Exception as e:
            logging.warning(f"Failed object prop {prop}: {e}")

    triples = list(set(triples))
    logging.info(f"Extracted {len(triples)} unique triples")
    return all_classes, triples, metadata


def process_owl_file(file_path, max_q=MAX_QUESTIONS):
    world = World()
    try:
        onto = world.get_ontology(f"file://{os.path.abspath(file_path)}").load()
    except Exception as e:
        logging.error(f"Failed to load ontology {file_path}: {e}")
        return

    all_classes, triples, metadata = extract_and_prepare(onto)
    gen = RelationQuestionGenerator(triples, all_classes, metadata)
    questions = gen.generate_all(max_q)

    out_dir = os.path.dirname(file_path).replace('data', 'bench/bench_1_2')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'relations_opt_{os.path.basename(file_path)}.json')
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved {len(questions)} questions to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save questions for {file_path}: {e}")
    finally:
        world.close()
        label_cache.clear()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    random.seed(42)

    files = [os.path.join(r, f) for r, _, fs in os.walk(BASE_DIR)
             for f in fs if f.lower().endswith(EXTENSIONS)]
    logging.info(f"Found {len(files)} files to process.")
    for fp in files:
        try:
            process_owl_file(fp, MAX_QUESTIONS)
        except Exception as e:
            logging.error(f"{fp} failed: {e}")