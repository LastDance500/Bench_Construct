import json
import os
import random
import logging
from owlready2 import World, ThingClass, owl

BASE_DIR = '../../../data'
EXTENSIONS = ('.owl', '.rdf', '.rdfs', '.ttl')
MAX_QUESTIONS = None
MIN_DISTRACTORS = 2

label_cache = {}


def get_label(entity):
    key = str(getattr(entity, 'iri', str(entity)))
    if key in label_cache:
        return label_cache[key]
    labs = getattr(entity, 'label', []) or getattr(entity, 'prefLabel', []) or []
    label = labs[0] if labs else (entity.name if hasattr(entity, 'name') else str(entity))
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
    parents = [p for p in entity.is_a if isinstance(p, ThingClass)]
    depth = 1 if not parents else max(compute_depth(p, memo) for p in parents) + 1
    memo[entity] = depth
    return depth

def get_siblings(entity):
    sibs = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass):
            sibs.update(p.subclasses())
    sibs.discard(entity)
    return sibs

def get_ancestors(entity, memo=None):
    if memo is None:
        memo = {}
    if entity in memo:
        return memo[entity]
    ancestors = set()
    for p in entity.is_a:
        if isinstance(p, ThingClass):
            ancestors.add(p)
            ancestors |= get_ancestors(p, memo)
    memo[entity] = ancestors
    return ancestors

def extract_dataproperty_info(onto):
    triples = []
    prop_info = {}
    data_props = list(onto.data_properties())
    logging.info(f"Found {len(data_props)} data properties in ontology")
    for prop in data_props:
        domains = set()
        ranges = set()
        for domain in prop.domain:
            if isinstance(domain, ThingClass):
                domains.add(domain)
        for range_type in prop.range:
            ranges.add(range_type)
        if domains:
            prop_info[prop] = {'domains': domains, 'ranges': ranges}
            for domain in domains:
                for range_type in ranges:
                    triples.append((prop, domain, range_type))
    logging.info(f"Generated {len(triples)} (Property, Domain, Range) triples")
    return triples, prop_info

class PropertyDomainRangeQuestionGenerator:
    def __init__(self, triples, prop_info, all_classes):
        self.triples = triples
        self.prop_info = prop_info
        self.all_classes = all_classes
        self.all_properties = list(prop_info.keys())

    def generate_one(self, prop, domain_cls, range_type, num_choices=4):
        try:
            depth = compute_depth(domain_cls)
            siblings = len(get_siblings(domain_cls))
            subclasses = len(list(domain_cls.subclasses()))
            parents = len([p for p in domain_cls.is_a if isinstance(p, ThingClass)])

            question_type = random.choice(['domain', 'range'])
            correct_answer = domain_cls if question_type == 'domain' else range_type

            distractors = []
            candidates = [c for c in self.all_classes if c != correct_answer]
            logging.debug(f"Generating {question_type} question for property {get_label(prop)}, candidates: {len(candidates)}")
            random.shuffle(candidates)
            for candidate in candidates:
                if question_type == 'domain':
                    if candidate not in self.prop_info[prop]['domains']:
                        distractors.append(candidate)
                else:  # range
                    if candidate not in self.prop_info[prop]['ranges']:
                        distractors.append(candidate)
                if len(distractors) >= num_choices - 1:
                    break

            if len(distractors) < num_choices - 1:
                logging.warning(f"Only {len(distractors)} distractors found for {get_label(prop)}, need {num_choices - 1}")
                for candidate in candidates:
                    if candidate != correct_answer and candidate not in distractors:
                        distractors.append(candidate)
                    if len(distractors) >= num_choices - 1:
                        break

            if len(distractors) < MIN_DISTRACTORS:
                logging.warning(f"Insufficient distractors ({len(distractors)}) for {get_label(prop)}, skipping question")
                return None

            options = [correct_answer] + distractors[:num_choices - 1]
            random.shuffle(options)

            letters = ['A', 'B', 'C', 'D'][:num_choices]
            opts = []
            correct = None
            for i, choice in enumerate(options):
                label = get_label(choice)
                opts.append({'option_letter': letters[i], 'label': label})
                if choice == correct_answer:
                    correct = letters[i]

            prompt = (f"Which of the following is a valid {question_type} for the data property '{get_label(prop)}'?")

            range_iri = str(getattr(range_type, 'iri', str(range_type))) if range_type else 'N/A'
            range_label = get_label(range_type) if range_type else str(range_type)

            return {
                'prompt': prompt,
                'options': opts,
                'correct_answer': correct,
                'meta': {
                    'property_iri': str(prop.iri),
                    'property_label': get_label(prop),
                    'question_type': question_type,
                    'domain_iri': str(domain_cls.iri),
                    'domain_label': get_label(domain_cls),
                    'range_iri': range_iri,
                    'range_label': range_label,
                    'depth': depth,
                    'sibling_count': siblings,
                    'subclass_count': subclasses,
                    'parent_count': parents
                }
            }
        except Exception as e:
            logging.error(f"Failed to generate question for property {prop}: {e}")
            return None

    def generate_all(self, max_q=None):
        questions = []
        for prop, domain, range_type in self.triples:
            q = self.generate_one(prop, domain, range_type)
            if q:
                questions.append(q)
                if max_q and len(questions) >= max_q:
                    logging.info(f"Reached MAX_QUESTIONS limit: {max_q}")
                    break
        logging.info(f"Generated {len(questions)} questions for this ontology")
        return questions

def save_questions(questions, save_path):
    if not questions:
        logging.info(f"No questions generated for {save_path}, skipping save")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved {len(questions)} questions to {save_path}")

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler('property2domain_range_questions.log', 'w', 'utf-8')])

    random.seed(42)
    files = []
    for root, _, filenames in os.walk(BASE_DIR):
        for fname in filenames:
            if fname.lower().endswith(EXTENSIONS):
                files.append(os.path.join(root, fname))
    logging.info(f"Found {len(files)} ontology files")

    files_processed = 0
    for fp in files:
        try:
            world = World()
            onto = world.get_ontology(f"file://{os.path.abspath(fp)}").load()
            logging.info(f"Successfully loaded ontology: {fp}")
            all_classes = list(onto.classes())
            triples, prop_info = extract_dataproperty_info(onto)
            gen = PropertyDomainRangeQuestionGenerator(triples, prop_info, all_classes)
            questions = gen.generate_all(MAX_QUESTIONS)
            out_dir = os.path.dirname(fp).replace('data', 'bench/bench_1_3')
            save_path = os.path.join(out_dir, f'property2domain_range_questions_{os.path.basename(fp)}.json')
            save_questions(questions, save_path)
            files_processed += 1
        except Exception as e:
            logging.error(f"Processing {fp} failed: {e}")
    logging.info(f"Processed {files_processed}/{len(files)} ontology files")

if __name__ == '__main__':
    main()