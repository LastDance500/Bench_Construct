import json
import os
import random
import logging
from owlready2 import World, owl, ThingClass

# Global caches for definitions and labels
definition_cache = {}
label_cache = {}

def get_definition(entity):
    """
    Retrieve the definition of an entity with preference for English definition.
    Returns a string definition or "No definition provided" if none exists.
    """
    try:
        key = str(entity.iri)
        if key in definition_cache:
            return definition_cache[key]

        definition = None
        # 1. Try IAO_0000115
        defs = getattr(entity, "IAO_0000115", None)
        if defs and len(defs) > 0:
            definition = next((d for d in defs if getattr(d, 'lang', None) == 'en'), defs[0])

        # 2. skos:definition
        if not definition:
            skos_defs = getattr(entity, "definition", None)
            if skos_defs and len(skos_defs) > 0:
                definition = next((d for d in skos_defs if getattr(d, 'lang', None) == 'en'), skos_defs[0])

        # 3. rdfs:comment
        if not definition:
            comment = getattr(entity, "comment", None)
            if comment:
                if isinstance(comment, list) and len(comment) > 0:
                    definition = next((d for d in comment if getattr(d, 'lang', None) == 'en'), comment[0])
                elif isinstance(comment, str):
                    definition = comment

        # Convert to string and ensure it's not empty
        definition = str(definition) if definition and str(definition).strip() else "No definition provided."
        definition_cache[key] = definition
        return definition
    except Exception as e:
        logging.warning(f"Error retrieving definition for {getattr(entity, 'name', 'unknown')}: {e}")
        return "No definition provided."

def get_label(entity):
    """
    Retrieve the label for an entity, preferring rdfs:label, otherwise using entity name.
    Returns a string label.
    """
    try:
        key = str(entity.iri)
        if key in label_cache:
            return label_cache[key]
        label = getattr(entity, "label", None)
        result = str(label[0]) if label and len(label) > 0 and isinstance(label, list) else str(entity.name)
        label_cache[key] = result
        return result
    except Exception as e:
        logging.warning(f"Error retrieving label for {getattr(entity, 'name', 'unknown')}: {e}")
        return str(entity.name)

def compute_depth(entity, memo=None):
    """
    Compute the depth of a concept in the ontology hierarchy from owl.Thing.
    Uses memoization to avoid redundant calculations.
    """
    if memo is None:
        memo = {}
    if entity in memo:
        return memo[entity]
    if entity == owl.Thing:
        memo[entity] = 0
        return 0
    parents = [p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing]
    depth = 1 if not parents else max((compute_depth(p, memo) for p in parents), default=1) + 1
    memo[entity] = depth
    return depth

def get_siblings(entity):
    """
    Get all sibling concepts (subclasses of the same parent(s)).
    """
    siblings = set()
    for parent in entity.is_a:
        if isinstance(parent, ThingClass) and parent != owl.Thing:
            siblings.update(parent.subclasses())
    siblings.discard(entity)
    return siblings

def compute_global_metrics(classes):
    """
    Compute global stats for normalization: max depth, sibling, subclass, parent counts.
    """
    max_depth = max_sibling = max_subclass = max_parent = 0
    for cls in classes:
        try:
            d = compute_depth(cls)
            s = len(get_siblings(cls))
            sub = len(list(cls.subclasses()))
            par = len([p for p in cls.is_a if isinstance(p, ThingClass) and p != owl.Thing])
            max_depth = max(max_depth, d)
            max_sibling = max(max_sibling, s)
            max_subclass = max(max_subclass, sub)
            max_parent = max(max_parent, par)
        except Exception as e:
            logging.warning(f"Error computing metrics for {getattr(cls, 'name', 'unknown')}: {e}")
    return {
        "max_depth": max(max_depth, 1),  # Avoid division by zero
        "max_sibling_count": max(max_sibling, 1),
        "max_subclass_count": max(max_subclass, 1),
        "max_parent_count": max(max_parent, 1)
    }

def compute_selection_weight(entity, global_metrics):
    """
    Compute selection weight for an entity based on normalized metrics.
    Weight = (norm_depth * (norm_sibling + 1)) / (norm_subclass + norm_parent + 1)
    """
    try:
        depth = compute_depth(entity)
        sib = len(get_siblings(entity))
        sub = len(list(entity.subclasses()))
        par = len([p for p in entity.is_a if isinstance(p, ThingClass) and p != owl.Thing])

        gm = global_metrics
        nd = depth / gm["max_depth"]
        ns = sib / gm["max_sibling_count"]
        nc = sub / gm["max_subclass_count"]
        np = par / gm["max_parent_count"]

        return nd * (ns + 1) / (nc + np + 1)
    except Exception as e:
        logging.warning(f"Error computing weight for {getattr(entity, 'name', 'unknown')}: {e}")
        return 0.0

class OntologyLoader:
    """
    Load an OWL ontology in an isolated World to avoid conflicts.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.world = World()
        self.onto = None

    def load(self):
        """
        Load the ontology from the specified file path.
        Returns the loaded ontology or None if loading fails.
        """
        logging.info(f"Loading ontology: {self.file_path}")
        ontology_ref = f"file://{os.path.abspath(self.file_path)}"
        try:
            onto = self.world.get_ontology(ontology_ref)
            onto.load(only_local=True)
            self.onto = onto
            return onto
        except Exception as e:
            logging.error(f"Error loading ontology {self.file_path}: {e}")
            return None

    def preload_entities(self):
        """
        Preload entity attributes to improve performance.
        """
        if not self.onto:
            return
        for cls in self.onto.classes():
            try:
                _ = getattr(cls, "IAO_0000115", None)
                _ = getattr(cls, "definition", None)
                _ = getattr(cls, "comment", None)
                _ = getattr(cls, "label", None)
            except Exception as e:
                logging.warning(f"Error preloading attributes for {getattr(cls, 'name', 'unknown')}: {e}")

    def get_all_classes_with_definition(self):
        """
        Get all classes with valid definitions (excluding owl.Thing).
        """
        if not self.onto:
            return []
        return [cls for cls in self.onto.classes() if cls != owl.Thing and get_definition(cls) != "No definition provided."]

class DefinitionQuestionGenerator:
    """
    Generate open-ended definition questions for ontology classes.
    """
    def __init__(self, classes, global_metrics, max_questions=100):
        self.classes = classes
        self.global_metrics = global_metrics
        self.max_questions = max_questions
        self.inferred = False  # Placeholder; extend if inference is implemented

    def generate_question_for_target(self, target):
        """
        Generate a question for a given target class, including meta information.
        """
        try:
            target_label = get_label(target)
            target_def = get_definition(target)
            depth = compute_depth(target)
            sibling_count = len(get_siblings(target))
            subclass_count = len(list(target.subclasses()))
            parent_count = len([p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing])

            # Get parent for relation and object_iri (assuming is_a relationship)
            parents = [p for p in target.is_a if isinstance(p, ThingClass) and p != owl.Thing]
            relation = "is_a" if parents else None
            object_iri = str(parents[0].iri) if parents else None

            return {
                "prompt": f"Please provide the definition of the concept '{target_label}'.",
                "definition": target_def,
                "meta": {
                    "iri": str(target.iri),
                    "label": target_label,
                    "depth": depth,
                    "sibling_count": sibling_count,
                    "subclass_count": subclass_count,
                    "parent_count": parent_count,
                    "relation": relation,
                    "object_iri": object_iri,
                    "inferred": self.inferred
                }
            }
        except Exception as e:
            logging.warning(f"Error generating question for {getattr(target, 'name', 'unknown')}: {e}")
            return None

    def generate_all_questions(self):
        """
        Generate questions by sampling classes based on their weights.
        Returns a list of question dictionaries.
        """
        if not self.classes:
            return []

        # Compute weights for all classes
        weights = [compute_selection_weight(cls, self.global_metrics) for cls in self.classes]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(self.classes)] * len(self.classes)  # Uniform weights if all are zero

        # Sample classes based on weights
        num_questions = min(self.max_questions, len(self.classes))
        selected_classes = random.choices(self.classes, weights=weights, k=num_questions)

        questions = []
        for target in selected_classes:
            question = self.generate_question_for_target(target)
            if question:
                questions.append(question)

        return questions

def save_questions(questions, save_path):
    """
    Save questions to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved {len(questions)} questions to {save_path}")
    except Exception as e:
        logging.error(f"Error saving questions to {save_path}: {e}")

def process_owl_file(file_path):
    """
    Process a single OWL file, generate questions, and save them.
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.dirname(file_path).replace("data", "bench/bench_3_1")
    save_path = os.path.join(out_dir, f"class_definitions_{base}.json")
    if os.path.exists(save_path):
        logging.info(f"Skip existing: {save_path}")
        return
    loader = OntologyLoader(file_path)
    onto = loader.load()
    if not onto:
        logging.error(f"Load failed: {file_path}")
        return
    loader.preload_entities()
    classes = loader.get_all_classes_with_definition()
    logging.info(f"Found {len(classes)} classes with definitions in {file_path}")
    if not classes:
        logging.info(f"No defined classes in {file_path}")
        return
    global_metrics = compute_global_metrics(classes)
    gen = DefinitionQuestionGenerator(classes, global_metrics, max_questions=100)
    questions = gen.generate_all_questions()
    logging.info(f"Generated {len(questions)} questions for {file_path}")
    if questions:
        save_questions(questions, save_path)
    # Clear caches
    definition_cache.clear()
    label_cache.clear()

def main():
    """
    Process all OWL/RDF files in the base directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("process.log", "w", "utf-8")
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