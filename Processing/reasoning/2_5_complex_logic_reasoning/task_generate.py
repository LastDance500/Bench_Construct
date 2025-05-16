import os
import re
import json
import random
import logging
import types
import multiprocessing
import queue
from contextlib import contextmanager
from owlready2 import (
    World,
    Thing,
    sync_reasoner,
    Not,
    Nothing,
    ThingClass,
    onto_path,
)

# ---------- Configuration ----------
BASE_DIR = "../../../data"
OUTPUT_DIR = "../../../bench/bench_2_5"
EXTENSIONS = (".owl", ".rdf", ".ttl", ".rdfs")
NUM_PUZZLES_PER_ONTOLOGY = 100  # Total puzzles per ontology
MIN_COMPLEXITY = 4  # Minimum complexity score for SAT puzzles
MAX_CLASSES = 10000  # Skip ontology if it has more than this number of classes
REASONER_TIMEOUT = 600  # seconds timeout for reasoner
SKIP_FILES = {"Thesaurus.owl"}  # Files to skip
# ----------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_satisfiability.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
onto_path.append(os.path.abspath(BASE_DIR))  # Avoid network fetches
os.environ["JAVA_MEMORY"] = "4g"  # Limit Java heap size for reasoner


@contextmanager
def run_reasoner_with_timeout(world, timeout=REASONER_TIMEOUT):
    """
    Run sync_reasoner in a separate process with a timeout.
    Ensures proper process cleanup using a context manager.
    Raises TimeoutError if the reasoner does not finish in time.
    """

    def target(out_q):
        try:
            with world:
                sync_reasoner()
            out_q.put("done")
        except Exception as e:
            out_q.put(e)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(q,))
    p.daemon = False
    try:
        p.start()
        yield p, q
        p.join(timeout)
        if p.is_alive():
            logger.info("Sending SIGTERM to reasoner process")
            p.terminate()
            p.join(15)  # Increased grace period
            if p.is_alive():
                logger.warning("Process still alive, sending SIGKILL")
                p.kill()
                p.join(3)
            raise TimeoutError("Reasoning timed out")
    finally:
        if p.is_alive():
            logger.info("Ensuring process termination in cleanup")
            p.terminate()
            p.join(15)
            if p.is_alive():
                p.kill()
                p.join(3)
        p.close()


def dl_str(expr_str: str) -> str:
    # Convert OWLReady2 expression string to DL notation
    s = expr_str
    # Normalize hyphens
    s = re.sub(r'([A-Za-z0-9_])\-([A-Za-z0-9_])', r'\1_\2', s)
    # Some, Only, Min, Max, Exactly, Not
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.some\(\s*([^\)]+?)\s*\)', r'∃\1.(\2)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.only\(\s*([^\)]+?)\s*\)', r'∀\1.(\2)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.min\(\s*([0-9]+)\s*,\s*([^\)]+?)\s*\)', r'≥\2 \1.(\3)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.max\(\s*([0-9]+)\s*,\s*([^\)]+?)\s*\)', r'≤\2 \1.(\3)', s)
    s = re.sub(r'(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.exactly\(\s*([0-9]+)\s*,\s*([^\)]+?)\s*\)', r'=\2 \1.(\3)', s)
    s = re.sub(r'Not\(\s*([^\)]+?)\s*\)', r'¬(\1)', s)
    # Boolean connectives
    s = s.replace(' & ', ' ⊓ ').replace(' | ', ' ⊔ ')
    # Remove namespace prefixes
    s = re.sub(r'\b(?:[A-Za-z0-9_]+\.)+([A-Z][A-Za-z0-9_]+)\b', r'\1', s)
    return s


def apply_labels_to_dl(dl: str, class_map: dict, prop_map: dict) -> str:
    # Replace property identifiers with labels (spaces in labels already underscores)
    for ident, label in sorted(prop_map.items(), key=lambda x: -len(x[0])):
        dl = re.sub(r'\b' + re.escape(ident) + r'\b', label, dl)
    # Replace class identifiers with labels
    for ident, label in sorted(class_map.items(), key=lambda x: -len(x[0])):
        dl = re.sub(r'\b' + re.escape(ident) + r'\b', label, dl)
    return dl


def complexity_score(dl: str) -> int:
    return (
            len(re.findall(r'[∃∀]', dl)) +
            len(re.findall(r'[≥≤=]', dl)) +
            len(re.findall(r'[⊓⊔]', dl))
    )


def generate_puzzles_for_ontology(path: str, world: World) -> list:
    logger.info(f"Starting puzzle generation for {path}")
    iri = f"file://{os.path.abspath(path)}"
    onto = world.get_ontology(iri)
    try:
        onto.load(only_local=True)
    except Exception as e:
        logger.warning(f"Skipping ontology {path}: {e}")
        return []

    classes = list(onto.classes())
    if len(classes) > MAX_CLASSES:
        logger.warning(f"Skipping {path}: too many classes ({len(classes)})")
        return []
    props = list(onto.object_properties())
    if not classes or not props:
        logger.warning(f"Skipping {path}: {len(classes)} classes, {len(props)} props")
        return []

    # Build label maps: replace spaces in labels with underscores
    class_map = {cls.name: next(iter(cls.label), cls.name).replace(' ', '_') for cls in classes}
    all_props = props + list(onto.data_properties())
    prop_map = {prop.name: next(iter(prop.label), prop.name).replace(' ', '_') for prop in all_props}

    sat_seeds, unsat_seeds = [], []
    sat_target = NUM_PUZZLES_PER_ONTOLOGY // 2
    unsat_target = NUM_PUZZLES_PER_ONTOLOGY - sat_target

    # Generate SAT and UNSAT seeds
    for R in props:
        for C in random.sample(classes, min(5, len(classes))):
            sat_seeds += [R.some(C), R.only(C)]
    for C, D in random.sample([(c, d) for c in classes for d in classes], k=min(10, len(classes) ** 2)):
        sat_seeds += [C & random.choice(props).some(D), (C | D) & random.choice(props).only(random.choice(classes))]
    for C in random.sample(classes, min(len(classes), 10)):
        unsat_seeds.append(C & Not(C))
    for C in classes:
        for D in getattr(C, "disjoint_with", []):
            unsat_seeds.append(C & D)
    for R in props:
        C = random.choice(classes)
        unsat_seeds += [R.min(2, C) & R.max(1, C), R.some(C) & R.only(Not(C)), R.exactly(0, C) & R.some(C)]
    if not unsat_seeds and classes:
        unsat_seeds.append(classes[0] & Not(classes[0]))
    for expr in list(unsat_seeds):
        R = random.choice(props)
        D = random.choice(classes)
        unsat_seeds.append(expr & R.some(D))

    # Sample candidates
    candidates = random.sample(sat_seeds, min(len(sat_seeds), sat_target)) + \
                 random.sample(unsat_seeds, min(len(unsat_seeds), unsat_target))
    while len(candidates) < NUM_PUZZLES_PER_ONTOLOGY:
        if random.random() < 0.6:
            seed = random.choice(sat_seeds)
            R, D = random.choice(props), random.choice(classes)
            expr = seed & R.only(D)
        else:
            seed = random.choice(unsat_seeds)
            R, D = random.choice(props), random.choice(classes)
            expr = seed & R.some(D)
        if not isinstance(expr, ThingClass):
            candidates.append(expr)

    # Inject into temp ontology and reason with timeout
    tmp_iri = f"http://temp.org/tmp_{os.path.splitext(os.path.basename(path))[0]}.owl"
    tmp_onto = world.get_ontology(tmp_iri)
    tmp_onto.imported_ontologies.append(onto)
    for idx, expr in enumerate(candidates):
        with tmp_onto:
            cls = types.new_class(f"Tmp{idx}", (Thing,), {})
            cls.equivalent_to = [expr]
    try:
        logger.info(f"Running reasoner for {path}")
        with run_reasoner_with_timeout(world, timeout=REASONER_TIMEOUT) as (p, q):
            result = q.get(timeout=1)
        if isinstance(result, Exception):
            raise result
        logger.info(f"Reasoner completed for {path}")
    except TimeoutError as e:
        logger.error(f"Timeout during reasoning for {path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Reasoning failed for {path}: {e}")
        return []

    sats, unsats = [], []
    for idx, expr in enumerate(candidates):
        cls = tmp_onto[f"Tmp{idx}"]
        is_sat = Nothing not in cls.ancestors()
        raw = dl_str(str(expr))
        labeled = apply_labels_to_dl(raw, class_map, prop_map)
        comp = complexity_score(labeled)
        entry = {"expression": labeled, "satisfiable": is_sat, "complexity": comp}
        if is_sat and comp >= MIN_COMPLEXITY:
            sats.append(entry)
        elif not is_sat:
            unsats.append(entry)

    # Fill up to targets if needed
    if len(sats) < sat_target:
        need = sat_target - len(sats)
        fill = [e for e in sat_seeds if
                complexity_score(apply_labels_to_dl(dl_str(str(e)), class_map, prop_map)) >= MIN_COMPLEXITY]
        for expr in random.sample(fill, min(need, len(fill))):
            raw = dl_str(str(expr))
            labeled = apply_labels_to_dl(raw, class_map, prop_map)
            sats.append({"expression": labeled, "satisfiable": True, "complexity": complexity_score(labeled)})
    if len(unsats) < unsat_target:
        need = unsat_target - len(unsats)
        for expr in random.sample(unsat_seeds, min(need, len(unsat_seeds))):
            raw = dl_str(str(expr))
            labeled = apply_labels_to_dl(raw, class_map, prop_map)
            unsats.append({"expression": labeled, "satisfiable": False, "complexity": complexity_score(labeled)})

    logger.info(f"{path}: {len(sats)} SAT, {len(unsats)} UNSAT")
    final = sats[:sat_target] + unsats[:unsat_target]
    random.shuffle(final)

    tmp_onto.destroy()
    world.ontologies.clear()
    return final


def save_questions(questions: list, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {"metadata": {"ontology_file": os.path.basename(save_path), "num_questions": len(questions)},
            "questions": questions}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(questions)} puzzles to {save_path}")


def process_owl_file(file_path: str) -> None:
    logger.info(f"Checking {file_path}")

    # Skip if file doesn't exist or is not accessible
    if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
        logger.warning(f"Skipping {file_path}: file does not exist or is not readable")
        return

    # Skip if in SKIP_FILES
    if os.path.basename(file_path) in SKIP_FILES:
        logger.info(f"Skipping {file_path}: explicitly marked for skipping")
        return

    # Skip if already processed
    rel = os.path.relpath(file_path, BASE_DIR)
    out_dir = os.path.join(OUTPUT_DIR, os.path.dirname(rel))
    save_path = os.path.join(out_dir, f"satisfiability_puzzles_{os.path.basename(file_path)}.json")
    if os.path.exists(save_path):
        logger.info(f"Skipping {file_path}: output file {save_path} already exists")
        return

    logger.info(f"Processing {file_path}")
    world = World()
    try:
        puzzles = generate_puzzles_for_ontology(file_path, world)
        if puzzles:
            os.makedirs(out_dir, exist_ok=True)
            save_questions(puzzles, save_path)
    finally:
        logger.info(f"Closing World for {file_path}")
        world.close()


def main():
    random.seed(42)
    files = []
    for root, _, fnames in os.walk(BASE_DIR):
        for fn in fnames:
            if fn.lower().endswith(EXTENSIONS):
                files.append(os.path.join(root, fn))
    logger.info(f"Found {len(files)} files")
    try:
        for fp in files:
            process_owl_file(fp)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, cleaning up...")
        # Close any open World instances
        for world in list(World._instances):
            try:
                world.close()
            except Exception as e:
                logger.warning(f"Error closing World: {e}")
        raise SystemExit("Terminated by user")


if __name__ == "__main__":
    main()