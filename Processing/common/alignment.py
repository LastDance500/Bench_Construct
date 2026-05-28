import csv
import logging
import os
from pathlib import Path

import rdflib
from lxml import etree
from rdflib import OWL, RDF, URIRef


ALIGNMENT_FIELDS = ["question", "answer", "task_label", "domain", "source_file", "labels"]


def iri_to_local(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rsplit("/", 1)[-1]


def write_alignment_csv(rows, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ALIGNMENT_FIELDS})


def extract_named_classes(owl_file_path: Path, max_label_length: int = 128):
    graph = rdflib.Graph()
    for fmt in ("xml", "turtle", "n3", "trig"):
        try:
            graph.parse(str(owl_file_path), format=fmt)
            break
        except Exception:
            graph = None

    if graph:
        iris, labels = [], []
        for subject in graph.subjects(RDF.type, OWL.Class):
            if not isinstance(subject, URIRef):
                continue
            iri = str(subject)
            label = iri_to_local(iri)
            if len(label) > max_label_length:
                continue
            iris.append(iri)
            labels.append(label)
        if labels:
            return iris, labels

    namespaces = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    }
    tree = etree.parse(str(owl_file_path))
    iris, labels = [], []
    for cls in tree.xpath("//owl:Class", namespaces=namespaces):
        iri = cls.get(f"{{{namespaces['rdf']}}}about") or cls.get(f"{{{namespaces['rdf']}}}ID")
        if not iri:
            continue
        label_node = cls.find("rdfs:label", namespaces=namespaces)
        label = label_node.text.strip() if label_node is not None and label_node.text else iri_to_local(iri)
        if len(label) > max_label_length:
            continue
        iris.append(iri)
        labels.append(label)
    return iris, labels


def parse_alignment_pairs(alignment_file_path: Path, max_label_length: int = 64):
    namespaces = {
        "al": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    tree = etree.parse(str(alignment_file_path))
    pairs = []
    for cell in tree.xpath("//al:Cell", namespaces=namespaces):
        left = cell.xpath("al:entity1/@rdf:resource", namespaces=namespaces)
        right = cell.xpath("al:entity2/@rdf:resource", namespaces=namespaces)
        if not left or not right:
            continue
        left_label = iri_to_local(left[0])
        right_label = iri_to_local(right[0])
        if len(left_label) > max_label_length or len(right_label) > max_label_length:
            continue
        pairs.append((left_label, right_label))
    return pairs


def build_alignment_rows(
    owl_dir: Path,
    rdf_dir: Path,
    task_label: str,
    domain: str,
    imported_scope_empty: bool = True,
    concept_scope: str = "all",
):
    rows = []
    for rdf_name in os.listdir(str(rdf_dir)):
        if not rdf_name.lower().endswith(".rdf"):
            continue
        rdf_path = rdf_dir / rdf_name
        base = rdf_name[:-4]
        if "-" not in base:
            continue
        left, right = base.split("-", 1)
        owl_left = owl_dir / f"{left}.owl"
        owl_right = owl_dir / f"{right}.owl"
        if not (owl_left.exists() and owl_right.exists()):
            logging.warning("Missing OWL for %s", rdf_name)
            continue

        _, left_labels = extract_named_classes(owl_left)
        _, right_labels = extract_named_classes(owl_right)
        if imported_scope_empty and concept_scope == "imported":
            left_labels, right_labels = [], []

        rows.append(
            {
                "question": (
                    "Please align the classes between these two ontologies:\n"
                    f"- Ontology 1 classes: {left_labels}\n"
                    f"- Ontology 2 classes: {right_labels}"
                ),
                "answer": str(parse_alignment_pairs(rdf_path)),
                "task_label": task_label,
                "domain": domain,
                "source_file": rdf_name,
                "labels": str(left_labels + right_labels),
            }
        )
    return rows


def extract_label_map(owl_path: str) -> dict:
    namespaces = {
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    tree = etree.parse(owl_path)
    mapping = {}
    for label_node in tree.xpath("//rdfs:label", namespaces=namespaces):
        text = (label_node.text or "").strip()
        if not text:
            continue
        parent = label_node.getparent()
        iri = parent.get(f"{{{namespaces['rdf']}}}about")
        if not iri:
            continue
        mapping[iri_to_local(iri)] = text
    return mapping


def build_global_label_alignment_rows(
    owl_dir: Path,
    rdf_dir: Path,
    task_label: str,
    domain: str,
    concept_scope: str = "all",
):
    owl_maps = {}
    global_map = {}
    for filename in os.listdir(str(owl_dir)):
        if not filename.lower().endswith(".owl"):
            continue
        base = os.path.splitext(filename)[0]
        mapping = extract_label_map(str(owl_dir / filename))
        owl_maps[base] = mapping
        global_map.update(mapping)

    namespaces = {
        "al": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    rows = []
    for filename in os.listdir(str(rdf_dir)):
        if not filename.lower().endswith(".rdf"):
            continue
        base = os.path.splitext(filename)[0]
        parts = base.split("-")
        if len(parts) < 3:
            continue
        project, lang1, lang2 = parts[0], parts[-2], parts[-1]
        onto1 = f"{project}-{lang1}"
        onto2 = f"{project}-{lang2}"
        labels1 = list(owl_maps.get(onto1, {}).values())
        labels2 = list(owl_maps.get(onto2, {}).values())
        if concept_scope == "imported":
            labels1, labels2 = [], []

        tree = etree.parse(str(rdf_dir / filename))
        pairs = []
        for cell in tree.xpath("//al:Cell", namespaces=namespaces):
            left = cell.xpath("al:entity1/@rdf:resource", namespaces=namespaces)
            right = cell.xpath("al:entity2/@rdf:resource", namespaces=namespaces)
            if not left or not right:
                continue
            left_local = iri_to_local(left[0])
            right_local = iri_to_local(right[0])
            pairs.append((global_map.get(left_local, left_local), global_map.get(right_local, right_local)))

        rows.append(
            {
                "question": (
                    "Please align the entities between these two ontologies:\n"
                    f"- Ontology 1 ({onto1}) labels: {labels1}\n"
                    f"- Ontology 2 ({onto2}) labels: {labels2}"
                ),
                "answer": str(pairs),
                "task_label": task_label,
                "domain": domain,
                "source_file": filename,
                "labels": str(labels1 + labels2),
            }
        )
    return rows
