import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional
from contextlib import redirect_stderr, redirect_stdout

from owlready2 import PREDEFINED_ONTOLOGIES, World, onto_path


_REGISTERED_IMPORT_ROOTS: set[Path] = set()
_ONTOLOGY_IRI_RE = re.compile(
    rb"<owl:Ontology[^>]*(?:rdf:about|rdf:IRI)\s*=\s*[\"']([^\"']+)[\"']",
    re.I,
)
_ONTOLOGY_EXTENSIONS = {".owl", ".rdf", ".rdfs", ".ttl", ".xml", ".n3"}


def configure_world_paths(world: World, extra_paths: Optional[Iterable[Path]]) -> None:
    if not extra_paths:
        return
    for path in extra_paths:
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            continue
        if hasattr(world, "_ontology_path") and resolved not in world._ontology_path:
            world._ontology_path.append(resolved)
        if resolved not in onto_path:
            onto_path.append(resolved)


def _local_import_root(file_path: Path) -> Path:
    resolved = Path(file_path).resolve()
    parts = resolved.parts
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
        return Path(*parts[: idx + 1])
    return resolved.parent


def _ontology_iri_from_file(path: Path) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            head = handle.read(262_144)
    except Exception:
        return None
    match = _ONTOLOGY_IRI_RE.search(head)
    if not match:
        return None
    try:
        iri = match.group(1).decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    return iri or None


def register_local_imports(file_path: Path) -> None:
    root = _local_import_root(file_path)
    if root in _REGISTERED_IMPORT_ROOTS or not root.exists():
        return
    _REGISTERED_IMPORT_ROOTS.add(root)
    registered = 0
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in _ONTOLOGY_EXTENSIONS:
            continue
        iri = _ontology_iri_from_file(candidate)
        if not iri:
            continue
        local_uri = str(candidate.resolve())
        PREDEFINED_ONTOLOGIES.setdefault(iri, local_uri)
        if iri.endswith(("/", "#")):
            PREDEFINED_ONTOLOGIES.setdefault(iri.rstrip("/#"), local_uri)
        registered += 1
    logging.debug("Registered %d local ontology import mappings under %s", registered, root)


def load_annotation_ontologies(world: World, iris: Iterable[str]) -> None:
    for iri in iris:
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    world.get_ontology(iri).load()
        except Exception as exc:
            logging.debug("Failed loading annotation ontology %s: %s", iri, exc)


def load_ontology(world: World, file_path: Path, load_imports: bool = True):
    file_path = Path(file_path)
    register_local_imports(file_path)
    configure_world_paths(world, [file_path.parent])
    iri = f"file://{file_path.resolve()}"
    ontology = world.get_ontology(iri)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                if load_imports:
                    ontology.load()
                else:
                    ontology.load(only_local=True)
    except Exception as exc:
        mode = "with imports" if load_imports else "local-only"
        logging.error("Failed loading ontology %s (%s): %s", file_path, mode, exc)
        return None
    return ontology


class BaseOntologyLoader:
    def __init__(self, file_path: Path, load_imports: bool = True, onto_paths: Optional[Iterable[Path]] = None):
        self.file_path = Path(file_path)
        self.world = World()
        self.onto = None
        self.load_imports = load_imports
        configure_world_paths(self.world, onto_paths)

    def annotation_ontology_iris(self) -> tuple[str, ...]:
        return ()

    def load(self):
        annotation_iris = self.annotation_ontology_iris()
        if annotation_iris:
            load_annotation_ontologies(self.world, annotation_iris)
        ontology = load_ontology(self.world, self.file_path, load_imports=self.load_imports)
        if ontology is None:
            return None
        self.onto = ontology
        return ontology
