"""Shared utilities for Processing task generators."""

from .alignment import (
    build_alignment_rows,
    build_global_label_alignment_rows,
    write_alignment_csv,
)
from .entities import get_comment, get_definition, get_label
from .ontology import BaseOntologyLoader, configure_world_paths, load_annotation_ontologies, load_ontology
from .runtime import (
    build_mirrored_output_dir,
    configure_logging,
    discover_ontology_files,
    empty_marker_path,
    FileProcessingTimeout,
    file_timeout,
    resolve_onto_paths,
    save_json,
    save_empty_marker,
    should_skip_existing,
    limit_questions_by_subject,
    slugify_for_windows,
    suppress_library_noise,
)
from .stats import (
    class_depth,
    class_stats,
    direct_parents,
    direct_subclasses,
    global_class_metrics,
    selection_weight,
    siblings,
)
from .tasks import AUXILIARY_TASKS, MAIN_SPLITS, MAIN_TASKS, TASKS_BY_SPLIT

__all__ = [
    "build_alignment_rows",
    "build_global_label_alignment_rows",
    "BaseOntologyLoader",
    "build_mirrored_output_dir",
    "configure_world_paths",
    "configure_logging",
    "discover_ontology_files",
    "empty_marker_path",
    "FileProcessingTimeout",
    "file_timeout",
    "get_comment",
    "get_definition",
    "get_label",
    "load_annotation_ontologies",
    "load_ontology",
    "resolve_onto_paths",
    "save_json",
    "save_empty_marker",
    "should_skip_existing",
    "limit_questions_by_subject",
    "slugify_for_windows",
    "suppress_library_noise",
    "write_alignment_csv",
    "class_depth",
    "class_stats",
    "direct_parents",
    "direct_subclasses",
    "global_class_metrics",
    "selection_weight",
    "siblings",
    "AUXILIARY_TASKS",
    "MAIN_SPLITS",
    "MAIN_TASKS",
    "TASKS_BY_SPLIT",
]
