import argparse
import logging
import sys
from pathlib import Path


PROCESSING_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from common import build_alignment_rows, configure_logging, write_alignment_csv


TASK_LABEL_DEFAULT = "A1"
DOMAIN_DEFAULT = "lebensmittel"


def main():
    parser = argparse.ArgumentParser(description="Generate ontology alignment tasks for the Lebensmittel dataset.")
    parser.add_argument("--owl-dir", type=str, required=True, help="Directory containing .owl files.")
    parser.add_argument("--rdf-dir", type=str, required=True, help="Directory containing .rdf alignment files.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--task-label", type=str, default=TASK_LABEL_DEFAULT, help="Task label.")
    parser.add_argument("--domain", type=str, default=DOMAIN_DEFAULT, help="Domain label.")
    parser.add_argument(
        "--concept-scope",
        type=str,
        choices=["all", "native", "imported"],
        default="all",
        help="Concept scope filter; for alignment tasks, imported is represented as empty label lists.",
    )
    parser.add_argument("--no-warnings", action="store_true", help="Unused compatibility flag kept for CLI stability.")
    parser.add_argument("--log", type=str, default="info", help="Logging level: debug, info, warning, error.")

    args = parser.parse_args()
    configure_logging(args.log)

    rows = build_alignment_rows(
        owl_dir=Path(args.owl_dir),
        rdf_dir=Path(args.rdf_dir),
        task_label=args.task_label,
        domain=args.domain,
        concept_scope=args.concept_scope,
    )
    write_alignment_csv(rows, Path(args.output_csv))
    logging.info("Generated CSV: %s", args.output_csv)


if __name__ == "__main__":
    main()
