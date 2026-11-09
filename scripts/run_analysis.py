
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_analyzer import DataAnalyzer
from src.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.data_preparer import DataPreparer


EDA_ANALYSES = EDAAnalyzer.ANALYSES
EDA_VISUALIZATIONS = EDAAnalyzer.VISUALIZATIONS
PREPARATION_ANALYSES = DataPreparer.ANALYSES
CLEANING_ANALYSES = DataCleaner.ANALYSES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Oil & Gas data-quality and EDA analyses."
    )

    execution_options = parser.add_mutually_exclusive_group()
    execution_options.add_argument(
        "--group",
        choices=[
            "prepare",
            "cleaning",
            "data-quality",
            "eda",
            "visualize",
            "both",
        ],
        help=(
            "Select data-quality analyses, EDA analyses, visualizations, "
            "or both analysis groups."
        ),
    )

    execution_options.add_argument(
        "--analysis",
        help="Run one selected analysis.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available analyses.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    data_quality_analyses = DataAnalyzer.ANALYSES

    if args.list:
        print("\nDATA QUALITY ANALYSES")
        for name in data_quality_analyses:
            print(f"  {name}")

        print("\nEDA ANALYSES")
        for name in EDA_ANALYSES:
            print(f"  {name}")

        print("\nEDA VISUALIZATIONS")
        for name in EDA_VISUALIZATIONS:
            print(f"  {name}")

        print("\nDATA PREPARATION")
        for name in PREPARATION_ANALYSES:
            print(f"  {name}")

        print("\nDATA CLEANING")
        for name in CLEANING_ANALYSES:
            print(f"  {name}")
        return

    if args.analysis:
        if args.analysis in PREPARATION_ANALYSES:
            logging.info("Running data preparation: %s", args.analysis)
            getattr(DataPreparer(), args.analysis)()
        elif args.analysis in CLEANING_ANALYSES:
            logging.info("Running data cleaning: %s", args.analysis)
            getattr(DataCleaner(), args.analysis)()
        elif args.analysis in EDA_VISUALIZATIONS:
            logging.info("Running visualization: %s", args.analysis)
            getattr(EDAAnalyzer(), args.analysis)()
        elif args.analysis in EDA_ANALYSES:
            logging.info("Running EDA: %s", args.analysis)
            getattr(EDAAnalyzer(), args.analysis)()
        elif args.analysis in data_quality_analyses:
            analyzer = DataAnalyzer()
            logging.info("Running data quality analysis: %s", args.analysis)
            getattr(analyzer, args.analysis)()
        else:
            parser.error(f"Unknown analysis: {args.analysis}")

        return

    if args.group == "eda":
        logging.info("Running all EDA analyses.")
        EDAAnalyzer().run_all()

    elif args.group == "visualize":
        logging.info("Running EDA visualizations.")
        EDAAnalyzer().run_visualizations()

    elif args.group == "prepare":
        logging.info("Running data preparation.")
        DataPreparer().run_all()

    elif args.group == "cleaning":
        logging.info("Running data cleaning.")
        DataCleaner().run_all()

    elif args.group == "data-quality":
        logging.info("Running all data quality analyses.")
        DataAnalyzer().run_all()

    elif args.group == "both":
        logging.info("Running all data quality and EDA analyses.")
        DataAnalyzer().run_all()
        EDAAnalyzer().run_all()

    else:
        parser.error(
            "Specify --group "
            "(prepare, cleaning, data-quality, eda, visualize, or both), "
            "--analysis, or use --list."
        )


if __name__ == "__main__":
    main()
