from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"


def get_parquet_files() -> list[Path]:
    """Return all Parquet files in the dataset."""
    return sorted(DATASET_DIR.rglob("*.parquet"))


def load_parquet_file(file_path: Path) -> pd.DataFrame:
    """Load an individual Parquet file."""
    return pd.read_parquet(file_path, engine="pyarrow")