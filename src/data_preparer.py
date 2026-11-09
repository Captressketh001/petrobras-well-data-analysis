from collections import Counter, defaultdict
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_loader import get_parquet_files, load_parquet_file


class DataPreparer:
    """Prepare complete real-instance data and a labelled modelling view."""

    ANALYSES = ("prepare_dataset",)

    CLASS_NAMES = {
        0: "Normal Operation",
        1: "Abrupt Increase of BSW",
        2: "Spurious Closure of DHSV",
        3: "Severe Slugging",
        4: "Flow Instability",
        5: "Rapid Productivity Loss",
        6: "Quick Restriction in PCK",
        7: "Scaling in PCK",
        8: "Hydrate in Production Line",
        9: "Hydrate in Service Line",
        101: "Transient: Abrupt Increase of BSW",
        102: "Transient: Spurious Closure of DHSV",
        103: "Transient: Severe Slugging",
        104: "Transient: Flow Instability",
        105: "Transient: Rapid Productivity Loss",
        106: "Transient: Quick Restriction in PCK",
        107: "Transient: Scaling in PCK",
        108: "Transient: Hydrate in Production Line",
        109: "Transient: Hydrate in Service Line",
    }

    INSTANCE_FILENAME = re.compile(
        r"(?P<well_id>WELL-\d+)_(?P<recording_id>\d+)\.parquet"
    )

    def __init__(
        self,
        processed_dir: Path | None = None,
        reports_dir: Path | None = None,
    ):
        self.project_root = Path(__file__).resolve().parent.parent
        self.processed_dir = (
            Path(processed_dir)
            if processed_dir is not None
            else self.project_root / "data" / "processed"
        )
        self.reports_dir = (
            Path(reports_dir) if reports_dir is not None
            else self.project_root / "reports"
        )
        self.complete_dataset_path = (
            self.processed_dir / "real_instances_complete.parquet"
        )
        self.labelled_dataset_path = (
            self.processed_dir / "real_instances_labelled.parquet"
        )
        self.report_path = self.reports_dir / "data_preparation_report.md"

    def run_all(self) -> None:
        """Run all registered preparation analyses."""
        for analysis_name in self.ANALYSES:
            print(f"Running data preparation: {analysis_name}")
            getattr(self, analysis_name)()

    def prepare_dataset(self) -> None:
        """Stream real instances into complete and labelled Parquet files."""
        real_files = [
            path for path in get_parquet_files()
            if path.name.startswith("WELL-")
        ]
        if not real_files:
            raise FileNotFoundError(
                "No real Parquet instances with the 'WELL-' prefix were found."
            )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        temporary_complete_path = self.complete_dataset_path.with_suffix(
            ".parquet.tmp"
        )
        temporary_labelled_path = self.labelled_dataset_path.with_suffix(
            ".parquet.tmp"
        )
        complete_writer = None
        labelled_writer = None
        output_schema = None
        process_sensors = None
        equipment_indicators = None
        expected_source_columns = None

        unique_wells = set()
        instances_by_folder = Counter()
        class_counts = Counter()
        unknown_class_count = 0
        total_observations_read = 0
        total_observations_retained = 0
        labelled_observations = 0
        missing_counts = defaultdict(int)

        try:
            for file_number, file_path in enumerate(real_files, start=1):
                match = self.INSTANCE_FILENAME.fullmatch(file_path.name)
                if match is None:
                    raise ValueError(
                        "Unexpected real-instance filename format: "
                        f"{file_path.name}"
                    )

                dataframe = load_parquet_file(file_path)
                if not isinstance(dataframe.index, pd.DatetimeIndex):
                    raise TypeError(
                        f"Expected a datetime index in {file_path.name}; "
                        f"got {type(dataframe.index).__name__}."
                    )
                if not dataframe.index.is_monotonic_increasing:
                    raise ValueError(
                        f"Timestamps are not chronological in {file_path.name}."
                    )
                if not {"class", "state"}.issubset(dataframe.columns):
                    raise ValueError(
                        f"Expected 'class' and 'state' columns in "
                        f"{file_path.name}."
                    )

                source_columns = tuple(dataframe.columns)
                if expected_source_columns is None:
                    expected_source_columns = source_columns
                    sensor_columns = [
                        column
                        for column in source_columns
                        if column not in {"class", "state"}
                    ]
                    equipment_indicators = [
                        column
                        for column in sensor_columns
                        if column.startswith("ESTADO-")
                    ]
                    process_sensors = [
                        column
                        for column in sensor_columns
                        if not column.startswith("ESTADO-")
                    ]
                elif source_columns != expected_source_columns:
                    raise ValueError(
                        "Source columns differ between instances; "
                        f"unexpected schema in {file_path.name}."
                    )

                timestamp_name = dataframe.index.name or "index"
                prepared = dataframe.reset_index().rename(
                    columns={timestamp_name: "timestamp"}
                )
                if "timestamp" not in prepared.columns:
                    raise ValueError(
                        f"Could not preserve timestamps from {file_path.name}."
                    )

                well_id = match.group("well_id")
                event_folder = file_path.parent.name
                prepared.insert(1, "instance_id", file_path.stem)
                prepared.insert(2, "file_name", file_path.name)
                prepared.insert(3, "well_id", well_id)
                prepared.insert(4, "event_folder", event_folder)
                prepared = prepared[
                    [
                        "timestamp",
                        "instance_id",
                        "file_name",
                        "well_id",
                        "event_folder",
                        *process_sensors,
                        *equipment_indicators,
                        "state",
                        "class",
                    ]
                ]

                table = pa.Table.from_pandas(
                    prepared,
                    preserve_index=False,
                )
                if output_schema is None:
                    output_schema = table.schema
                    complete_writer = pq.ParquetWriter(
                        temporary_complete_path,
                        output_schema,
                        compression="brotli",
                    )
                    labelled_writer = pq.ParquetWriter(
                        temporary_labelled_path,
                        output_schema,
                        compression="brotli",
                    )
                elif not table.schema.equals(
                    output_schema,
                    check_metadata=False,
                ):
                    raise ValueError(
                        "Prepared column types differ between instances; "
                        f"unexpected data type in {file_path.name}."
                    )

                complete_writer.write_table(table)
                labelled = prepared.loc[prepared["class"].notna()]
                labelled_table = pa.Table.from_pandas(
                    labelled,
                    schema=output_schema,
                    preserve_index=False,
                )
                labelled_writer.write_table(labelled_table)

                row_count = len(prepared)
                labelled_count = int(prepared["class"].notna().sum())
                total_observations_read += row_count
                total_observations_retained += row_count
                labelled_observations += labelled_count
                unique_wells.add(well_id)
                instances_by_folder[event_folder] += 1

                class_values = prepared["class"]
                for class_code, count in class_values.value_counts(
                    dropna=False
                ).items():
                    if pd.isna(class_code):
                        unknown_class_count += int(count)
                    else:
                        class_counts[int(class_code)] += int(count)

                for feature in [*process_sensors, *equipment_indicators]:
                    missing_counts[feature] += int(
                        prepared[feature].isna().sum()
                    )

                if file_number % 100 == 0 or file_number == len(real_files):
                    print(
                        f"Processed {file_number:,}/{len(real_files):,} "
                        "real instances"
                    )

        finally:
            if complete_writer is not None:
                complete_writer.close()
            if labelled_writer is not None:
                labelled_writer.close()

        if output_schema is None or process_sensors is None:
            raise RuntimeError("No real instances were prepared.")

        temporary_complete_path.replace(self.complete_dataset_path)
        temporary_labelled_path.replace(self.labelled_dataset_path)
        self._write_report(
            instance_count=len(real_files),
            unique_well_count=len(unique_wells),
            total_observations_read=total_observations_read,
            total_observations_retained=total_observations_retained,
            labelled_observations=labelled_observations,
            unknown_class_count=unknown_class_count,
            class_counts=class_counts,
            process_sensors=process_sensors,
            equipment_indicators=equipment_indicators,
            missing_counts=missing_counts,
            instances_by_folder=instances_by_folder,
        )
        print(f"Complete dataset saved to: {self.complete_dataset_path}")
        print(f"Labelled modelling dataset saved to: {self.labelled_dataset_path}")
        print(f"Preparation report saved to: {self.report_path}")

    def _write_report(
        self,
        instance_count: int,
        unique_well_count: int,
        total_observations_read: int,
        total_observations_retained: int,
        labelled_observations: int,
        unknown_class_count: int,
        class_counts: Counter,
        process_sensors: list[str],
        equipment_indicators: list[str],
        missing_counts: dict,
        instances_by_folder: Counter,
    ) -> None:
        """Write preparation totals and per-feature availability."""
        lines = [
            "# Data Preparation Report",
            "",
            "## Dataset summary",
            "",
            "- Dataset: Petrobras 3W Dataset v2.0.0",
            "- Selection: real instances with the documented `WELL-` prefix",
            f"- Real instances processed: {instance_count:,}",
            f"- Unique wells represented: {unique_well_count:,}",
            f"- Total observations read: {total_observations_read:,}",
            (
                "- Observations retained in the complete dataset: "
                f"{total_observations_retained:,}"
            ),
            f"- Labelled observations (`class` known): {labelled_observations:,}",
            f"- Observations with unknown `class`: {unknown_class_count:,}",
            (
                "- Complete dataset: "
                f"`{self._display_path(self.complete_dataset_path)}`"
            ),
            (
                "- Labelled modelling view: "
                f"`{self._display_path(self.labelled_dataset_path)}`"
            ),
            "",
            "The complete dataset preserves the original timestamp order, "
            "sensor values, `class` and `state` labels, including unknown "
            "labels. The modelling view contains only observations with known "
            "`class` and retains all metadata and measurements. No values "
            "were imputed or filtered by feature availability. Class codes "
            "are categorical labels, not ordered or continuous measurements.",
            "",
            "## Class distribution",
            "",
            "| Class code | Condition | Observations |",
            "|---:|---|---:|",
        ]
        for class_code in sorted(class_counts):
            condition = self.CLASS_NAMES.get(class_code, "Unmapped code")
            lines.append(
                f"| {class_code} | {condition} | "
                f"{class_counts[class_code]:,} |"
            )
        lines.extend([
            "| Unknown (NaN) | Unknown condition | "
            f"{unknown_class_count:,} |",
            "",
            (
                f"Class counts include {labelled_observations:,} labelled "
                f"observations and {unknown_class_count:,} unknown labels."
            ),
            "",
            "## Feature availability",
            "",
            (
                "Availability is calculated over all retained real-instance "
                "observations. The two feature groups remain separate; no "
                "missingness threshold was applied."
            ),
            "",
            "| Feature group | Feature | Available | Missing | Availability | Missingness |",
            "|---|---|---:|---:|---:|---:|",
        ])

        for group_name, features in [
            ("Process sensor measurement", process_sensors),
            ("Equipment-state indicator", equipment_indicators),
        ]:
            for feature in features:
                missing = missing_counts[feature]
                available = total_observations_retained - missing
                availability_percentage = (
                    available / total_observations_retained * 100
                    if total_observations_retained
                    else 0
                )
                missing_percentage = (
                    missing / total_observations_retained * 100
                    if total_observations_retained
                    else 0
                )
                lines.append(
                    f"| {group_name} | `{feature}` | {available:,} | "
                    f"{missing:,} | {availability_percentage:.2f}% | "
                    f"{missing_percentage:.2f}% |"
                )

        lines.extend([
            "",
            "## Real instances by event folder",
            "",
            "| Event folder | Instances |",
            "|---:|---:|",
        ])
        for event_folder in sorted(
            instances_by_folder,
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        ):
            lines.append(
                f"| {event_folder} | {instances_by_folder[event_folder]:,} |"
            )

        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.name
