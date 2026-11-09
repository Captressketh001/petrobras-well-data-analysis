import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class DataCleaner:
    """Clean sensor columns and run focused data investigations."""

    BATCH_SIZE = 65_536
    SUSPICIOUS_SENSOR_COLUMNS = ("P-PDG", "P-JUS-CKP", "T-PDG")
    NON_SENSOR_COLUMNS = frozenset({
        "timestamp",
        "instance_id",
        "file_name",
        "well_id",
        "event_folder",
        "state",
        "class",
    })

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
            Path(reports_dir)
            if reports_dir is not None
            else self.project_root / "reports"
        )
        self.input_dataset_path = (
            self.processed_dir / "real_instances_complete.parquet"
        )
        self.cleaned_dataset_path = (
            self.processed_dir / "real_instances_cleaned.parquet"
        )
        self.report_path = (
            self.reports_dir / "cleaning" / "cleaning_report.md"
        )

    def run_all(self) -> None:
        """Run all registered cleaning analyses."""
        for analysis_name in self.ANALYSES:
            print(f"Running data cleaning: {analysis_name}")
            getattr(self, analysis_name)()

    ANALYSES = (
        "clean_dataset",
        "investigate_p_pdg_extreme_values",
        "investigate_other_sensor_extreme_values",
    )

    def clean_dataset(self) -> None:
        """Remove empty sensors and report checks without altering values."""
        parquet_file = pq.ParquetFile(self.input_dataset_path)
        input_columns = parquet_file.schema_arrow.names
        required_columns = self.NON_SENSOR_COLUMNS
        missing_columns = required_columns.difference(input_columns)
        if missing_columns:
            raise ValueError(
                "The complete dataset is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        sensor_columns = [
            column
            for column in input_columns
            if column not in self.NON_SENSOR_COLUMNS
        ]
        if not sensor_columns:
            raise ValueError("The complete dataset contains no sensor columns.")

        diagnostics = self._inspect_dataset(
            parquet_file,
            input_columns,
            sensor_columns,
        )
        valid_counts = diagnostics["valid_counts"]

        removed_columns = [
            column for column in sensor_columns if valid_counts[column] == 0
        ]
        output_columns = [
            column for column in input_columns if column not in removed_columns
        ]
        input_schema = parquet_file.schema_arrow
        schema_metadata = input_schema.metadata
        if schema_metadata is not None and b"pandas" in schema_metadata:
            pandas_metadata = json.loads(schema_metadata[b"pandas"])
            pandas_metadata["columns"] = [
                column
                for column in pandas_metadata.get("columns", [])
                if column.get("field_name") in output_columns
            ]
            schema_metadata = {
                **schema_metadata,
                b"pandas": json.dumps(pandas_metadata).encode("utf-8"),
            }
        output_schema = pa.schema(
            [input_schema.field(column) for column in output_columns],
            metadata=schema_metadata,
        )
        temporary_output_path = self.cleaned_dataset_path.with_suffix(
            ".parquet.tmp"
        )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        rows_retained = 0
        writer = pq.ParquetWriter(
            temporary_output_path,
            output_schema,
            compression="brotli",
        )
        try:
            for batch in parquet_file.iter_batches(
                columns=output_columns,
                batch_size=self.BATCH_SIZE,
            ):
                writer.write_table(
                    pa.Table.from_batches(
                        [batch],
                        schema=output_schema,
                    )
                )
                rows_retained += batch.num_rows
        finally:
            writer.close()

        rows_before = parquet_file.metadata.num_rows
        if rows_retained != rows_before:
            raise RuntimeError(
                "Cleaning changed the row count: "
                f"{rows_before:,} rows before, {rows_retained:,} after."
            )

        temporary_output_path.replace(self.cleaned_dataset_path)
        self._write_cleaning_report(
            output_columns=output_columns,
            removed_columns=removed_columns,
            rows_before=rows_before,
            rows_after=rows_retained,
            diagnostics=diagnostics,
        )
        print(f"Cleaned dataset saved to: {self.cleaned_dataset_path}")
        print(f"Data cleaning report saved to: {self.report_path}")

    def _inspect_dataset(
        self,
        parquet_file: pq.ParquetFile,
        input_columns: list[str],
        sensor_columns: list[str],
    ) -> dict:
        """Collect schema, missingness, duplicate-row, and anomaly findings."""
        schema = parquet_file.schema_arrow
        schema_issues = self._find_invalid_types(schema, sensor_columns)
        missing_counts = dict.fromkeys(input_columns, 0)
        valid_counts = dict.fromkeys(sensor_columns, 0)
        duplicate_rows = 0
        repeated_timestamp_pairs = 0
        timestamp_order_violations = 0
        repeated_instance_groups = 0
        duplicate_check_possible = (
            pa.types.is_timestamp(schema.field("timestamp").type)
            and (
                pa.types.is_string(schema.field("instance_id").type)
                or pa.types.is_large_string(
                    schema.field("instance_id").type
                )
            )
        )
        seen_instances: set[str] = set()
        previous_instance = None
        previous_timestamp = None
        previous_row = None
        suspicious_minima = {
            column: {
                "value": None,
                "file_counts": Counter(),
            }
            for column in self.SUSPICIOUS_SENSOR_COLUMNS
            if column in sensor_columns
        }
        timestamp_index = input_columns.index("timestamp")
        instance_index = input_columns.index("instance_id")
        file_name_index = input_columns.index("file_name")

        for batch in parquet_file.iter_batches(
            columns=input_columns,
            batch_size=self.BATCH_SIZE,
        ):
            for column_name, array in zip(batch.schema.names, batch.columns):
                nan_count = 0
                if pa.types.is_floating(array.type):
                    nan_count = int(
                        pc.sum(
                            pc.fill_null(pc.is_nan(array), False)
                        ).as_py()
                        or 0
                    )
                missing_count = array.null_count + nan_count
                missing_counts[column_name] += missing_count
                if column_name in valid_counts:
                    valid_counts[column_name] += len(array) - missing_count

            if duplicate_check_possible and batch.num_rows:
                batch_instances = np.asarray(
                    batch.column(instance_index).to_pylist(),
                    dtype=object,
                )
                batch_timestamps = batch.column(
                    timestamp_index
                ).to_numpy(zero_copy_only=False)
                if previous_instance == batch_instances[0]:
                    if batch_timestamps[0] < previous_timestamp:
                        timestamp_order_violations += 1
                    if batch_timestamps[0] == previous_timestamp:
                        repeated_timestamp_pairs += 1
                        if self._rows_equal(
                            previous_row,
                            self._row_values(batch, 0),
                        ):
                            duplicate_rows += 1

                if batch.num_rows > 1:
                    same_instance = batch_instances[1:] == batch_instances[:-1]
                    same_timestamp = (
                        batch_timestamps[1:] == batch_timestamps[:-1]
                    )
                    timestamp_order_violations += int(
                        np.count_nonzero(
                            same_instance
                            & (
                                batch_timestamps[1:]
                                < batch_timestamps[:-1]
                            )
                        )
                    )
                    duplicate_candidates = np.flatnonzero(
                        same_instance & same_timestamp
                    ) + 1
                    repeated_timestamp_pairs += len(duplicate_candidates)
                    for row_index in duplicate_candidates:
                        if self._rows_equal(
                            self._row_values(batch, int(row_index) - 1),
                            self._row_values(batch, int(row_index)),
                        ):
                            duplicate_rows += 1

                    group_starts = (
                        np.flatnonzero(
                            batch_instances[1:] != batch_instances[:-1]
                        )
                        + 1
                    )
                else:
                    group_starts = np.array([], dtype=np.int64)

                if previous_instance != batch_instances[0]:
                    if batch_instances[0] in seen_instances:
                        repeated_instance_groups += 1
                    seen_instances.add(batch_instances[0])
                for group_start in group_starts:
                    instance_id = batch_instances[int(group_start)]
                    if instance_id in seen_instances:
                        repeated_instance_groups += 1
                    seen_instances.add(instance_id)

                previous_instance = batch_instances[-1]
                previous_timestamp = batch_timestamps[-1]
                previous_row = self._row_values(
                    batch,
                    batch.num_rows - 1,
                )

            file_names = batch.column(file_name_index).to_pylist()
            for sensor_name, minimum_data in suspicious_minima.items():
                sensor_type = schema.field(sensor_name).type
                if not (
                    pa.types.is_floating(sensor_type)
                    or pa.types.is_integer(sensor_type)
                ):
                    continue
                sensor_array = batch.column(
                    batch.schema.get_field_index(sensor_name)
                ).to_numpy(zero_copy_only=False)
                non_nan_mask = ~np.isnan(sensor_array)
                if not np.any(non_nan_mask):
                    continue
                local_minimum = float(np.min(sensor_array[non_nan_mask]))
                current_minimum = minimum_data["value"]
                if (
                    current_minimum is None
                    or local_minimum < current_minimum
                ):
                    minimum_data["value"] = local_minimum
                    minimum_data["file_counts"].clear()
                if local_minimum == minimum_data["value"]:
                    matching_rows = np.flatnonzero(
                        non_nan_mask & (sensor_array == local_minimum)
                    )
                    for row_index in matching_rows:
                        minimum_data["file_counts"][
                            file_names[int(row_index)]
                        ] += 1

        return {
            "schema_issues": schema_issues,
            "missing_counts": missing_counts,
            "valid_counts": valid_counts,
            "duplicate_rows": duplicate_rows,
            "repeated_timestamp_pairs": repeated_timestamp_pairs,
            "duplicate_check_possible": duplicate_check_possible,
            "timestamp_order_violations": timestamp_order_violations,
            "repeated_instance_groups": repeated_instance_groups,
            "suspicious_minima": suspicious_minima,
        }

    @staticmethod
    def _find_invalid_types(
        schema: pa.Schema,
        sensor_columns: list[str],
    ) -> list[tuple[str, pa.DataType, str]]:
        """Check field types against the prepared dataset's semantic roles."""
        expected_metadata = {
            "instance_id",
            "file_name",
            "well_id",
            "event_folder",
        }
        issues = []
        for field in schema:
            if field.name == "timestamp":
                valid = pa.types.is_timestamp(field.type)
                expected = "timestamp"
            elif field.name in expected_metadata:
                valid = (
                    pa.types.is_string(field.type)
                    or pa.types.is_large_string(field.type)
                )
                expected = "string"
            elif field.name in {"class", "state"}:
                valid = pa.types.is_integer(field.type)
                expected = "integer label"
            elif field.name in sensor_columns:
                valid = (
                    pa.types.is_floating(field.type)
                    or pa.types.is_integer(field.type)
                )
                expected = "numeric sensor"
            else:
                valid = True
                expected = "any"
            if not valid:
                issues.append((field.name, field.type, expected))
        return issues

    @staticmethod
    def _row_values(batch: pa.RecordBatch, index: int) -> list:
        return [column[index].as_py() for column in batch.columns]

    @staticmethod
    def _rows_equal(left: list, right: list) -> bool:
        for left_value, right_value in zip(left, right):
            if (
                isinstance(left_value, float)
                and isinstance(right_value, float)
                and math.isnan(left_value)
                and math.isnan(right_value)
            ):
                continue
            if left_value != right_value:
                return False
        return True

    def _write_cleaning_report(
        self,
        *,
        output_columns: list[str],
        removed_columns: list[str],
        rows_before: int,
        rows_after: int,
        diagnostics: dict,
    ) -> None:
        """Document the cleaning step and findings without hiding concerns."""
        lines = [
            "# Data Cleaning Report",
            "",
            "## Operations performed",
            "",
            "- Removed sensor columns with no valid observations.",
            "- Preserved all remaining values without imputation, coercion, "
            "or outlier replacement.",
            "- Preserved row order, timestamps, labels, and instance metadata.",
            "",
            "## Columns removed",
            "",
        ]
        if removed_columns:
            lines.extend(f"- `{column}`" for column in removed_columns)
        else:
            lines.append("- None")
        lines.extend([
            "",
            "## Rows",
            "",
            f"- Rows before cleaning: {rows_before:,}",
            f"- Rows after cleaning: {rows_after:,}",
            f"- Rows removed: {rows_before - rows_after:,}",
            "",
            "## Missing values preserved",
            "",
            "Null/NaN values in retained columns are preserved as missing "
            "values; no imputation or row filtering was performed.",
            "",
            "| Column | Missing values |",
            "|---|---:|",
        ])
        for column in output_columns:
            missing_count = diagnostics["missing_counts"][column]
            if missing_count:
                lines.append(f"| `{column}` | {missing_count:,} |")
        if not any(
            diagnostics["missing_counts"][column]
            for column in output_columns
        ):
            lines.append("| None | 0 |")

        lines.extend([
            "",
            "## Duplicate-row check",
            "",
        ])
        if not diagnostics["duplicate_check_possible"]:
            lines.append(
                "Could not verify exact duplicate rows because timestamp "
                "and instance identifier types are not suitable for the "
                "ordered-row check."
            )
        else:
            lines.append(
                f"- Exact duplicate rows found: "
                f"{diagnostics['duplicate_rows']:,}."
            )
            lines.append(
                f"- Repeated instance/timestamp key pairs: "
                f"{diagnostics['repeated_timestamp_pairs']:,}."
            )
            lines.append(
                f"- Timestamps out of order within an instance: "
                f"{diagnostics['timestamp_order_violations']:,}."
            )
            lines.append(
                f"- Instance groups that reappear after another instance: "
                f"{diagnostics['repeated_instance_groups']:,}."
            )
            if diagnostics["timestamp_order_violations"] or (
                diagnostics["repeated_instance_groups"]
                or diagnostics["repeated_timestamp_pairs"]
            ):
                lines.append(
                    "The duplicate count covers exact adjacent matches. "
                    "Repeated timestamps or ordering issues prevent ruling "
                    "out non-adjacent duplicates within a timestamp group."
                )
            elif diagnostics["duplicate_rows"] == 0:
                lines.append(
                    "Rows are grouped by instance and timestamps are "
                    "chronological, so exact duplicate rows would be "
                    "adjacent; none were found."
                )
            lines.append(
                "No duplicate rows were deleted."
            )

        lines.extend([
            "",
            "## Data types",
            "",
        ])
        if diagnostics["schema_issues"]:
            lines.append(
                "Unexpected data types were found; they were reported and "
                "left unchanged:"
            )
            for (
                column,
                actual_type,
                expected_type,
            ) in diagnostics["schema_issues"]:
                lines.append(
                    f"- `{column}`: `{actual_type}` "
                    f"(expected {expected_type})."
                )
        else:
            lines.append(
                "No invalid data types were found. Timestamps, metadata, "
                "labels, and sensor columns retain their source types; no "
                "conversions were performed."
            )

        lines.extend([
            "",
            "## Suspicious values retained for further consideration",
            "",
            "Previously investigated extreme negative readings remain "
            "unchanged. Their physical meaning has not been confirmed, so "
            "they are documented as unresolved rather than treated as "
            "invalid or replaced.",
            "",
        ])
        for column, minimum_data in diagnostics["suspicious_minima"].items():
            value = minimum_data["value"]
            files = minimum_data["file_counts"]
            if value is None:
                lines.append(f"- `{column}`: no valid values were found.")
                continue
            file_summary = ", ".join(
                f"`{file_name}` ({count:,})"
                for file_name, count in sorted(files.items())
            )
            lines.append(
                f"- `{column}` minimum `{value:.17g}` occurred "
                f"{sum(files.values()):,} time(s)"
                + (f" in {file_summary}." if file_summary else ".")
            )
        lines.extend([
            "",
            "Related investigations: "
            "`reports/cleaning/p_pdg_extreme_values.md` and "
            "`reports/cleaning/other_sensor_extreme_values.md`.",
            "",
            "## Remaining data-quality concerns",
            "",
            "- Missing sensor measurements and unknown labels remain in the "
            "dataset by design.",
            "- The suspicious extreme readings above require domain or source "
            "documentation before they can be classified.",
            "- This step removes only completely empty sensor columns; it "
            "does not filter rows or perform further cleaning.",
            "",
            (
                "Cleaned dataset: "
                f"`{self._display_path(self.cleaned_dataset_path)}`"
            ),
            (
                "Source dataset was not modified: "
                f"`{self._display_path(self.input_dataset_path)}`"
            ),
        ])
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.name

    def investigate_p_pdg_extreme_values(self) -> None:
        """Summarize P-PDG and report the most extreme distinct negatives."""
        parquet_file = pq.ParquetFile(self.cleaned_dataset_path)
        required_columns = {"P-PDG", "file_name", "instance_id"}
        missing_columns = required_columns.difference(
            parquet_file.schema_arrow.names
        )
        if missing_columns:
            raise ValueError(
                "The cleaned dataset is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        sensor_type = parquet_file.schema_arrow.field("P-PDG").type
        if not (
            pa.types.is_floating(sensor_type)
            or pa.types.is_integer(sensor_type)
        ):
            raise TypeError(
                f"Expected a numeric P-PDG column; got {sensor_type}."
            )

        values = parquet_file.read(columns=["P-PDG"]).column(0)
        if pa.types.is_floating(sensor_type):
            nan_mask = pc.fill_null(pc.is_nan(values), False)
            valid_mask = pc.and_(
                pc.is_valid(values),
                pc.invert(nan_mask),
            )
        else:
            valid_mask = pc.is_valid(values)
        valid_values = pc.filter(values, valid_mask)

        valid_count = len(valid_values)
        missing_count = parquet_file.metadata.num_rows - valid_count
        zero_count = self._count_matching(valid_values, 0)
        negative_count = self._count_matching(
            valid_values,
            pc.less(valid_values, 0),
        )
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        if valid_count:
            minimum = pc.min(valid_values).as_py()
            maximum = pc.max(valid_values).as_py()
            percentile_values = pc.quantile(
                valid_values,
                q=[percentile / 100 for percentile in percentiles],
                interpolation="linear",
            ).to_pylist()
        else:
            minimum = maximum = None
            percentile_values = [None] * len(percentiles)

        extreme_values: list[float] = []
        if negative_count:
            for batch in parquet_file.iter_batches(
                columns=["P-PDG"],
                batch_size=self.BATCH_SIZE,
            ):
                batch_values = batch.column(0).to_numpy(
                    zero_copy_only=False
                )
                negative_values = np.unique(batch_values[batch_values < 0])
                extreme_values = sorted(
                    {*extreme_values, *negative_values.tolist()}
                )[:20]

        value_file_counts: Counter[tuple[float, str]] = Counter()
        value_file_instances: dict[tuple[float, str], str] = {}
        if extreme_values:
            extreme_value_array = np.array(extreme_values)
            for batch in parquet_file.iter_batches(
                columns=["P-PDG", "file_name", "instance_id"],
                batch_size=self.BATCH_SIZE,
            ):
                batch_values = batch.column(0).to_numpy(
                    zero_copy_only=False
                )
                matching_indices = np.flatnonzero(
                    np.isin(batch_values, extreme_value_array)
                )
                file_names = batch.column(1)
                instance_ids = batch.column(2)
                for row_index in matching_indices:
                    value = float(batch_values[row_index])
                    file_name = file_names[row_index].as_py()
                    instance_id = instance_ids[row_index].as_py()
                    key = (value, file_name)
                    value_file_counts[key] += 1
                    value_file_instances[key] = instance_id

        report_dir = self.reports_dir / "cleaning"
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = report_dir / "p_pdg_extreme_values.csv"
        summary_path = report_dir / "p_pdg_extreme_values.md"
        self._write_p_pdg_extremes_csv(
            csv_path,
            extreme_values,
            value_file_counts,
            value_file_instances,
        )
        self._write_p_pdg_extremes_summary(
            summary_path,
            valid_count=valid_count,
            missing_count=missing_count,
            zero_count=zero_count,
            negative_count=negative_count,
            minimum=minimum,
            maximum=maximum,
            percentiles=percentiles,
            percentile_values=percentile_values,
            extreme_values=extreme_values,
            value_file_counts=value_file_counts,
        )
        print(f"P-PDG extreme values saved to: {csv_path}")
        print(f"P-PDG summary saved to: {summary_path}")

    def investigate_other_sensor_extreme_values(self) -> None:
        """Report extreme negatives in P-JUS-CKP and T-PDG."""
        parquet_file = pq.ParquetFile(self.cleaned_dataset_path)
        sensors = ["P-JUS-CKP", "T-PDG"]
        required_columns = {*sensors, "file_name"}
        missing_columns = required_columns.difference(
            parquet_file.schema_arrow.names
        )
        if missing_columns:
            raise ValueError(
                "The cleaned dataset is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )

        sensor_results = {}
        for sensor in sensors:
            sensor_type = parquet_file.schema_arrow.field(sensor).type
            if not (
                pa.types.is_floating(sensor_type)
                or pa.types.is_integer(sensor_type)
            ):
                raise TypeError(
                    f"Expected a numeric {sensor} column; got {sensor_type}."
                )

            values = parquet_file.read(columns=[sensor]).column(0)
            if pa.types.is_floating(sensor_type):
                nan_mask = pc.fill_null(pc.is_nan(values), False)
                valid_mask = pc.and_(
                    pc.is_valid(values),
                    pc.invert(nan_mask),
                )
            else:
                valid_mask = pc.is_valid(values)
            valid_values = pc.filter(values, valid_mask)
            negative_count = self._count_matching(
                valid_values,
                pc.less(valid_values, 0),
            )
            percentile_values = pc.quantile(
                valid_values,
                q=[0.01, 0.05, 0.5, 0.99],
                interpolation="linear",
            ).to_pylist() if len(valid_values) else [None] * 4

            extreme_values: list[float] = []
            if negative_count:
                for batch in parquet_file.iter_batches(
                    columns=[sensor],
                    batch_size=self.BATCH_SIZE,
                ):
                    batch_values = batch.column(0).to_numpy(
                        zero_copy_only=False
                    )
                    negative_values = np.unique(
                        batch_values[batch_values < 0]
                    )
                    extreme_values = sorted(
                        {*extreme_values, *negative_values.tolist()}
                    )[:20]

            sensor_results[sensor] = {
                "negative_count": negative_count,
                "minimum": (
                    pc.min(valid_values).as_py()
                    if len(valid_values)
                    else None
                ),
                "maximum": (
                    pc.max(valid_values).as_py()
                    if len(valid_values)
                    else None
                ),
                "percentiles": percentile_values,
                "extreme_values": extreme_values,
                "file_counts": Counter(),
            }

        extreme_sets = {
            sensor: np.array(result["extreme_values"])
            for sensor, result in sensor_results.items()
        }
        for batch in parquet_file.iter_batches(
            columns=[*sensors, "file_name"],
            batch_size=self.BATCH_SIZE,
        ):
            file_names = batch.column(2)
            for sensor_index, sensor in enumerate(sensors):
                extreme_values = extreme_sets[sensor]
                if not len(extreme_values):
                    continue
                batch_values = batch.column(sensor_index).to_numpy(
                    zero_copy_only=False
                )
                matching_indices = np.flatnonzero(
                    np.isin(batch_values, extreme_values)
                )
                file_counts = sensor_results[sensor]["file_counts"]
                for row_index in matching_indices:
                    value = float(batch_values[row_index])
                    file_name = file_names[row_index].as_py()
                    file_counts[(value, file_name)] += 1

        report_path = (
            self.reports_dir
            / "cleaning"
            / "other_sensor_extreme_values.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_other_sensor_extremes_summary(
            report_path,
            sensor_results,
        )
        print(f"Other sensor extreme values saved to: {report_path}")

    @staticmethod
    def _write_other_sensor_extremes_summary(
        report_path: Path,
        sensor_results: dict,
    ) -> None:
        """Write statistics and per-file occurrence counts for each sensor."""
        lines = [
            "# P-JUS-CKP and T-PDG Extreme Negative Values",
            "",
            "Source: `data/processed/real_instances_cleaned.parquet`.",
            "",
            "The 20 most extreme entries are the 20 lowest distinct negative "
            "values per sensor. Data values were not modified.",
        ]
        percentiles = [1, 5, 50, 99]
        for sensor, result in sensor_results.items():
            lines.extend([
                "",
                f"## {sensor}",
                "",
                f"- Negative values: {result['negative_count']:,}",
                "",
                "| Statistic | Value |",
                "|---|---:|",
                (
                    f"| Minimum | "
                    f"{DataCleaner._format_statistic(result['minimum'])} |"
                ),
                *[
                    f"| {DataCleaner._ordinal(percentile)} percentile | "
                    f"{DataCleaner._format_statistic(value)} |"
                    for percentile, value in zip(
                        percentiles,
                        result["percentiles"],
                    )
                ],
                (
                    f"| Maximum | "
                    f"{DataCleaner._format_statistic(result['maximum'])} |"
                ),
                "",
                "| Rank | Value | Occurrences | Affected instance files "
                "(occurrences per file) |",
                "|---:|---:|---:|---|",
            ])
            for rank, value in enumerate(
                result["extreme_values"],
                start=1,
            ):
                file_counts = sorted(
                    (
                        (file_name, count)
                        for (entry_value, file_name), count
                        in result["file_counts"].items()
                        if entry_value == value
                    ),
                    key=lambda entry: entry[0],
                )
                total_count = sum(count for _, count in file_counts)
                files_text = ", ".join(
                    f"`{file_name}` ({count:,})"
                    for file_name, count in file_counts
                )
                lines.append(
                    f"| {rank} | {value:.17g} | {total_count:,} | "
                    f"{files_text or 'None'} |"
                )
            if not result["extreme_values"]:
                lines.append("| - | None | 0 | None |")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _count_matching(values: pa.Array | pa.ChunkedArray, predicate) -> int:
        """Count values matching an Arrow compute predicate."""
        if isinstance(predicate, int):
            predicate = pc.equal(values, predicate)
        return int(pc.sum(pc.cast(predicate, pa.int64())).as_py() or 0)

    @staticmethod
    def _write_p_pdg_extremes_csv(
        csv_path: Path,
        extreme_values: list[float],
        value_file_counts: Counter[tuple[float, str]],
        value_file_instances: dict[tuple[float, str], str],
    ) -> None:
        """Write one row per extreme value and instance file."""
        value_totals = Counter()
        for (value, _), count in value_file_counts.items():
            value_totals[value] += count
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "rank",
                    "value",
                    "dataset_occurrences",
                    "file_name",
                    "instance_id",
                    "file_occurrences",
                ],
            )
            writer.writeheader()
            for rank, value in enumerate(extreme_values, start=1):
                file_entries = sorted(
                    (
                        (file_name, count)
                        for (entry_value, file_name), count
                        in value_file_counts.items()
                        if entry_value == value
                    ),
                    key=lambda entry: entry[0],
                )
                for file_name, count in file_entries:
                    key = (value, file_name)
                    writer.writerow({
                        "rank": rank,
                        "value": format(value, ".17g"),
                        "dataset_occurrences": value_totals[value],
                        "file_name": file_name,
                        "instance_id": value_file_instances[key],
                        "file_occurrences": count,
                    })

    @staticmethod
    def _write_p_pdg_extremes_summary(
        summary_path: Path,
        *,
        valid_count: int,
        missing_count: int,
        zero_count: int,
        negative_count: int,
        minimum: float | None,
        maximum: float | None,
        percentiles: list[int],
        percentile_values: list[float | None],
        extreme_values: list[float],
        value_file_counts: Counter[tuple[float, str]],
    ) -> None:
        """Write summary statistics and point to per-file extreme counts."""
        lines = [
            "# P-PDG Extreme-Value Investigation",
            "",
            "## Value counts",
            "",
            f"- Valid: {valid_count:,}",
            f"- Missing: {missing_count:,}",
            f"- Zero: {zero_count:,}",
            f"- Negative: {negative_count:,}",
            "",
            "## Distribution",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Minimum | {DataCleaner._format_statistic(minimum)} |",
            *[
                f"| {DataCleaner._ordinal(percentile)} percentile | "
                f"{DataCleaner._format_statistic(value)} |"
                for percentile, value in zip(percentiles, percentile_values)
            ],
            f"| Maximum | {DataCleaner._format_statistic(maximum)} |",
            "",
            "## Most extreme distinct negative values",
            "",
        ]
        if not extreme_values:
            lines.append("No negative values were found.")
        else:
            lines.extend([
                "| Rank | Value | Dataset occurrences | Instance files |",
                "|---:|---:|---:|---:|",
            ])
            for rank, value in enumerate(extreme_values, start=1):
                file_counts = {
                    file_name: count
                    for (entry_value, file_name), count
                    in value_file_counts.items()
                    if entry_value == value
                }
                total_count = sum(file_counts.values())
                lines.append(
                    f"| {rank} | {value:.17g} | {total_count:,} | "
                    f"{len(file_counts):,} |"
                )
        lines.extend([
            "",
            "The CSV lists each extreme value by instance file, with both "
            "dataset-wide and per-file occurrence counts. Values were not "
            "modified.",
        ])
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _format_statistic(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.17g}"

    @staticmethod
    def _ordinal(value: int) -> str:
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"
