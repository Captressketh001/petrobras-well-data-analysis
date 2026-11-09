from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data_loader import get_parquet_files, load_parquet_file


class DataAnalyzer:
    """Run the project's data-quality and performance analyses."""

    ANALYSES = (
        "explore_dataset",
        "profile_labels",
        "summarize_label_distribution",
        "summarise_overall_label_distributions",
        "summarize_recording_durations",
        "analyze_missing_label_intervals",
        "check_label_alignment",
        "profile_instance_sources",
        "profile_all_files",
        "profile_sensor_missingness",
        "summarize_sensor_coverage",
        "profile_sensor_availability_by_source",
        "verify_missing_labels_by_source",
    )

    CLASS_LABELS = list(range(10)) + list(range(101, 110))

    SENSOR_VARIABLES = [
        "ABER-CKGL",
        "ABER-CKP",
        "ESTADO-DHSV",
        "ESTADO-M1",
        "ESTADO-M2",
        "ESTADO-PXO",
        "ESTADO-SDV-GL",
        "ESTADO-SDV-P",
        "ESTADO-W1",
        "ESTADO-W2",
        "ESTADO-XO",
        "P-ANULAR",
        "P-JUS-BS",
        "P-JUS-CKGL",
        "P-JUS-CKP",
        "P-MON-CKGL",
        "P-MON-CKP",
        "P-MON-SDV-P",
        "P-PDG",
        "PT-P",
        "P-TPT",
        "QBS",
        "QGL",
        "T-JUS-CKP",
        "T-MON-CKP",
        "T-PDG",
        "T-TPT",
    ]

    EXPECTED_COUNTS = {
        "0": {"real": 594, "simulated": 0, "hand_drawn": 0},
        "1": {"real": 4, "simulated": 114, "hand_drawn": 10},
        "2": {"real": 22, "simulated": 16, "hand_drawn": 0},
        "3": {"real": 32, "simulated": 74, "hand_drawn": 0},
        "4": {"real": 343, "simulated": 0, "hand_drawn": 0},
        "5": {"real": 11, "simulated": 439, "hand_drawn": 0},
        "6": {"real": 6, "simulated": 215, "hand_drawn": 0},
        "7": {"real": 36, "simulated": 0, "hand_drawn": 10},
        "8": {"real": 14, "simulated": 81, "hand_drawn": 0},
        "9": {"real": 57, "simulated": 150, "hand_drawn": 0},
    }

    def __init__(self, reports_dir: Path | None = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.reports_dir = (
            Path(reports_dir) if reports_dir is not None
            else self.project_root / "reports"
        )

    def run_all(self) -> None:
        """Run all analyses in report-dependency order."""
        for analysis_name in self.ANALYSES:
            print(f"\nRunning analysis: {analysis_name}")
            getattr(self, analysis_name)()

    @staticmethod
    def classify_source(filename: str) -> str:
        """Identify the source type from the official filename prefix."""
        if filename.startswith("WELL-"):
            return "real"
        if filename.startswith("SIMULATED_"):
            return "simulated"
        if filename.startswith("DRAWN_"):
            return "hand_drawn"
        return "unknown"

    @staticmethod
    def get_filename(value) -> str:
        """Extract the filename from either a filename or a full path."""
        return Path(str(value)).name

    @staticmethod
    def find_filename_column(dataframe: pd.DataFrame) -> str:
        """Find a column containing filenames or file paths."""
        candidates = ["file_name", "filename", "file_path", "filepath", "path"]

        for column in candidates:
            if column in dataframe.columns:
                return column

        raise ValueError(
            "Could not identify a filename column. "
            f"Available columns: {list(dataframe.columns)}"
        )

    @staticmethod
    def get_well_id(file_path: Path) -> str:
        """Extract the well identifier from the filename."""
        return file_path.stem.split("_")[0]

    @staticmethod
    def update_statistics(
        stats: dict,
        key: tuple,
        variable: str,
        missing_count: int,
        total_rows: int,
    ) -> None:
        stat_key = (*key, variable)

        if stat_key not in stats:
            stats[stat_key] = {
                "total_observations": 0,
                "missing_count": 0,
                "entirely_missing_files": 0,
                "file_count": 0,
            }

        stats[stat_key]["total_observations"] += total_rows
        stats[stat_key]["missing_count"] += missing_count
        stats[stat_key]["file_count"] += 1

        if missing_count == total_rows:
            stats[stat_key]["entirely_missing_files"] += 1

    @staticmethod
    def convert_to_dataframe(stats: dict, grouping_columns: list[str]) -> pd.DataFrame:
        rows = []

        for key, values in stats.items():
            *group_values, variable = key

            total = values["total_observations"]
            missing = values["missing_count"]

            missing_percentage = (missing / total) * 100

            row = dict(zip(grouping_columns, group_values))

            row.update({
                "variable": variable,
                "file_count": values["file_count"],
                "total_observations": total,
                "missing_count": missing,
                "missing_percentage": missing_percentage,
                "availability_percentage": 100 - missing_percentage,
                "entirely_missing_files": values["entirely_missing_files"],
            })

            rows.append(row)

        return pd.DataFrame(rows)

    def explore_dataset(self) -> None:
        parquet_files = get_parquet_files()

        print(f"Total Parquet files: {len(parquet_files)}")

        sample_file = parquet_files[0]

        print(f"\nSample file: {sample_file}")
        print(f"Event folder: {sample_file.parent.name}")

        df = load_parquet_file(sample_file)

        print("\nDataset dimensions:")
        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nFirst five observations:")
        print(df.head())

        print("\nMissing values:")
        print(df.isna().sum())

        print("\nMISSING VALUE PERCENTAGES")
        missing_summary = pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_percentage": df.isna().mean() * 100,
        })
        missing_summary = missing_summary.sort_values(
            by="missing_percentage",
            ascending=False,
        )
        print(missing_summary.round(2))

        print("\nCLASS DISTRIBUTION")
        print(df["class"].value_counts(dropna=False).sort_index())

        print("\nSTATE DISTRIBUTION")
        print(df["state"].value_counts(dropna=False).sort_index())

        print("\nTIMESTAMP CHECKS")
        timestamps = df.index
        print("Timestamp data type:", timestamps.dtype)
        print("Duplicate timestamps:", timestamps.duplicated().sum())
        print("Chronologically ordered:", timestamps.is_monotonic_increasing)
        print("First timestamp:", timestamps.min())
        print("Last timestamp:", timestamps.max())
        print("\nTimestamp intervals:")
        print(timestamps.to_series().diff().value_counts().head(10))

        print("\nENTIRELY MISSING COLUMNS")
        empty_columns = df.columns[df.isna().all()].tolist()
        print(empty_columns)
        print("Number of entirely missing columns:", len(empty_columns))

        print("\nMISSING LABEL INVESTIGATION")
        class_missing = df["class"].isna()
        state_missing = df["state"].isna()
        print(
            "Class and state missing at identical timestamps:",
            class_missing.equals(state_missing),
        )
        segment_id = class_missing.ne(class_missing.shift()).cumsum()
        segments = df.groupby(segment_id).agg(
            start_time=("class", lambda x: x.index[0]),
            end_time=("class", lambda x: x.index[-1]),
            observations=("class", "size"),
            labels_missing=("class", lambda x: x.isna().all()),
        )
        print("\nLabel segments:")
        print(segments.to_string(index=False))

        print("\nRows around missing-label transitions:")
        transition_points = class_missing.ne(class_missing.shift()).fillna(True)
        transition_indices = df.index[transition_points]

        for timestamp in transition_indices:
            position = df.index.get_loc(timestamp)
            start = max(0, position - 2)
            end = min(len(df), position + 3)
            print(f"\nAround {timestamp}:")
            print(df.iloc[start:end][["class", "state"]])

    def summarise_overall_label_distributions(self) -> None:
        input_file = self.reports_dir / "label_distribution_by_folder.csv"
        output_file = self.reports_dir / "overall_label_distributions.csv"
        df = pd.read_csv(input_file)

        summary = (
            df.groupby(["variable", "label"], dropna=False)["count"]
            .sum()
            .reset_index()
        )
        summary["percentage"] = (
            summary["count"]
            / summary.groupby("variable")["count"].transform("sum")
        ) * 100
        summary = summary.sort_values(
            ["variable", "count"],
            ascending=[True, False],
        )
        summary.to_csv(output_file, index=False)

        for variable in ["class", "state"]:
            print(f"\nOVERALL {variable.upper()} DISTRIBUTION")
            print("-" * 40)
            print(
                summary[summary["variable"] == variable].to_string(
                    index=False,
                    formatters={"percentage": "{:.2f}%".format},
                )
            )

        print(f"\nReport saved to: {output_file}")

    def profile_instance_sources(self) -> None:
        print("Loading existing file-level reports...")

        recording_df = pd.read_csv(
            self.reports_dir / "recording_summary_by_file.csv"
        )
        missing_df = pd.read_csv(
            self.reports_dir / "missing_label_summary_by_file.csv"
        )

        recording_filename_column = self.find_filename_column(recording_df)
        missing_filename_column = self.find_filename_column(missing_df)

        recording_df["file_name"] = recording_df[
            recording_filename_column
        ].apply(self.get_filename)
        recording_df["source_type"] = recording_df["file_name"].apply(
            self.classify_source
        )

        if "event_folder" not in recording_df.columns:
            raise ValueError(
                "The recording summary does not contain an 'event_folder' column. "
                f"Available columns: {list(recording_df.columns)}"
            )

        recording_df["event_folder"] = (
            recording_df["event_folder"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )
        missing_df["file_name"] = missing_df[
            missing_filename_column
        ].apply(self.get_filename)
        files_with_missing_labels = set(missing_df["file_name"])
        recording_df["has_missing_labels"] = recording_df["file_name"].isin(
            files_with_missing_labels
        )

        enriched_path = (
            self.reports_dir / "recording_summary_by_file_with_source.csv"
        )
        recording_df.to_csv(enriched_path, index=False)

        source_summary = (
            recording_df.groupby(
                ["event_folder", "source_type"], dropna=False
            )
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for source in ["real", "simulated", "hand_drawn", "unknown"]:
            if source not in source_summary.columns:
                source_summary[source] = 0
        source_summary["total_files"] = source_summary[
            ["real", "simulated", "hand_drawn", "unknown"]
        ].sum(axis=1)

        source_summary_path = self.reports_dir / "instance_source_summary.csv"
        source_summary.to_csv(source_summary_path, index=False)

        missing_by_source = (
            recording_df.groupby("source_type")
            .agg(
                total_files=("file_name", "count"),
                files_with_missing_labels=("has_missing_labels", "sum"),
            )
            .reset_index()
        )
        missing_by_source["percentage_with_missing_labels"] = (
            missing_by_source["files_with_missing_labels"]
            / missing_by_source["total_files"]
            * 100
        )
        missing_by_source_path = (
            self.reports_dir / "missing_labels_by_source.csv"
        )
        missing_by_source.to_csv(missing_by_source_path, index=False)

        observed = (
            recording_df.groupby(["event_folder", "source_type"])
            .size()
            .to_dict()
        )
        comparison_rows = []
        for folder, expected_sources in self.EXPECTED_COUNTS.items():
            for source, expected_count in expected_sources.items():
                actual_count = observed.get((folder, source), 0)
                comparison_rows.append({
                    "event_folder": folder,
                    "source_type": source,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "matches_documentation": actual_count == expected_count,
                })
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = self.reports_dir / "source_count_validation.csv"
        comparison_df.to_csv(comparison_path, index=False)

        print("\nSOURCE COUNTS BY EVENT FOLDER")
        print("-" * 60)
        print(source_summary.to_string(index=False))
        print("\nMISSING LABELS BY SOURCE")
        print("-" * 60)
        print(missing_by_source.to_string(index=False))
        print("\nVALIDATION AGAINST OFFICIAL DOCUMENTATION")
        print("-" * 60)
        print(comparison_df.to_string(index=False))
        print("\nVALIDATION SUMMARY")
        print("-" * 60)
        print(
            "All official source counts match:",
            comparison_df["matches_documentation"].all(),
        )
        print(
            "Files with unknown source:",
            (recording_df["source_type"] == "unknown").sum(),
        )
        print(
            "Total files with missing labels:",
            recording_df["has_missing_labels"].sum(),
        )
        print(
            "Real files with missing labels:",
            recording_df.loc[
                recording_df["source_type"] == "real",
                "has_missing_labels",
            ].sum(),
        )
        print(
            "Non-real files with missing labels:",
            recording_df.loc[
                recording_df["source_type"].isin(["simulated", "hand_drawn"]),
                "has_missing_labels",
            ].sum(),
        )
        print("\nReports saved:")
        print(enriched_path)
        print(source_summary_path)
        print(missing_by_source_path)
        print(comparison_path)

    def analyse_file(self, file_path: Path) -> tuple[list, dict]:
        """Identify missing intervals and summarise one recording."""
        df = load_parquet_file(file_path)

        if df.empty:
            raise ValueError(f"Empty recording: {file_path.name}")

        class_missing = df["class"].isna()
        state_missing = df["state"].isna()
        if not class_missing.equals(state_missing):
            raise ValueError(
                f"Class/state missingness mismatch: {file_path.name}"
            )

        first_timestamp = df.index[0]
        last_timestamp = df.index[-1]
        group_ids = (
            class_missing.ne(class_missing.shift(fill_value=False)).cumsum()
        )
        missing_groups = group_ids[class_missing].unique()
        intervals = []

        for interval_number, group_id in enumerate(missing_groups, start=1):
            interval_mask = class_missing & group_ids.eq(group_id)
            interval_timestamps = df.index[interval_mask]
            start_time = interval_timestamps[0]
            end_time = interval_timestamps[-1]
            duration_seconds = (end_time - start_time).total_seconds() + 1
            start_offset_seconds = (
                start_time - first_timestamp
            ).total_seconds()
            end_offset_seconds = (end_time - first_timestamp).total_seconds()
            position = (
                "beginning"
                if start_time == first_timestamp
                else "end"
                if end_time == last_timestamp
                else "middle"
            )

            intervals.append({
                "event_folder": file_path.parent.name,
                "file_name": file_path.name,
                "well_id": file_path.stem.split("_")[0],
                "interval_number": interval_number,
                "interval_start": start_time,
                "interval_end": end_time,
                "duration_seconds": duration_seconds,
                "duration_minutes": duration_seconds / 60,
                "duration_hours": duration_seconds / 3600,
                "position_in_recording": position,
                "start_offset_minutes": start_offset_seconds / 60,
                "end_offset_minutes": end_offset_seconds / 60,
                "is_first_interval": interval_number == 1,
                "is_last_interval": interval_number == len(missing_groups),
            })

        total_missing = int(class_missing.sum())
        summary = {
            "event_folder": file_path.parent.name,
            "file_name": file_path.name,
            "well_id": file_path.stem.split("_")[0],
            "recording_start": first_timestamp,
            "recording_end": last_timestamp,
            "total_observations": len(df),
            "missing_observations": total_missing,
            "missing_percentage": total_missing / len(df) * 100,
            "missing_interval_count": len(intervals),
            "total_missing_duration_hours": total_missing / 3600,
            "first_missing_start": (
                intervals[0]["interval_start"] if intervals else pd.NaT
            ),
            "last_missing_end": (
                intervals[-1]["interval_end"] if intervals else pd.NaT
            ),
        }
        return intervals, summary

    @staticmethod
    def build_pattern_report(intervals_df: pd.DataFrame) -> pd.DataFrame:
        """Compare missing interval patterns across folders and wells."""
        if intervals_df.empty:
            return pd.DataFrame()

        grouping_columns = [
            "event_folder",
            "interval_number",
            "position_in_recording",
        ]
        folder_patterns = (
            intervals_df.groupby(grouping_columns)
            .agg(
                recording_count=("file_name", "nunique"),
                unique_wells=("well_id", "nunique"),
                median_start_offset_minutes=("start_offset_minutes", "median"),
                min_start_offset_minutes=("start_offset_minutes", "min"),
                max_start_offset_minutes=("start_offset_minutes", "max"),
                median_duration_minutes=("duration_minutes", "median"),
                min_duration_minutes=("duration_minutes", "min"),
                max_duration_minutes=("duration_minutes", "max"),
            )
            .reset_index()
        )
        folder_patterns.insert(0, "comparison_level", "folder")
        folder_patterns.insert(2, "well_id", "ALL")
        well_patterns = (
            intervals_df.groupby(
                [
                    "event_folder",
                    "well_id",
                    "interval_number",
                    "position_in_recording",
                ]
            )
            .agg(
                recording_count=("file_name", "nunique"),
                unique_wells=("well_id", "nunique"),
                median_start_offset_minutes=("start_offset_minutes", "median"),
                min_start_offset_minutes=("start_offset_minutes", "min"),
                max_start_offset_minutes=("start_offset_minutes", "max"),
                median_duration_minutes=("duration_minutes", "median"),
                min_duration_minutes=("duration_minutes", "min"),
                max_duration_minutes=("duration_minutes", "max"),
            )
            .reset_index()
        )
        well_patterns.insert(0, "comparison_level", "well")
        return pd.concat(
            [folder_patterns, well_patterns],
            ignore_index=True,
        )

    def analyze_missing_label_intervals(self) -> None:
        parquet_files = get_parquet_files()
        all_intervals = []
        all_summaries = []

        print(f"Total Parquet files: {len(parquet_files)}")
        print("Analysing missing-label intervals and patterns...")

        for index, file_path in enumerate(parquet_files, start=1):
            intervals, summary = self.analyse_file(file_path)
            all_intervals.extend(intervals)
            all_summaries.append(summary)
            if index % 100 == 0:
                print(f"Processed {index}/{len(parquet_files)} files")

        intervals_df = pd.DataFrame(all_intervals)
        summaries_df = pd.DataFrame(all_summaries)
        patterns_df = self.build_pattern_report(intervals_df)

        intervals_path = self.reports_dir / "missing_label_intervals.csv"
        summaries_path = self.reports_dir / "missing_label_summary_by_file.csv"
        patterns_path = (
            self.reports_dir / "missing_label_position_patterns.csv"
        )
        intervals_df.to_csv(intervals_path, index=False)
        summaries_df.to_csv(summaries_path, index=False)
        patterns_df.to_csv(patterns_path, index=False)

        print("\nAnalysis completed successfully.")
        print(f"\nDetailed intervals: {intervals_path}")
        print(f"File summaries: {summaries_path}")
        print(f"Position patterns: {patterns_path}")
        print("\nOverall missing-label statistics:")
        print(f"Total files: {len(summaries_df)}")
        print(
            "Files with missing labels:",
            (summaries_df["missing_observations"] > 0).sum(),
        )
        print(
            "Total missing observations:",
            summaries_df["missing_observations"].sum(),
        )
        print("Total continuous missing intervals:", len(intervals_df))

        if not intervals_df.empty:
            print("\nMissing interval positions:")
            print(
                intervals_df["position_in_recording"]
                .value_counts()
                .to_string()
            )
            print("\nMissing interval durations (minutes):")
            print(intervals_df["duration_minutes"].describe().to_string())
            print("\nFolder-level pattern comparison:")
            print(
                patterns_df[
                    patterns_df["comparison_level"] == "folder"
                ].to_string(index=False)
            )

    def check_label_alignment(self) -> None:
        parquet_files = get_parquet_files()
        records = []
        print(f"Checking {len(parquet_files):,} Parquet files...\n")

        for number, file_path in enumerate(parquet_files, start=1):
            df = load_parquet_file(file_path)
            class_missing = df["class"].isna()
            state_missing = df["state"].isna()
            both_missing = class_missing & state_missing
            class_only = class_missing & ~state_missing
            state_only = state_missing & ~class_missing
            neither_missing = ~class_missing & ~state_missing
            records.append({
                "event_folder": file_path.parent.name,
                "file_name": file_path.name,
                "total_rows": len(df),
                "both_missing": int(both_missing.sum()),
                "class_only_missing": int(class_only.sum()),
                "state_only_missing": int(state_only.sum()),
                "neither_missing": int(neither_missing.sum()),
                "mismatch_count": int(class_only.sum() + state_only.sum()),
                "missing_patterns_match": class_missing.equals(state_missing),
            })
            if number % 100 == 0:
                print(f"Processed {number:,}/{len(parquet_files):,} files")

        file_report = pd.DataFrame(records)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        file_output = (
            self.reports_dir / "label_missingness_alignment_by_file.csv"
        )
        file_report.to_csv(file_output, index=False)

        folder_summary = (
            file_report.groupby("event_folder")
            .agg(
                file_count=("file_name", "count"),
                total_rows=("total_rows", "sum"),
                both_missing=("both_missing", "sum"),
                class_only_missing=("class_only_missing", "sum"),
                state_only_missing=("state_only_missing", "sum"),
                neither_missing=("neither_missing", "sum"),
                total_mismatches=("mismatch_count", "sum"),
                files_with_mismatches=(
                    "missing_patterns_match",
                    lambda x: (~x).sum(),
                ),
            )
            .reset_index()
        )
        folder_output = (
            self.reports_dir / "label_missingness_alignment_by_folder.csv"
        )
        folder_summary.to_csv(folder_output, index=False)

        total_mismatches = file_report["mismatch_count"].sum()
        aligned_files = file_report["missing_patterns_match"].sum()
        print("\nOVERALL LABEL MISSINGNESS ALIGNMENT")
        print("----------------------------------")
        print(f"Total files checked: {len(file_report):,}")
        print(f"Files with identical missing patterns: {aligned_files:,}")
        print(
            f"Files with different missing patterns: "
            f"{len(file_report) - aligned_files:,}"
        )
        print(
            f"Class-only missing timestamps: "
            f"{file_report['class_only_missing'].sum():,}"
        )
        print(
            f"State-only missing timestamps: "
            f"{file_report['state_only_missing'].sum():,}"
        )
        print(f"Total mismatched timestamps: {total_mismatches:,}")
        print("\nSUMMARY BY EVENT FOLDER")
        print(folder_summary.to_string(index=False))
        print(f"\nFile-level report saved to: {file_output}")
        print(f"Folder summary saved to: {folder_output}")

    def summarize_label_distribution(self) -> None:
        input_file = self.reports_dir / "label_distribution_by_folder.csv"
        df = pd.read_csv(input_file)
        totals = (
            df.groupby(["event_folder", "variable"])["count"]
            .transform("sum")
        )
        df["percentage"] = (df["count"] / totals) * 100

        output_file = self.reports_dir / "label_distribution_percentages.csv"
        df.to_csv(output_file, index=False)

        class_summary = df[df["variable"] == "class"]
        print("\nCLASS DISTRIBUTION BY EVENT FOLDER")
        print("----------------------------------")
        print(
            class_summary.to_string(
                index=False,
                formatters={"percentage": "{:.2f}%".format},
            )
        )

        state_summary = df[df["variable"] == "state"]
        print("\nSTATE DISTRIBUTION BY EVENT FOLDER")
        print("----------------------------------")
        print(
            state_summary.to_string(
                index=False,
                formatters={"percentage": "{:.2f}%".format},
            )
        )
        print(f"\nReport saved to: {output_file}")

    def profile_labels(self) -> None:
        parquet_files = get_parquet_files()
        records = []
        for number, file_path in enumerate(parquet_files, start=1):
            df = load_parquet_file(file_path)
            folder = file_path.parent.name
            for column in ["class", "state"]:
                counts = df[column].value_counts(dropna=False)
                for label, count in counts.items():
                    records.append({
                        "event_folder": folder,
                        "variable": column,
                        "label": (
                            "MISSING" if pd.isna(label) else int(label)
                        ),
                        "count": int(count),
                    })
            if number % 100 == 0:
                print(f"Processed {number}/{len(parquet_files)} files")

        report = pd.DataFrame(records)
        summary = (
            report.groupby(
                ["event_folder", "variable", "label"],
                dropna=False,
            )["count"]
            .sum()
            .reset_index()
        )
        output_path = self.reports_dir / "label_distribution_by_folder.csv"
        summary.to_csv(output_path, index=False)
        print("\nLABEL DISTRIBUTION BY EVENT FOLDER")
        print(summary.to_string(index=False))
        print(f"\nReport saved to: {output_path}")

    def summarize_recording_durations(self) -> None:
        parquet_files = get_parquet_files()
        print(f"Total Parquet files: {len(parquet_files)}")
        print("Profiling recording durations and class composition...")
        records = []

        for index, file_path in enumerate(parquet_files, start=1):
            records.append(self.profile_recording(file_path))
            if index % 100 == 0:
                print(f"Processed {index}/{len(parquet_files)} files")

        file_summary = pd.DataFrame(records)
        file_output = self.reports_dir / "recording_summary_by_file.csv"
        file_summary.to_csv(file_output, index=False)
        folder_summary = (
            file_summary.groupby("event_folder")
            .agg(
                file_count=("file_name", "count"),
                total_observations=("total_rows", "sum"),
                mean_recording_hours=("recording_duration_hours", "mean"),
                median_recording_hours=("recording_duration_hours", "median"),
                min_recording_hours=("recording_duration_hours", "min"),
                max_recording_hours=("recording_duration_hours", "max"),
                mean_recording_rows=("total_rows", "mean"),
                median_recording_rows=("total_rows", "median"),
            )
            .reset_index()
        )
        class_count_columns = [
            f"class_{label}_count" for label in self.CLASS_LABELS
        ]
        class_totals = (
            file_summary.groupby("event_folder")[class_count_columns]
            .sum()
            .reset_index()
        )
        folder_summary = folder_summary.merge(
            class_totals,
            on="event_folder",
        )
        for label in self.CLASS_LABELS:
            count_column = f"class_{label}_count"
            folder_summary[f"class_{label}_pct"] = (
                folder_summary[count_column]
                / folder_summary["total_observations"]
                * 100
            )
        folder_summary["class_missing_count"] = (
            file_summary.groupby("event_folder")["class_missing_count"]
            .sum()
            .values
        )
        folder_summary["class_missing_pct"] = (
            folder_summary["class_missing_count"]
            / folder_summary["total_observations"]
            * 100
        )
        folder_output = self.reports_dir / "recording_summary_by_folder.csv"
        folder_summary.to_csv(folder_output, index=False)
        print("\nRecording summary completed.")
        print(f"File-level report: {file_output}")
        print(f"Folder-level report: {folder_output}")
        print("\nRecording summary by folder:")
        print(
            folder_summary[
                [
                    "event_folder",
                    "file_count",
                    "total_observations",
                    "mean_recording_hours",
                    "median_recording_hours",
                    "min_recording_hours",
                    "max_recording_hours",
                ]
            ].to_string(index=False)
        )

    def profile_recording(self, file_path: Path) -> dict:
        """Summarise recording duration and class composition for one file."""
        df = load_parquet_file(file_path)
        total_rows = len(df)
        start_timestamp = df.index.min()
        end_timestamp = df.index.max()
        elapsed_seconds = (end_timestamp - start_timestamp).total_seconds()
        recording_seconds = elapsed_seconds + 1 if total_rows else 0
        class_counts = df["class"].value_counts(dropna=False)
        result = {
            "event_folder": file_path.parent.name,
            "file_name": file_path.name,
            "well_id": self.get_well_id(file_path),
            "total_rows": total_rows,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "elapsed_duration_seconds": elapsed_seconds,
            "recording_duration_seconds": recording_seconds,
            "recording_duration_hours": recording_seconds / 3600,
            "class_missing_count": int(df["class"].isna().sum()),
        }
        for label in self.CLASS_LABELS:
            count = int(class_counts.get(label, 0))
            result[f"class_{label}_count"] = count
            result[f"class_{label}_pct"] = (
                count / total_rows * 100 if total_rows else 0
            )
        return result

    def profile_sensor_availability_by_source(self) -> None:
        parquet_files = get_parquet_files()
        print(f"Total Parquet files: {len(parquet_files)}")
        print("Starting sensor availability profiling by source...")
        source_stats = {}
        folder_source_stats = {}

        for index, file_path in enumerate(parquet_files, start=1):
            filename = file_path.name
            source_type = self.classify_source(filename)
            event_folder = file_path.parent.name
            if source_type == "unknown":
                print(f"WARNING: Unknown source: {filename}")
                continue
            df = load_parquet_file(file_path)
            total_rows = len(df)
            missing_counts = df[self.SENSOR_VARIABLES].isna().sum()
            for variable in self.SENSOR_VARIABLES:
                missing_count = int(missing_counts[variable])
                self.update_statistics(
                    source_stats,
                    (source_type,),
                    variable,
                    missing_count,
                    total_rows,
                )
                self.update_statistics(
                    folder_source_stats,
                    (event_folder, source_type),
                    variable,
                    missing_count,
                    total_rows,
                )
            if index % 100 == 0 or index == len(parquet_files):
                print(f"Processed {index}/{len(parquet_files)} files")

        source_df = self.convert_to_dataframe(
            source_stats,
            ["source_type"],
        )
        folder_source_df = self.convert_to_dataframe(
            folder_source_stats,
            ["event_folder", "source_type"],
        )
        source_df.to_csv(
            self.reports_dir / "sensor_availability_by_source.csv",
            index=False,
        )
        folder_source_df.to_csv(
            self.reports_dir / "sensor_availability_by_folder_and_source.csv",
            index=False,
        )
        print("\nSENSOR AVAILABILITY BY SOURCE")
        print("-" * 70)
        for source in ["real", "simulated", "hand_drawn"]:
            print(f"\n{source.upper()}")
            result = source_df[
                source_df["source_type"] == source
            ].sort_values("availability_percentage")
            print(
                result[
                    [
                        "variable",
                        "availability_percentage",
                        "entirely_missing_files",
                    ]
                ].to_string(index=False)
            )
        print("\nReports saved:")
        print("reports/sensor_availability_by_source.csv")
        print("reports/sensor_availability_by_folder_and_source.csv")

    def profile_sensor_missingness(self) -> None:
        parquet_files = get_parquet_files()
        missing_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(lambda: defaultdict(int))
        entirely_missing_files = defaultdict(lambda: defaultdict(int))

        for number, file_path in enumerate(parquet_files, start=1):
            df = load_parquet_file(file_path)
            folder = file_path.parent.name
            for column in df.columns:
                missing_counts[folder][column] += df[column].isna().sum()
                total_counts[folder][column] += len(df)
                if df[column].isna().all():
                    entirely_missing_files[folder][column] += 1
            if number % 100 == 0:
                print(f"Processed {number}/{len(parquet_files)} files")

        records = []
        for folder in sorted(missing_counts):
            for column in missing_counts[folder]:
                missing = missing_counts[folder][column]
                total = total_counts[folder][column]
                records.append({
                    "event_folder": folder,
                    "variable": column,
                    "missing_count": missing,
                    "total_observations": total,
                    "missing_percentage": (missing / total) * 100,
                    "entirely_missing_files": (
                        entirely_missing_files[folder][column]
                    ),
                })
        report = pd.DataFrame(records)
        output_path = (
            self.reports_dir / "sensor_missingness_by_folder.csv"
        )
        report.to_csv(output_path, index=False)
        sensor_report = report[
            ~report["variable"].isin(["class", "state"])
        ]
        print("\nSENSOR MISSINGNESS BY EVENT FOLDER")
        for folder in sorted(sensor_report["event_folder"].unique()):
            folder_data = sensor_report[
                sensor_report["event_folder"] == folder
            ].sort_values("missing_percentage", ascending=False)
            print(f"\nEvent folder: {folder}")
            print(
                folder_data[
                    [
                        "variable",
                        "missing_percentage",
                        "entirely_missing_files",
                    ]
                ].round(2).to_string(index=False)
            )
        print(f"\nDetailed report saved to: {output_path}")

    def summarize_sensor_coverage(self) -> None:
        report_path = self.reports_dir / "sensor_missingness_by_folder.csv"
        report = pd.read_csv(report_path)
        sensor_report = report[
            ~report["variable"].isin(["class", "state"])
        ].copy()
        sensor_report["availability_percentage"] = (
            100 - sensor_report["missing_percentage"]
        )
        coverage_matrix = sensor_report.pivot(
            index="variable",
            columns="event_folder",
            values="availability_percentage",
        )
        coverage_matrix = coverage_matrix.sort_index()
        print("\nSENSOR AVAILABILITY MATRIX (%)")
        print(coverage_matrix.round(2).to_string())
        overall = (
            sensor_report.groupby("variable")
            .agg(
                total_missing=("missing_count", "sum"),
                total_observations=("total_observations", "sum"),
            )
        )
        overall["missing_percentage"] = (
            overall["total_missing"]
            / overall["total_observations"]
            * 100
        )
        overall["availability_percentage"] = (
            100 - overall["missing_percentage"]
        )
        overall = overall.sort_values(
            "missing_percentage",
            ascending=False,
        )
        print("\nOVERALL SENSOR AVAILABILITY")
        print(overall.round(2).to_string())
        coverage_matrix.to_csv(
            self.reports_dir / "sensor_coverage_matrix.csv"
        )
        overall.to_csv(
            self.reports_dir / "overall_sensor_availability.csv"
        )
        print("\nReports successfully saved.")

    def verify_missing_labels_by_source(self) -> None:
        recordings = pd.read_csv(
            self.reports_dir / "recording_summary_by_file.csv"
        )
        missing = pd.read_csv(
            self.reports_dir / "missing_label_summary_by_file.csv"
        )
        recordings["source_type"] = recordings["file_name"].apply(
            self.classify_source
        )
        missing_files = set(
            missing.loc[
                missing["missing_observations"] > 0,
                "file_name",
            ]
        )
        recordings["has_missing_labels"] = recordings["file_name"].isin(
            missing_files
        )
        source_counts = (
            recordings.groupby("source_type")
            .agg(
                total_files=("file_name", "count"),
                files_with_missing_labels=("has_missing_labels", "sum"),
            )
            .reset_index()
        )
        source_counts["files_without_missing_labels"] = (
            source_counts["total_files"]
            - source_counts["files_with_missing_labels"]
        )
        folder_source_counts = (
            recordings.groupby(["event_folder", "source_type"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        missing_source_counts = (
            recordings[recordings["has_missing_labels"]]["source_type"]
            .value_counts()
        )
        real_files = recordings[recordings["source_type"] == "real"]
        unmatched_missing_files = missing_files - set(recordings["file_name"])

        print("\nSOURCE COUNTS")
        print("-" * 60)
        print(source_counts.to_string(index=False))
        print("\nSOURCE COUNTS BY EVENT FOLDER")
        print("-" * 60)
        print(folder_source_counts.to_string(index=False))
        print("\nMISSING-LABEL FILES BY SOURCE")
        print("-" * 60)
        print(missing_source_counts.to_string())
        print("\nVERIFICATION")
        print("-" * 60)
        print("Total recording files:", len(recordings))
        print("Total missing-label files:", len(missing_files))
        print("Real instances:", len(real_files))
        print(
            "Real instances with missing labels:",
            real_files["has_missing_labels"].sum(),
        )
        print(
            "Real instances without missing labels:",
            (~real_files["has_missing_labels"]).sum(),
        )
        print(
            "Non-real instances with missing labels:",
            recordings.loc[
                recordings["source_type"].isin(["simulated", "hand_drawn"]),
                "has_missing_labels",
            ].sum(),
        )
        print(
            "Unknown source files:",
            (recordings["source_type"] == "unknown").sum(),
        )
        print("Unmatched filenames:", len(unmatched_missing_files))
        source_counts.to_csv(
            self.reports_dir / "missing_label_source_verification.csv",
            index=False,
        )
        folder_source_counts.to_csv(
            self.reports_dir / "source_counts_by_event_folder.csv",
            index=False,
        )
        print("\nVerification reports saved.")

    def profile_all_files(self) -> None:
        parquet_files = get_parquet_files()
        print(f"Profiling {len(parquet_files)} Parquet files...")
        file_summaries = []
        folder_summaries = defaultdict(list)

        for number, file_path in enumerate(parquet_files, start=1):
            df = load_parquet_file(file_path)
            timestamps = df.index
            missing = df.isna().mean() * 100
            intervals = timestamps.to_series().diff()
            summary = {
                "file": file_path.name,
                "event_folder": file_path.parent.name,
                "rows": len(df),
                "columns": len(df.columns),
                "start_time": timestamps.min(),
                "end_time": timestamps.max(),
                "duplicate_timestamps": timestamps.duplicated().sum(),
                "chronologically_ordered": timestamps.is_monotonic_increasing,
                "non_1_second_intervals": (
                    (intervals.dropna() != pd.Timedelta(seconds=1)).sum()
                ),
                "class_missing_pct": missing.get("class", 0),
                "state_missing_pct": missing.get("state", 0),
                "entirely_missing_columns": ";".join(
                    df.columns[df.isna().all()].tolist()
                ),
            }
            file_summaries.append(summary)
            folder_summaries[file_path.parent.name].append(summary)
            if number % 100 == 0:
                print(f"Processed {number}/{len(parquet_files)} files")

        report = pd.DataFrame(file_summaries)
        output_path = self.reports_dir / "file_quality_summary.csv"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
        print("\nSUMMARY BY EVENT FOLDER")
        folder_report = report.groupby("event_folder").agg(
            file_count=("file", "count"),
            total_rows=("rows", "sum"),
            average_class_missing_pct=("class_missing_pct", "mean"),
            average_state_missing_pct=("state_missing_pct", "mean"),
            files_with_duplicate_timestamps=(
                "duplicate_timestamps",
                lambda x: (x > 0).sum(),
            ),
            files_with_timestamp_gaps=(
                "non_1_second_intervals",
                lambda x: (x > 0).sum(),
            ),
        )
        print(folder_report.round(2).to_string())
        print(f"\nDetailed report saved to: {output_path}")
