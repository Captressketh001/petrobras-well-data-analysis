from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.data_loader import get_parquet_files, load_parquet_file


class EDAAnalyzer:
    """Run exploratory analyses on real production instances."""

    ANALYSES = (
        "analyse_sensor_conditions",
        "analyse_sensor_value_quality",
        "analyse_sensor_operational_states",
        "analyse_event_patterns",
        "analyse_operational_patterns",
    )

    VISUALIZATIONS = (
        "visualize_sensor_condition_boxplots",
        "visualize_sensor_correlation_heatmap",
        "visualize_sensor_measurements_by_state",
        "visualize_event_phase_durations",
        "visualize_operational_state_distribution",
    )

    VISUALIZED_SENSORS = ("P-ANULAR", "P-TPT", "T-TPT")
    PHASE_ORDER = {
        "Normal Operation": 0,
        "Steady-state event": 1,
        "Transient": 2,
    }
    PHASE_COLORS = {
        "Normal Operation": "#4C78A8",
        "Steady-state event": "#F58518",
        "Transient": "#54A24B",
    }
    MIN_CORRELATION_PAIRS = 3

    EVENT_NAMES = {
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
    }

    STATE_NAMES = {
        0: "Open",
        1: "Shut-In",
        2: "Flushing Diesel",
        3: "Flushing Gas",
        4: "Bullheading",
        5: "Closed With Diesel",
        6: "Closed With Gas",
        7: "Restart",
        8: "Depressurization",
    }

    def __init__(self, reports_dir: Path | None = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.reports_dir = (
            Path(reports_dir) if reports_dir is not None
            else self.project_root / "reports"
        )
        self.eda_reports_dir = self.reports_dir / "eda"

    def run_all(self) -> None:
        """Run all exploratory analyses."""
        for analysis_name in self.ANALYSES:
            print(f"Running EDA: {analysis_name}")
            getattr(self, analysis_name)()

    def run_visualizations(self) -> None:
        """Run all registered EDA visualizations."""
        for visualization_name in self.VISUALIZATIONS:
            print(f"Running visualization: {visualization_name}")
            getattr(self, visualization_name)()

    def visualize_sensor_condition_boxplots(self) -> None:
        """Plot per-instance sensor medians by production condition."""
        input_file = (
            self.eda_reports_dir / "sensor_measurements_by_condition.csv"
        )
        if not input_file.exists():
            print(f"EDA results not found: {input_file}")
            return

        required_columns = {
            "file_name",
            "class_code",
            "condition_phase",
            "event_name",
            "sensor_variable",
            "median",
        }
        data = (
            pd.DataFrame(columns=sorted(required_columns))
            if input_file.stat().st_size == 0
            else pd.read_csv(input_file)
        )
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            print(
                "Cannot create condition boxplots; input is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
            return

        self.eda_reports_dir.joinpath("figures").mkdir(
            parents=True,
            exist_ok=True,
        )
        output_dir = self.eda_reports_dir / "figures"
        data["median"] = pd.to_numeric(data["median"], errors="coerce")
        data["class_code"] = pd.to_numeric(
            data["class_code"],
            errors="coerce",
        )

        for sensor in self.VISUALIZED_SENSORS:
            sensor_data = data.loc[
                data["sensor_variable"] == sensor,
                [
                    "file_name",
                    "class_code",
                    "condition_phase",
                    "event_name",
                    "median",
                ],
            ].dropna(
                subset=[
                    "file_name",
                    "class_code",
                    "condition_phase",
                    "event_name",
                    "median",
                ]
            )
            per_instance = (
                sensor_data.groupby(
                    [
                        "file_name",
                        "class_code",
                        "condition_phase",
                        "event_name",
                    ],
                    as_index=False,
                    dropna=False,
                )["median"]
                .median()
            )
            conditions = (
                per_instance.groupby(
                    ["class_code", "condition_phase", "event_name"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    instance_count=("file_name", "nunique"),
                    values=("median", list),
                )
            )
            conditions["_phase_order"] = (
                conditions["condition_phase"]
                .map(self.PHASE_ORDER)
                .fillna(len(self.PHASE_ORDER))
            )
            conditions = conditions.sort_values(
                ["_phase_order", "class_code", "event_name"]
            )
            figure_width = max(12, len(conditions) * 1.15)
            figure, axis = plt.subplots(figsize=(figure_width, 7))

            if conditions.empty:
                axis.text(
                    0.5,
                    0.5,
                    f"No usable per-instance condition data for {sensor}.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            else:
                labels = [
                    (
                        f"{row.condition_phase}\n{row.event_name} "
                        f"(class {int(row.class_code)}, n={row.instance_count})"
                    )
                    for row in conditions.itertuples()
                ]
                boxplot = axis.boxplot(
                    conditions["values"].tolist(),
                    tick_labels=labels,
                    patch_artist=True,
                    showfliers=False,
                )
                plotted_values = [
                    value
                    for values in conditions["values"]
                    for value in values
                ]
                if plotted_values and min(plotted_values) > 0:
                    value_range_ratio = (
                        max(plotted_values) / min(plotted_values)
                    )
                    if value_range_ratio >= 100:
                        axis.set_yscale("log")

                for box, phase in zip(
                    boxplot["boxes"],
                    conditions["condition_phase"],
                ):
                    box.set_facecolor(
                        self.PHASE_COLORS.get(phase, "#999999")
                    )
                    box.set_alpha(0.75)

                phases_present = conditions[
                    "condition_phase"
                ].drop_duplicates()
                axis.legend(
                    handles=[
                        Patch(
                            facecolor=self.PHASE_COLORS.get(
                                phase,
                                "#999999",
                            ),
                            label=phase,
                        )
                        for phase in phases_present
                    ],
                    title="Condition phase",
                    loc="best",
                )
                axis.set_xticklabels(labels, rotation=55, ha="right")

            axis.set_title(
                f"{sensor}: Per-instance median across production conditions"
            )
            axis.set_xlabel(
                "Production condition (phase, event, class, instances)"
            )
            axis.set_ylabel(f"{sensor} per-instance median")
            figure.tight_layout()
            output_file = (
                output_dir
                / f"{sensor.lower().replace('-', '_')}_condition_boxplot.png"
            )
            figure.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {output_file}")

    def visualize_sensor_correlation_heatmap(self) -> None:
        """Plot pairwise process-sensor correlations across instances."""
        input_file = (
            self.eda_reports_dir / "sensor_measurements_by_condition.csv"
        )
        if not input_file.exists():
            print(f"EDA results not found: {input_file}")
            return

        required_columns = {
            "file_name",
            "class_code",
            "sensor_variable",
            "median",
        }
        data = (
            pd.DataFrame(columns=sorted(required_columns))
            if input_file.stat().st_size == 0
            else pd.read_csv(input_file)
        )
        missing_columns = required_columns - set(data.columns)
        output_dir = self.eda_reports_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(12, 10))

        if missing_columns:
            message = (
                "Input is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
            axis.text(0.5, 0.5, message, ha="center", va="center")
            axis.set_axis_off()
        else:
            process_data = data.loc[
                ~data["sensor_variable"].astype(str).str.startswith("ESTADO-"),
                ["file_name", "class_code", "sensor_variable", "median"],
            ].copy()
            process_data["class_code"] = pd.to_numeric(
                process_data["class_code"],
                errors="coerce",
            )
            process_data["median"] = pd.to_numeric(
                process_data["median"],
                errors="coerce",
            )
            process_data = process_data.dropna(
                subset=["file_name", "class_code", "sensor_variable", "median"]
            )
            process_data = process_data[
                np.isfinite(process_data["median"])
            ]

            if process_data.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No process-sensor medians are available.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            else:
                per_instance = (
                    process_data.groupby(
                        ["file_name", "class_code", "sensor_variable"],
                        as_index=False,
                    )["median"]
                    .median()
                )
                sensor_values = per_instance.pivot(
                    index=["file_name", "class_code"],
                    columns="sensor_variable",
                    values="median",
                )
                sensor_values = sensor_values.replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                pair_counts = sensor_values.notna().astype(int).T.dot(
                    sensor_values.notna().astype(int)
                )
                sensors = [
                    sensor
                    for sensor in sensor_values.columns
                    if any(
                        pair_counts.loc[sensor, other]
                        >= self.MIN_CORRELATION_PAIRS
                        for other in sensor_values.columns
                        if other != sensor
                    )
                ]

                if len(sensors) < 2:
                    axis.text(
                        0.5,
                        0.5,
                        "Insufficient paired process-sensor medians "
                        "to calculate correlations.",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                    axis.set_axis_off()
                else:
                    sensor_values = sensor_values[sensors]
                    pair_counts = pair_counts.loc[sensors, sensors]
                    correlations = sensor_values.corr(
                        method="pearson",
                        min_periods=self.MIN_CORRELATION_PAIRS,
                    )
                    correlations = correlations.where(
                        pair_counts >= self.MIN_CORRELATION_PAIRS
                    )
                    correlation_values = correlations.to_numpy(dtype=float)
                    masked_values = np.ma.masked_invalid(correlation_values)
                    image = axis.imshow(
                        masked_values,
                        cmap="coolwarm",
                        vmin=-1,
                        vmax=1,
                        aspect="auto",
                    )
                    colorbar = figure.colorbar(image, ax=axis)
                    colorbar.set_label("Pearson correlation")
                    axis.set_xticks(range(len(sensors)), labels=sensors)
                    axis.set_yticks(range(len(sensors)), labels=sensors)
                    axis.tick_params(axis="x", labelrotation=55)
                    for label in axis.get_xticklabels():
                        label.set_ha("right")

                    for row in range(len(sensors)):
                        for column in range(len(sensors)):
                            correlation = correlation_values[row, column]
                            if np.isfinite(correlation):
                                axis.text(
                                    column,
                                    row,
                                    (
                                        f"{correlation:.2f}\n"
                                        f"n={pair_counts.iloc[row, column]}"
                                    ),
                                    ha="center",
                                    va="center",
                                    fontsize=7,
                                )
                    axis.set_title(
                        "Process sensor median correlations "
                        "matched by file and class"
                    )
                    figure.tight_layout()

        output_file = output_dir / "sensor_correlation_heatmap.png"
        figure.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_file}")

    def visualize_sensor_measurements_by_state(self) -> None:
        """Compare per-instance sensor medians across operational states."""
        input_file = (
            self.eda_reports_dir / "sensor_measurements_by_state.csv"
        )
        if not input_file.exists():
            print(f"EDA results not found: {input_file}")
            return

        required_columns = {
            "file_name",
            "state_code",
            "state_name",
            "sensor_variable",
            "median",
        }
        data = (
            pd.DataFrame(columns=sorted(required_columns))
            if input_file.stat().st_size == 0
            else pd.read_csv(input_file)
        )
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            print(
                "Cannot create operational-state sensor plots; input is "
                "missing columns: " + ", ".join(sorted(missing_columns))
            )
            return

        output_dir = self.eda_reports_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        data["state_code"] = pd.to_numeric(data["state_code"], errors="coerce")
        data["median"] = pd.to_numeric(data["median"], errors="coerce")

        for sensor in self.VISUALIZED_SENSORS:
            sensor_data = data.loc[
                data["sensor_variable"] == sensor,
                ["file_name", "state_code", "state_name", "median"],
            ].dropna(
                subset=["file_name", "state_code", "state_name", "median"]
            )
            per_instance = (
                sensor_data.groupby(
                    ["file_name", "state_code", "state_name"],
                    as_index=False,
                    dropna=False,
                )["median"]
                .median()
            )
            states = (
                per_instance.groupby(
                    ["state_code", "state_name"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    instance_count=("file_name", "nunique"),
                    values=("median", list),
                )
                .sort_values("state_code")
            )
            figure, axis = plt.subplots(
                figsize=(max(10, len(states) * 1.1), 6)
            )

            if states.empty:
                axis.text(
                    0.5,
                    0.5,
                    f"No usable operational-state data for {sensor}.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            else:
                labels = [
                    f"{row.state_name}\n(n={row.instance_count})"
                    for row in states.itertuples()
                ]
                axis.boxplot(
                    states["values"].tolist(),
                    tick_labels=labels,
                    showfliers=False,
                )
                axis.set_xticklabels(labels, rotation=40, ha="right")

            axis.set_title(
                f"{sensor}: Per-instance median by operational state"
            )
            axis.set_xlabel("Operational state (contributing instances)")
            axis.set_ylabel(f"{sensor} per-instance median")
            figure.tight_layout()
            output_file = (
                output_dir
                / (
                    f"{sensor.lower().replace('-', '_')}_"
                    "operational_state_boxplot.png"
                )
            )
            figure.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved {output_file}")

    def visualize_event_phase_durations(self) -> None:
        """Compare recorded segment durations across event phases."""
        input_file = self.eda_reports_dir / "event_phase_patterns.csv"
        if not input_file.exists():
            print(f"EDA results not found: {input_file}")
            return

        required_columns = {
            "class_code",
            "condition_phase",
            "event_name",
            "duration_hours",
        }
        data = (
            pd.DataFrame(columns=sorted(required_columns))
            if input_file.stat().st_size == 0
            else pd.read_csv(input_file)
        )
        missing_columns = required_columns - set(data.columns)
        output_dir = self.eda_reports_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_height = max(8, len(data) * 0.015)
        figure, axis = plt.subplots(figsize=(15, figure_height))

        if missing_columns:
            axis.text(
                0.5,
                0.5,
                "Input is missing columns: "
                + ", ".join(sorted(missing_columns)),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        else:
            data["class_code"] = pd.to_numeric(
                data["class_code"],
                errors="coerce",
            )
            data["duration_hours"] = pd.to_numeric(
                data["duration_hours"],
                errors="coerce",
            )
            data = data.dropna(
                subset=[
                    "class_code",
                    "condition_phase",
                    "event_name",
                    "duration_hours",
                ]
            )
            data = data[
                np.isfinite(data["duration_hours"])
                & (data["duration_hours"] > 0)
            ]
            segments = (
                data.groupby(
                    ["class_code", "condition_phase", "event_name"],
                    as_index=False,
                    dropna=False,
                )
                .agg(segment_count=("duration_hours", "size"),
                     durations=("duration_hours", list))
            )
            segments["_phase_order"] = (
                segments["condition_phase"]
                .map(self.PHASE_ORDER)
                .fillna(len(self.PHASE_ORDER))
            )
            segments = segments.sort_values(
                ["_phase_order", "event_name", "class_code"]
            )

            if segments.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No valid event segment durations are available.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            else:
                labels = [
                    (
                        f"{row.condition_phase} | {row.event_name} "
                        f"(class {int(row.class_code)}, "
                        f"n={row.segment_count} segments)"
                    )
                    for row in segments.itertuples()
                ]
                boxplot = axis.boxplot(
                    segments["durations"].tolist(),
                    vert=False,
                    tick_labels=labels,
                    patch_artist=True,
                    showfliers=False,
                )
                for box, phase in zip(
                    boxplot["boxes"],
                    segments["condition_phase"],
                ):
                    box.set_facecolor(
                        self.PHASE_COLORS.get(phase, "#999999")
                    )
                    box.set_alpha(0.75)
                axis.set_xscale("log")
                axis.legend(
                    handles=[
                        Patch(
                            facecolor=self.PHASE_COLORS.get(
                                phase,
                                "#999999",
                            ),
                            label=phase,
                        )
                        for phase in segments[
                            "condition_phase"
                        ].drop_duplicates()
                    ],
                    title="Condition phase",
                    loc="best",
                )
                axis.set_yticklabels(labels, fontsize=8)
                axis.invert_yaxis()

        axis.set_title("Recorded segment durations by production condition")
        axis.set_xlabel("Segment duration (hours, logarithmic scale)")
        axis.set_ylabel(
            "Condition categories; n counts recorded segments, not events"
        )
        figure.tight_layout()
        output_file = output_dir / "event_phase_durations.png"
        figure.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_file}")

    def visualize_operational_state_distribution(self) -> None:
        """Compare per-instance recording-time shares by state."""
        input_file = self.eda_reports_dir / "operational_state_patterns.csv"
        if not input_file.exists():
            print(f"EDA results not found: {input_file}")
            return

        required_columns = {
            "file_name",
            "state_code",
            "state_name",
            "percentage",
        }
        data = (
            pd.DataFrame(columns=sorted(required_columns))
            if input_file.stat().st_size == 0
            else pd.read_csv(input_file)
        )
        missing_columns = required_columns - set(data.columns)
        output_dir = self.eda_reports_dir / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(13, 7))

        if missing_columns:
            axis.text(
                0.5,
                0.5,
                "Input is missing columns: "
                + ", ".join(sorted(missing_columns)),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        else:
            data["state_code"] = pd.to_numeric(data["state_code"], errors="coerce")
            data["percentage"] = pd.to_numeric(
                data["percentage"],
                errors="coerce",
            )
            data = data.dropna(
                subset=["file_name", "state_code", "state_name", "percentage"]
            )
            per_instance = (
                data.groupby(
                    ["file_name", "state_code", "state_name"],
                    as_index=False,
                    dropna=False,
                )["percentage"]
                .mean()
            )
            states = (
                per_instance.groupby(
                    ["state_code", "state_name"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    instance_count=("file_name", "nunique"),
                    percentages=("percentage", list),
                )
                .sort_values("state_code")
            )

            if states.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No operational-state percentages are available.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            else:
                labels = [
                    f"{row.state_name}\n(n={row.instance_count})"
                    for row in states.itertuples()
                ]
                axis.boxplot(
                    states["percentages"].tolist(),
                    tick_labels=labels,
                    showfliers=False,
                )
                axis.set_xticklabels(labels, rotation=35, ha="right")

        axis.set_title(
            "Per-instance share of recorded time by operational state"
        )
        axis.set_xlabel("Operational state (contributing instances)")
        axis.set_ylabel("Share of labeled recording time per instance (%)")
        figure.tight_layout()
        output_file = output_dir / "operational_state_distribution.png"
        figure.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_file}")

    @staticmethod
    def get_real_files() -> list[Path]:
        """Return only real production instances."""
        return [
            file_path
            for file_path in get_parquet_files()
            if file_path.name.startswith("WELL-")
        ]

    @staticmethod
    def get_sensor_columns(df: pd.DataFrame) -> list:
        """Return sensor columns, excluding labels."""
        return [
            column
            for column in df.columns
            if column not in {"class", "state"}
        ]

    @classmethod
    def describe_class(cls, class_code: int) -> tuple[str, str]:
        """Map class codes to production conditions."""
        if class_code == 0:
            return "Normal Operation", "Normal Operation"

        if 1 <= class_code <= 9:
            return "Steady-state event", cls.EVENT_NAMES[class_code]

        if 101 <= class_code <= 109:
            event_code = class_code - 100
            return "Transient", cls.EVENT_NAMES[event_code]

        return "Unknown", "Unknown"

    def analyse_sensor_conditions(self) -> None:
        """Compare sensor measurements across production conditions."""
        self.eda_reports_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for file_number, file_path in enumerate(self.get_real_files(), start=1):
            df = load_parquet_file(file_path)
            labelled_data = df.dropna(subset=["class"])

            for class_code, group in labelled_data.groupby("class"):
                class_code = int(class_code)
                phase, event_name = self.describe_class(class_code)
                sensor_columns = self.get_sensor_columns(df)

                for sensor in sensor_columns:
                    values = group[sensor].dropna()

                    if values.empty:
                        continue

                    records.append({
                        "file_name": file_path.name,
                        "class_code": class_code,
                        "condition_phase": phase,
                        "event_name": event_name,
                        "sensor_variable": sensor,
                        "observation_count": len(values),
                        "median": values.median(),
                        "q25": values.quantile(0.25),
                        "q75": values.quantile(0.75),
                        "mean": values.mean(),
                        "std": values.std(),
                    })

            if file_number % 100 == 0:
                print(f"Processed {file_number} real instances")

        output = pd.DataFrame(
            records,
            columns=[
                "file_name",
                "class_code",
                "condition_phase",
                "event_name",
                "sensor_variable",
                "observation_count",
                "median",
                "q25",
                "q75",
                "mean",
                "std",
            ],
        )
        output.to_csv(
            self.eda_reports_dir / "sensor_measurements_by_condition.csv",
            index=False,
        )
        print("Sensor condition analysis completed.")

    def summarise_sensor_conditions_across_instances(self) -> None:
        """Summarise per-instance sensor medians by condition."""
        self.eda_reports_dir.mkdir(parents=True, exist_ok=True)
        input_path = (
            self.eda_reports_dir / "sensor_measurements_by_condition.csv"
        )
        input_columns = [
            "file_name",
            "class_code",
            "condition_phase",
            "event_name",
            "sensor_variable",
            "observation_count",
            "median",
        ]
        output_columns = [
            "class_code",
            "condition_phase",
            "event_name",
            "sensor_variable",
            "instance_count",
            "total_observations",
            "median_of_instance_medians",
            "mean_of_instance_medians",
            "q25_of_instance_medians",
            "q75_of_instance_medians",
        ]

        if input_path.stat().st_size == 0:
            per_instance = pd.DataFrame(columns=input_columns)
        else:
            per_instance = pd.read_csv(input_path)

        group_columns = [
            "class_code",
            "condition_phase",
            "event_name",
            "sensor_variable",
        ]

        def summarize(category_data: pd.DataFrame) -> pd.DataFrame:
            if category_data.empty:
                return pd.DataFrame(columns=output_columns)

            return (
                category_data.groupby(group_columns, dropna=False)
                .agg(
                    instance_count=("file_name", "nunique"),
                    total_observations=("observation_count", "sum"),
                    median_of_instance_medians=("median", "median"),
                    mean_of_instance_medians=("median", "mean"),
                    q25_of_instance_medians=(
                        "median",
                        lambda values: values.quantile(0.25),
                    ),
                    q75_of_instance_medians=(
                        "median",
                        lambda values: values.quantile(0.75),
                    ),
                )
                .reset_index()
                .reindex(columns=output_columns)
            )

        if per_instance.empty and not set(input_columns).issubset(
            per_instance.columns
        ):
            per_instance = pd.DataFrame(columns=input_columns)
        else:
            per_instance = per_instance[input_columns]

        equipment_mask = per_instance["sensor_variable"].str.startswith(
            "ESTADO-",
            na=False,
        )
        process_summary = summarize(per_instance[~equipment_mask])
        equipment_summary = summarize(per_instance[equipment_mask])

        process_summary.to_csv(
            self.eda_reports_dir / "process_sensor_condition_summary.csv",
            index=False,
        )
        equipment_summary.to_csv(
            self.eda_reports_dir / "equipment_state_condition_summary.csv",
            index=False,
        )
        print("Sensor condition summaries across instances completed.")

    def analyse_sensor_operational_states(self) -> None:
        """Examine sensor measurements across operational states."""
        self.eda_reports_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for file_path in self.get_real_files():
            df = load_parquet_file(file_path)

            # Exclude observations with unknown operational states.
            labelled_data = df.dropna(subset=["state"])
            sensor_columns = self.get_sensor_columns(df)

            for state_code, group in labelled_data.groupby("state"):
                state_code = int(state_code)

                for sensor in sensor_columns:
                    values = group[sensor].dropna()

                    if values.empty:
                        continue

                    records.append({
                        "file_name": file_path.name,
                        "state_code": state_code,
                        "state_name": self.STATE_NAMES[state_code],
                        "sensor_variable": sensor,
                        "observation_count": len(values),
                        "median": values.median(),
                        "q25": values.quantile(0.25),
                        "q75": values.quantile(0.75),
                    })

        output = pd.DataFrame(records)
        output.to_csv(
            self.eda_reports_dir / "sensor_measurements_by_state.csv",
            index=False,
        )
        print("Sensor operational-state analysis completed.")

    def analyse_event_patterns(self) -> None:
        """Examine the duration and occurrence of production conditions."""
        self.eda_reports_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for file_path in self.get_real_files():
            df = load_parquet_file(file_path)
            class_series = df["class"]
            previous_class = class_series.shift()
            same_as_previous = (
                class_series.eq(previous_class).fillna(False)
                | (class_series.isna() & previous_class.isna())
            )
            segment_ids = (~same_as_previous).cumsum()

            for segment_number, (_, segment_df) in enumerate(
                df.groupby(segment_ids), start=1
            ):
                class_value = segment_df["class"].iloc[0]
                if pd.isna(class_value):
                    continue

                class_code = int(class_value)
                phase, event_name = self.describe_class(class_code)

                records.append({
                    "file_name": file_path.name,
                    "event_folder": file_path.parent.name,
                    "segment_number": segment_number,
                    "class_code": class_code,
                    "condition_phase": phase,
                    "event_name": event_name,
                    "start_timestamp": segment_df.index.min(),
                    "end_timestamp": segment_df.index.max(),
                    "observations": len(segment_df),
                    "duration_hours": len(segment_df) / 3600,
                })

        output = pd.DataFrame(records)
        output.to_csv(
            self.eda_reports_dir / "event_phase_patterns.csv",
            index=False,
        )
        print("Event pattern analysis completed.")

    def analyse_operational_patterns(self) -> None:
        """Examine operational-state distribution and durations."""
        self.eda_reports_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for file_path in self.get_real_files():
            df = load_parquet_file(file_path)
            labelled = df.dropna(subset=["state"])

            if labelled.empty:
                continue

            total_observations = len(labelled)
            state_counts = labelled["state"].value_counts()

            for state_code, observation_count in state_counts.items():
                state_code = int(state_code)
                records.append({
                    "file_name": file_path.name,
                    "event_folder": file_path.parent.name,
                    "state_code": state_code,
                    "state_name": self.STATE_NAMES[state_code],
                    "observation_count": observation_count,
                    "percentage": (
                        observation_count / total_observations
                    ) * 100,
                    "total_duration_hours": observation_count / 3600,
                })

        output = pd.DataFrame(records)
        output.to_csv(
            self.eda_reports_dir / "operational_state_patterns.csv",
            index=False,
        )
        print("Operational pattern analysis completed.")


    def analyse_sensor_value_quality(self) -> None:
        """Investigate unusual values in selected process sensors."""

        self.eda_reports_dir.mkdir(parents=True, exist_ok=True)

        sensors_to_check = [
            "P-PDG",
            "P-JUS-CKP",
            "T-PDG",
        ]

        records = []

        for file_number, file_path in enumerate(self.get_real_files(), start=1):
            df = load_parquet_file(file_path)

            for sensor in sensors_to_check:
                if sensor not in df.columns:
                    continue

                raw_values = df[sensor]
                numeric_values = pd.to_numeric(raw_values, errors="coerce")

                missing_count = int(raw_values.isna().sum())

                non_numeric_count = int(
                    (raw_values.notna() & numeric_values.isna()).sum()
                )

                non_finite = numeric_values.isin([
                    float("inf"),
                    float("-inf"),
                ])

                non_finite_count = int(non_finite.sum())

                finite_values = numeric_values[
                    numeric_values.notna() & ~non_finite
                ]

                zero_count = int((finite_values == 0).sum())

                minimum = None
                maximum = None
                median = None
                q25 = None
                q75 = None
                outlier_count = None

                if not finite_values.empty:
                    minimum = finite_values.min()
                    maximum = finite_values.max()
                    median = finite_values.median()
                    q25 = finite_values.quantile(0.25)
                    q75 = finite_values.quantile(0.75)

                    iqr = q75 - q25

                    if iqr > 0:
                        lower_bound = q25 - (1.5 * iqr)
                        upper_bound = q75 + (1.5 * iqr)

                        outlier_count = int(
                            (
                                (finite_values < lower_bound)
                                | (finite_values > upper_bound)
                            ).sum()
                        )

                records.append({
                    "file_name": file_path.name,
                    "event_folder": file_path.parent.name,
                    "sensor_variable": sensor,
                    "observation_count": len(raw_values),
                    "valid_observation_count": len(finite_values),
                    "missing_count": missing_count,
                    "non_numeric_count": non_numeric_count,
                    "non_finite_count": non_finite_count,
                    "zero_count": zero_count,
                    "minimum": minimum,
                    "maximum": maximum,
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "potential_outlier_count": outlier_count,
                })

            if file_number % 100 == 0:
                print(f"Processed {file_number} real instances")

        output = pd.DataFrame(records)

        output.to_csv(
            self.eda_reports_dir / "sensor_value_quality_check.csv",
            index=False,
        )

        summary = output.groupby("sensor_variable").agg(
            instance_count=("file_name", "nunique"),
            total_observations=("observation_count", "sum"),
            valid_observations=("valid_observation_count", "sum"),
            missing_count=("missing_count", "sum"),
            non_numeric_count=("non_numeric_count", "sum"),
            non_finite_count=("non_finite_count", "sum"),
            zero_count=("zero_count", "sum"),
            potential_outlier_count=("potential_outlier_count", "sum"),
            minimum=("minimum", "min"),
            maximum=("maximum", "max"),
        )

        print("\nSENSOR VALUE QUALITY SUMMARY")
        print("-" * 80)
        print(summary.to_string())

        print("\nDetailed report saved to:")
        print(self.eda_reports_dir / "sensor_value_quality_check.csv")