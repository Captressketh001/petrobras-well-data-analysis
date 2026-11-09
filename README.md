# Petrobras Well Data Analysis

This project prepares, cleans, and analyzes the Petrobras 3W Dataset v2.0.0,
with a focus on real well instances. It includes data-quality profiling,
exploratory data analysis (EDA), visualizations, and reports for sensor
missingness and label distributions.

## Requirements

- Python 3.14
- Pipenv
- The Petrobras 3W Dataset v2.0.0 in Parquet format

Install the project dependencies from the repository root:

```bash
pipenv install
```

Place the dataset under `data/raw/dataset/`, retaining its directory structure
and Parquet files. The preparation step selects real instance files whose
names begin with `WELL-`.

The raw dataset is licensed under [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). Review the
dataset's included `data/raw/dataset/README.md` and `LICENSE-CC-BY` for
attribution and license details before sharing dataset files or derived
material.

## Data workflow

Run commands from the repository root using `pipenv run`:

```bash
# See all available analyses
pipenv run python scripts/run_analysis.py --list

# Prepare complete and labelled datasets from raw real instances
pipenv run python scripts/run_analysis.py --group prepare

# Clean the complete dataset and generate cleaning investigations
pipenv run python scripts/run_analysis.py --group cleaning

# Run data-quality profiles and EDA
pipenv run python scripts/run_analysis.py --group data-quality
pipenv run python scripts/run_analysis.py --group eda

# Run EDA visualizations
pipenv run python scripts/run_analysis.py --group visualize

# Run data-quality analyses and EDA analyses
pipenv run python scripts/run_analysis.py --group both
```

Preparation should be run before cleaning. The cleaning group reads
`data/processed/real_instances_complete.parquet` and writes
`data/processed/real_instances_cleaned.parquet`. It removes only sensor columns
with no valid observations, preserves rows and partially missing values, and
does not impute or replace measurements. Suspicious extreme values are
documented for review rather than automatically changed.

To run one analysis by name:

```bash
pipenv run python scripts/run_analysis.py --analysis clean_dataset
```

Use `--list` to see valid names for the current version of the project.

## Data and reports

| Path | Description |
|---|---|
| `data/raw/dataset/` | Source Petrobras 3W Parquet instances |
| `data/processed/real_instances_complete.parquet` | Prepared real-instance observations, including unknown labels |
| `data/processed/real_instances_labelled.parquet` | Prepared view containing observations with known `class` |
| `data/processed/real_instances_cleaned.parquet` | Complete dataset without completely empty sensor columns |
| [`reports/`](reports/) | Preparation, data-quality, and EDA outputs |
| [Cleaning report](reports/cleaning/cleaning_report.md) | Cleaning operations, row counts, missingness, duplicate/type checks, and unresolved concerns |
| [P-PDG extreme-value observations](reports/cleaning/p_pdg_extreme_values.md) | P-PDG distribution and extreme-value summary |
| [P-JUS-CKP and T-PDG observations](reports/cleaning/other_sensor_extreme_values.md) | Negative-value findings for both sensors |
| [Data preparation report](reports/data_preparation_report.md) | Prepared dataset coverage, labels, and feature availability |
| [Data quality notes](reports/data_quality_notes.md) | Additional data-quality observations |
| `reports/cleaning/p_pdg_extreme_values.csv` | P-PDG extreme-value counts by instance file |
| [`reports/eda/`](reports/eda/) | EDA tables and visualizations |

### EDA visualizations

- [Event phase durations](reports/eda/figures/event_phase_durations.png)
- [Operational state distribution](reports/eda/figures/operational_state_distribution.png)
- [Sensor correlation heatmap](reports/eda/figures/sensor_correlation_heatmap.png)
- [P-ANULAR measurements by condition](reports/eda/figures/p_anular_condition_boxplot.png)
- [P-ANULAR measurements by operational state](reports/eda/figures/p_anular_operational_state_boxplot.png)
- [P-TPT measurements by condition](reports/eda/figures/p_tpt_condition_boxplot.png)
- [P-TPT measurements by operational state](reports/eda/figures/p_tpt_operational_state_boxplot.png)
- [T-TPT measurements by condition](reports/eda/figures/t_tpt_condition_boxplot.png)
- [T-TPT measurements by operational state](reports/eda/figures/t_tpt_operational_state_boxplot.png)

Processed datasets are generated locally and are excluded from version control.
Run the relevant preparation or analysis command to regenerate outputs.

## Project layout

```text
data/
  raw/                 Source dataset (local)
  processed/           Generated Parquet datasets (local)
notebooks/             Dataset exploration and quality-assessment notebooks
reports/               Generated reports
scripts/run_analysis.py
src/                   Data preparation, cleaning, analysis, and loading code
```
