
# Data Quality Assessment — Petrobras 3W Dataset

## 1. Dataset Overview

**Dataset:** Petrobras 3W Dataset  
**Version:** 2.0.0  
**Total Parquet files (instances):** 2,228  
**Total observations:** 76,587,318  
**Variables:** 27 sensor variables, `class`, and `state`  
**Sampling frequency:** 1 Hz (one observation per second)

The dataset contains multivariate time series from three sources:

- **Real:** Measurements obtained from Petrobras' Plant Information Management Systems (PIMS).
- **Simulated:** Time series generated using the OLGA multiphase-flow simulator.
- **Hand-drawn:** Time series created by Petrobras experts using a specialised digital tool.

Each Parquet file represents an instance, not necessarily one unique real-world event occurrence.

### Instance distribution by source

| Source | Instances |
|---|---:|
| Real | 1,119 |
| Simulated | 1,089 |
| Hand-drawn | 20 |
| **Total** | **2,228** |

Source types were identified using the official filename prefixes:

- `WELL-`: Real
- `SIMULATED_`: Simulated
- `DRAWN_`: Hand-drawn

All 2,228 files were successfully classified. Their counts match the official dataset documentation.

## 2. Data Quality Findings

### 2.1 File and timestamp integrity

Profiling of all 2,228 files established:

- No duplicate timestamps within individual files.
- No missing one-second intervals within individual files.
- Timestamps are chronologically ordered.
- All files follow the expected one-second sampling frequency.

Inter-file timestamp continuity for instances belonging to the same well has not yet been assessed.

### 2.2 Missing labels

The `class` and `state` columns have identical missingness patterns.

| Metric | Result |
|---|---:|
| Total missing observations | 4,028,400 |
| Percentage of dataset | 5.26% |
| Files affected | 1,119 |
| Missing intervals per affected file | 1 |
| Duration of each interval | 3,600 seconds (1 hour) |
| Position of interval | Beginning of recording |

Further investigation established:

- All 1,119 real instances have exactly one initial hour of missing `class` and `state` labels.
- None of the 1,089 simulated instances has missing labels.
- None of the 20 hand-drawn instances has missing labels.
- No affected file has missing-label intervals in the middle or at the end of its recording.

The 1,119 affected files correspond exactly to the 1,119 real instances.

**Interpretation:** Missing label values are systematically associated with the real data source. Differences in missing-label percentages across event folders are explained by their different proportions of real and synthetic instances, rather than evidence of event-specific annotation failure.

The dataset documentation defines:
- `class = NaN`: Normality/event condition unknown.
- `state = NaN`: Operational condition unknown.

The reason for the initial unlabelled hour has not been established. It should not automatically be interpreted as normal operation.

**Documentation discrepancy:** The v2.0.0 dataset README states that labelling gaps in real instances were eliminated. However, the 2026 research article reports 4,028,400 unlabelled observations (5.26%), matching the results obtained from this analysis. This discrepancy remains unresolved.

### 2.3 Sensor missingness and availability

Sensor availability was initially assessed across the complete dataset and by event folder. Further profiling separated the three data sources.

#### Real instances

| Sensor | Availability |
|---|---:|
| T-TPT | 86.44% |
| P-TPT | 83.63% |
| P-MON-CKP | 78.46% |
| P-ANULAR | 77.45% |
| P-JUS-CKGL | 76.21% |
| P-PDG | 75.92% |
| ESTADO-SDV-P | 73.97% |
| T-PDG | 63.31% |
| ESTADO-W1 | 57.61% |
| ESTADO-XO | 56.40% |

Four sensors have zero availability across all real instances:

- `QBS`
- `PT-P`
- `P-MON-SDV-P`
- `P-JUS-BS`

`P-MON-CKGL` has only 0.94% availability and is entirely missing in 1,117 of the 1,119 real instances.

#### Simulated instances

Nine sensors have some available observations.

Four sensors have 100% availability:

- `P-TPT`
- `P-MON-CKP`
- `T-JUS-CKP`
- `P-PDG`

`T-TPT` has 85.75% availability.

Eighteen sensors are entirely missing across all simulated instances.

#### Hand-drawn instances

Five sensors have available observations:

- `P-MON-CKP`
- `P-PDG`
- `P-TPT`
- `T-JUS-CKP`
- `T-TPT`

All five have 100% availability.

The remaining 22 sensors are entirely missing across all 20 hand-drawn instances.

**Interpretation:** Sensor availability differs substantially across data sources. Some missingness is structural and reflects the measurement coverage of each source. A sensor's low overall availability does not necessarily mean it is unsuitable for every analytical use case.

A single global missingness threshold should therefore not be applied to all sensor variables.

### 2.4 Class label distribution

The `class` variable identifies normal operation, transient conditions and undesirable production events.

Important distinctions:

- Classes 0–9 represent normal operation and nine undesirable event types.
- Classes 101–109 represent transient phases of events 1–9.
- Class values are categorical codes, not continuous numerical measurements.
- Not every event has a corresponding transient class.

Missing class observations account for 5.26% of the dataset.

Class distributions based on observation counts represent recorded time, not independent event occurrences. Longer recordings contribute more observations and can therefore dominate the distribution.

### 2.5 Operational state distribution

The `state` variable describes the well's operational status, independently of the event class.

| Code | Operational state |
|---:|---|
| 0 | Open |
| 1 | Shut-In |
| 2 | Flushing Diesel |
| 3 | Flushing Gas |
| 4 | Bullheading |
| 5 | Closed With Diesel |
| 6 | Closed With Gas |
| 7 | Restart |
| 8 | Depressurization |

Overall, state 0 (Open) accounts for approximately 90.96% of observations.

The state variable must not be confused with the class variable. For example, class 0 means Normal Operation, while state 0 means Open.

### 2.6 Recording durations

Recording durations vary substantially across event folders.

Notable observations:

- Folder 7 has the longest median recording duration at approximately 49.52 hours.
- Folder 8 has a mean duration of approximately 20.46 hours but a median of 7.50 hours, indicating that some very long recordings influence its mean.
- Folder 4 has the shortest typical recording duration, with a median of approximately 2.99 hours.

Observation counts should not be treated as counts of independent production events.

## 3. Data Quality Assessment and Handling Principles

| Finding | Assessment | Proposed handling |
|---|---|---|
| Initial missing labels in real instances | Systematic, source-associated | Preserve as unknown; do not impute automatically |
| Sensor missingness in simulated data | Largely structural | Assess against source-specific sensor coverage |
| Sensor missingness in hand-drawn data | Largely structural | Use only available variables for relevant analyses |
| Sensors with zero real-data availability | Unavailable in this dataset | Exclude from real-data analyses |
| Sensors with low real-data availability | Limited coverage | Evaluate usefulness by event and analytical objective |
| Unequal recording durations | Different observation contributions | Use appropriate aggregation and avoid interpreting row counts as event counts |
| Class and state variables | Categorical labels | Preserve categorical meaning; do not treat codes as continuous measurements |
| Timestamp integrity | Consistent within files | Retain timestamps; investigate inter-file continuity separately if needed |

No final cleaning or imputation decisions have been made yet. These will be based on the requirements of the subsequent analysis.

## 4. Existing Analysis Outputs

The following reports have been generated:

### File and recording profiles
- `file_quality_summary.csv`
- `recording_summary_by_file.csv`
- `recording_summary_by_folder.csv`
- `recording_summary_by_file_with_source.csv`

### Sensor analysis
- `sensor_missingness_by_folder.csv`
- `sensor_coverage_matrix.csv`
- `overall_sensor_availability.csv`
- `sensor_availability_by_source.csv`
- `sensor_availability_by_folder_and_source.csv`

### Label analysis
- `label_distribution_by_folder.csv`
- `label_missingness_alignment_by_file.csv`
- `label_missingness_alignment_by_folder.csv`
- `label_distribution_percentages.csv`
- `overall_label_distributions.csv`
- `missing_label_intervals.csv`
- `missing_label_summary_by_file.csv`
- `missing_label_position_patterns.csv`
- `missing_label_source_verification.csv`
- `source_counts_by_event_folder.csv`

## 5. Next Steps

1. Perform exploratory data analysis of production measurements across event categories.
2. Identify useful sensor variables for the intended analytical questions.
3. Investigate distributions, relationships, trends and potential outliers.
4. Make evidence-based data preparation and cleaning decisions.
5. Generate analytical datasets and visualisations for Power BI.
6. Document insights, limitations and recommendations.

## 6. Data Handling Principles

- Preserve the original raw dataset.
- Do not impute unknown class or state labels without a justified analytical reason.
- Distinguish structural missingness from potentially accidental missingness.
- Consider source-specific sensor availability before selecting variables.
- Avoid global missingness thresholds that ignore source and event differences.
- Treat class and state as categorical variables.
- Distinguish observation counts from independent event counts.
- Document every transformation and cleaning decision.
- Ensure analytical conclusions reflect the limitations of the dataset.
