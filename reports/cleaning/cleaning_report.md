# Data Cleaning Report

## Operations performed

- Removed sensor columns with no valid observations.
- Preserved all remaining values without imputation, coercion, or outlier replacement.
- Preserved row order, timestamps, labels, and instance metadata.

## Columns removed

- `P-JUS-BS`
- `P-MON-SDV-P`
- `PT-P`
- `QBS`

## Rows

- Rows before cleaning: 32,872,250
- Rows after cleaning: 32,872,250
- Rows removed: 0

## Missing values preserved

Null/NaN values in retained columns are preserved as missing values; no imputation or row filtering was performed.

| Column | Missing values |
|---|---:|
| `ABER-CKGL` | 24,380,737 |
| `ABER-CKP` | 20,545,792 |
| `P-ANULAR` | 7,411,735 |
| `P-JUS-CKGL` | 7,821,871 |
| `P-JUS-CKP` | 29,423,349 |
| `P-MON-CKGL` | 32,563,524 |
| `P-MON-CKP` | 7,079,851 |
| `P-PDG` | 7,916,216 |
| `P-TPT` | 5,381,783 |
| `QGL` | 15,152,145 |
| `T-JUS-CKP` | 16,032,900 |
| `T-MON-CKP` | 27,921,704 |
| `T-PDG` | 12,060,978 |
| `T-TPT` | 4,457,250 |
| `ESTADO-DHSV` | 19,006,853 |
| `ESTADO-M1` | 15,099,045 |
| `ESTADO-M2` | 15,183,968 |
| `ESTADO-PXO` | 14,696,047 |
| `ESTADO-SDV-GL` | 15,766,979 |
| `ESTADO-SDV-P` | 8,556,143 |
| `ESTADO-W1` | 13,934,214 |
| `ESTADO-W2` | 14,578,788 |
| `ESTADO-XO` | 14,333,520 |
| `state` | 4,028,400 |
| `class` | 4,028,400 |

## Duplicate-row check

- Exact duplicate rows found: 0.
- Repeated instance/timestamp key pairs: 0.
- Timestamps out of order within an instance: 0.
- Instance groups that reappear after another instance: 0.
Rows are grouped by instance and timestamps are chronological, so exact duplicate rows would be adjacent; none were found.
No duplicate rows were deleted.

## Data types

No invalid data types were found. Timestamps, metadata, labels, and sensor columns retain their source types; no conversions were performed.

## Suspicious values retained for further consideration

Previously investigated extreme negative readings remain unchanged. Their physical meaning has not been confirmed, so they are documented as unresolved rather than treated as invalid or replaced.

- `P-PDG` minimum `-1.180116e+42` occurred 305,679 time(s) in `WELL-00006_20180617190257.parquet` (57,424), `WELL-00006_20180618103000.parquet` (131,407), `WELL-00006_20180620152903.parquet` (116,848).
- `P-JUS-CKP` minimum `-9999000` occurred 161,538 time(s) in `WELL-00030_20141108012623.parquet` (161,538).
- `T-PDG` minimum `-1.7116129999999999e+38` occurred 305,679 time(s) in `WELL-00006_20180617190257.parquet` (57,424), `WELL-00006_20180618103000.parquet` (131,407), `WELL-00006_20180620152903.parquet` (116,848).

Related investigations: `reports/cleaning/p_pdg_extreme_values.md` and `reports/cleaning/other_sensor_extreme_values.md`.

## Remaining data-quality concerns

- Missing sensor measurements and unknown labels remain in the dataset by design.
- The suspicious extreme readings above require domain or source documentation before they can be classified.
- This step removes only completely empty sensor columns; it does not filter rows or perform further cleaning.

Cleaned dataset: `data/processed/real_instances_cleaned.parquet`
Source dataset was not modified: `data/processed/real_instances_complete.parquet`
