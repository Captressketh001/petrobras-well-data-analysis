# Data Preparation Report

## Dataset summary

- Dataset: Petrobras 3W Dataset v2.0.0
- Selection: real instances with the documented `WELL-` prefix
- Real instances processed: 1,119
- Unique wells represented: 40
- Total observations read: 32,872,250
- Observations retained in the complete dataset: 32,872,250
- Labelled observations (`class` known): 28,843,850
- Observations with unknown `class`: 4,028,400
- Complete dataset: `data/processed/real_instances_complete.parquet`
- Labelled modelling view: `data/processed/real_instances_labelled.parquet`

The complete dataset preserves the original timestamp order, sensor values, `class` and `state` labels, including unknown labels. The modelling view contains only observations with known `class` and retains all metadata and measurements. No values were imputed or filtered by feature availability. Class codes are categorical labels, not ordered or continuous measurements.

## Class distribution

| Class code | Condition | Observations |
|---:|---|---:|
| 0 | Normal Operation | 14,570,142 |
| 1 | Abrupt Increase of BSW | 9,783 |
| 2 | Spurious Closure of DHSV | 21,274 |
| 3 | Severe Slugging | 569,152 |
| 4 | Flow Instability | 2,454,883 |
| 5 | Rapid Productivity Loss | 13,205 |
| 6 | Quick Restriction in PCK | 13,285 |
| 7 | Scaling in PCK | 30,339 |
| 8 | Hydrate in Production Line | 140,920 |
| 9 | Hydrate in Service Line | 17,752 |
| 101 | Transient: Abrupt Increase of BSW | 65,340 |
| 102 | Transient: Spurious Closure of DHSV | 89,091 |
| 105 | Transient: Rapid Productivity Loss | 320,672 |
| 106 | Transient: Quick Restriction in PCK | 6,569 |
| 107 | Transient: Scaling in PCK | 6,670,582 |
| 108 | Transient: Hydrate in Production Line | 3,655,418 |
| 109 | Transient: Hydrate in Service Line | 195,443 |
| Unknown (NaN) | Unknown condition | 4,028,400 |

Class counts include 28,843,850 labelled observations and 4,028,400 unknown labels.

## Feature availability

Availability is calculated over all retained real-instance observations. The two feature groups remain separate; no missingness threshold was applied.

| Feature group | Feature | Available | Missing | Availability | Missingness |
|---|---|---:|---:|---:|---:|
| Process sensor measurement | `ABER-CKGL` | 8,491,513 | 24,380,737 | 25.83% | 74.17% |
| Process sensor measurement | `ABER-CKP` | 12,326,458 | 20,545,792 | 37.50% | 62.50% |
| Process sensor measurement | `P-ANULAR` | 25,460,515 | 7,411,735 | 77.45% | 22.55% |
| Process sensor measurement | `P-JUS-BS` | 0 | 32,872,250 | 0.00% | 100.00% |
| Process sensor measurement | `P-JUS-CKGL` | 25,050,379 | 7,821,871 | 76.21% | 23.79% |
| Process sensor measurement | `P-JUS-CKP` | 3,448,901 | 29,423,349 | 10.49% | 89.51% |
| Process sensor measurement | `P-MON-CKGL` | 308,726 | 32,563,524 | 0.94% | 99.06% |
| Process sensor measurement | `P-MON-CKP` | 25,792,399 | 7,079,851 | 78.46% | 21.54% |
| Process sensor measurement | `P-MON-SDV-P` | 0 | 32,872,250 | 0.00% | 100.00% |
| Process sensor measurement | `P-PDG` | 24,956,034 | 7,916,216 | 75.92% | 24.08% |
| Process sensor measurement | `PT-P` | 0 | 32,872,250 | 0.00% | 100.00% |
| Process sensor measurement | `P-TPT` | 27,490,467 | 5,381,783 | 83.63% | 16.37% |
| Process sensor measurement | `QBS` | 0 | 32,872,250 | 0.00% | 100.00% |
| Process sensor measurement | `QGL` | 17,720,105 | 15,152,145 | 53.91% | 46.09% |
| Process sensor measurement | `T-JUS-CKP` | 16,839,350 | 16,032,900 | 51.23% | 48.77% |
| Process sensor measurement | `T-MON-CKP` | 4,950,546 | 27,921,704 | 15.06% | 84.94% |
| Process sensor measurement | `T-PDG` | 20,811,272 | 12,060,978 | 63.31% | 36.69% |
| Process sensor measurement | `T-TPT` | 28,415,000 | 4,457,250 | 86.44% | 13.56% |
| Equipment-state indicator | `ESTADO-DHSV` | 13,865,397 | 19,006,853 | 42.18% | 57.82% |
| Equipment-state indicator | `ESTADO-M1` | 17,773,205 | 15,099,045 | 54.07% | 45.93% |
| Equipment-state indicator | `ESTADO-M2` | 17,688,282 | 15,183,968 | 53.81% | 46.19% |
| Equipment-state indicator | `ESTADO-PXO` | 18,176,203 | 14,696,047 | 55.29% | 44.71% |
| Equipment-state indicator | `ESTADO-SDV-GL` | 17,105,271 | 15,766,979 | 52.04% | 47.96% |
| Equipment-state indicator | `ESTADO-SDV-P` | 24,316,107 | 8,556,143 | 73.97% | 26.03% |
| Equipment-state indicator | `ESTADO-W1` | 18,938,036 | 13,934,214 | 57.61% | 42.39% |
| Equipment-state indicator | `ESTADO-W2` | 18,293,462 | 14,578,788 | 55.65% | 44.35% |
| Equipment-state indicator | `ESTADO-XO` | 18,538,730 | 14,333,520 | 56.40% | 43.60% |

## Real instances by event folder

| Event folder | Instances |
|---:|---:|
| 0 | 594 |
| 1 | 4 |
| 2 | 22 |
| 3 | 32 |
| 4 | 343 |
| 5 | 11 |
| 6 | 6 |
| 7 | 36 |
| 8 | 14 |
| 9 | 57 |
