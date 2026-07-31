# Data dictionary

## Clinical inputs

| MIMIC-III table | Fields used | Purpose |
|---|---|---|
| `LABEVENTS` | `ROW_ID`, `SUBJECT_ID`, `HADM_ID`, `ITEMID`, `CHARTTIME`, `VALUENUM`, `VALUEUOM`, `FLAG` | Potassium value and time |
| `D_LABITEMS` | `ITEMID`, `LABEL`, `FLUID`, `CATEGORY` | Verify potassium item meaning |

## Waveform index

| Field | Meaning |
|---|---|
| `subject_id` | MIMIC-III patient identifier |
| `waveform_record` | WFDB matched master-record path |
| `record_start_time`, `record_end_time` | Surrogate-dated coverage interval |
| `sampling_rate`, `signal_length` | Sample timing |
| `lead_names` | Header-level signal names; final availability is rechecked |

## Materialized cohort

| Field | Meaning |
|---|---|
| `study_id` | Unique lab-event ID used as sample ID |
| `labevent_id` | MIMIC-III `LABEVENTS.ROW_ID` |
| `potassium_time`, `potassium` | Reference test |
| `ecg_anchor_time` | Chosen ECG instant |
| `abs_delta_minutes` | Distance from lab to available record coverage |
| `sample_start`, `sample_end` | WFDB sample interval |
| `waveform_cache_path` | Selectively cached NPZ |
| `label_id`, `label` | 0/HypoK, 1/NK, 2/HyperK |

All files containing these identifiers are restricted local artifacts.
