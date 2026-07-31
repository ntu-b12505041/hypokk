# Research protocol

## Primary question

Can a ten-second Lead II ICU bedside ECG window classify concurrent serum
potassium status as HypoK, NK, or HyperK?

## Data

- MIMIC-III Clinical Database v1.4: `LABEVENTS`, `D_LABITEMS`.
- MIMIC-III Waveform Database Matched Subset v1.0.
- These datasets share MIMIC-III subject identifiers and surrogate dates.
- MIMIC-IV-ECG is not linked because its identifiers and date shifts are from a
  different release.

## Unit of analysis

One eligible potassium laboratory event paired with at most one fixed-length
ECG window. When the lab time lies inside a waveform record, the window is
centered on the lab time. Otherwise, the nearest record boundary is allowed
only within the prespecified ±60-minute window.

Near-duplicate windows from the same record are removed using a prespecified
minimum separation. Windows with missing requested leads, gaps, NaN, or
unreadable WFDB data are excluded before splitting.

## Labels

- HypoK: K⁺ < 3.5 mmol/L.
- NK: 3.5 ≤ K⁺ < 5.5 mmol/L.
- HyperK: K⁺ ≥ 5.5 mmol/L.
- Primary analysis: serum/plasma chemistry potassium, item ID verified from
  `D_LABITEMS`.
- Whole-blood potassium is sensitivity analysis only.

## Split

Patients, not windows, are randomly divided 70/15/15. All windows belonging to
one `subject_id` stay in one split. The test set remains locked during
preprocessing, model selection, calibration, and threshold selection.

## Model and validation

The primary model is a one-lead SE-ResNet1D with classification, ordinal, and
continuous potassium heads. Model selection uses validation macro one-vs-rest
AUROC. Temperature and ordered decision thresholds are fitted on validation
only. Final confidence intervals use patient-cluster bootstrap.

The acceptance rule is strict: every class must have recall >0.85 and
specificity >0.85 on the locked test set.

## Prespecified sensitivity analyses

- matching windows: ±30, ±60, and ±120 minutes;
- only lab times directly covered by waveform records;
- serum only versus serum plus whole blood;
- different minimum separations between windows;
- maximum contribution cap per training patient;
- alternate bedside ECG lead availability.

These decisions must not be chosen using final test performance.
