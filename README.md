# MIMIC-III Bedside ECG Dyskalemia

An independent, reproducible research pipeline for classifying:

- **HypoK**: serum K⁺ < 3.5 mmol/L
- **NK**: 3.5 ≤ serum K⁺ < 5.5 mmol/L
- **HyperK**: serum K⁺ ≥ 5.5 mmol/L

This repository is separate from `ntu-b12505041/hypok` and does not modify that
MIMIC-IV project.

## Feasible data combination

The valid pairing is:

| Role | Dataset | Version | Access |
|---|---|---:|---|
| Potassium labels | MIMIC-III Clinical Database | 1.4 | Credentialed |
| ECG signal | MIMIC-III Waveform Database Matched Subset | 1.0 | Open |

Official pages:

- [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)
- [MIMIC-III Waveform Database Matched Subset v1.0](https://physionet.org/content/mimic3wdb-matched/1.0/)
- [MIMIC-III Clinical Database Demo v1.4](https://physionet.org/content/mimiciii-demo/1.4/)

MIMIC-III Clinical alone contains no ECG waveform and cannot train an ECG model.
MIMIC-III cannot be joined to MIMIC-IV-ECG by `subject_id`: the two releases use
different patient identifier and date-shift mappings.

MIMIC-III waveforms are continuous ICU bedside-monitor signals, not ten-second
diagnostic 12-lead ECGs. The primary analysis therefore uses a ten-second
**Lead II** window nearest each eligible potassium test. A 12-lead ECGFounder
checkpoint is deliberately not forced onto this modality.

## Research status

No real performance is claimed by this code package. Formal training requires
MIMIC-III Clinical access. The open 100-patient Clinical Demo can exercise
schema and matching code but is not large enough to establish the requested
performance.

MIMIC-III Clinical is also protected data. Switching from MIMIC-IV Clinical
does not bypass PhysioNet credentialing and the applicable data-use agreement.

The prespecified final acceptance criterion remains:

> On the once-locked held-out patient-level test set, every class must have
> sensitivity/recall > 0.85 and specificity > 0.85.

The software reports whether this target is met; it never guarantees it.

## Pipeline

1. Index only the 22,317 matched WFDB master headers.
2. Query potassium from `LABEVENTS` and validate item IDs against `D_LABITEMS`.
3. Match each lab to the nearest ECG-covered instant within ±60 minutes.
4. Fetch only the selected ten-second Lead II windows and cache compressed NPZ.
5. Remove unusable/gap windows.
6. Split by `subject_id` into patient-disjoint 70/15/15 sets.
7. Audit patient and class concentration.
8. Train, calibrate on validation, and evaluate the locked test set once.

The full 2.4 TB waveform database is **not** downloaded.

## Installation

```bash
git clone https://github.com/ntu-b12505041/hypokk.git
cd hypokk
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
hypok-mimic3 validate-config --config configs/mimic3.yaml
```

## Minimal open-data preparation

Create metadata directories:

```bash
sudo mkdir -p /data/mimic3wdb-matched/1.0
sudo chown -R "$USER":"$USER" /data/mimic3wdb-matched
```

Download only the waveform record index:

```bash
wget -c \
  -P /data/mimic3wdb-matched/1.0 \
  https://physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms
```

Optional open Clinical Demo:

```bash
cd /data
wget -r -N -c -np \
  https://physionet.org/files/mimiciii-demo/1.4/
```

Adjust `configs/mimic3_demo.yaml` if wget creates an extra
`physionet.org/files/...` directory level.

## Formal Clinical data

After credentialing and signing the MIMIC-III v1.4 DUA, the minimum required
files are:

```text
LABEVENTS.csv.gz
D_LABITEMS.csv.gz
```

Optional patient and admission summaries may additionally use:

```text
PATIENTS.csv.gz
ADMISSIONS.csv.gz
ICUSTAYS.csv.gz
```

Never commit these files.

## Commands

### 1. Index headers without downloading signals

```bash
hypok-mimic3 index-waveforms \
  --config configs/mimic3.yaml \
  --workers 16
```

### 2. Build the ECG–potassium candidate manifest

```bash
hypok-mimic3 build-cohort --config configs/mimic3.yaml
```

### 3. Fetch only selected ten-second windows

```bash
hypok-mimic3 materialize-windows \
  --config configs/mimic3.yaml \
  --workers 8
```

### 4. Create patient-level splits and audits

```bash
hypok-mimic3 split --config configs/mimic3.yaml
```

### 5. Train and evaluate

```bash
nvidia-smi
hypok-mimic3 train --config configs/mimic3.yaml
hypok-mimic3 evaluate --config configs/mimic3.yaml
```

Do not use `run-all` until individual stages have been verified.

## Outputs

The formal run produces:

```text
outputs/mimic3_lead2_se_resnet/
├── checkpoints/
│   ├── best.pt
│   └── model_weights.h5
├── figures/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── per_class_metrics.png
│   └── roc_pr_curves.png
├── logs/
│   ├── training_history.csv
│   └── training_summary.json
├── metrics/
│   ├── calibration.json
│   ├── test_metrics.json
│   ├── test_confidence_intervals.json
│   └── test_predictions.csv
└── reports/
    └── validation_report.md
```

The split stage additionally produces restricted local audit files:

```text
data/processed/split_audit/
├── class_subject_ids.csv
├── patient_contributions.csv
└── split_patient_audit.json
```

They show, for every split and class:

- number of independent patients;
- exact `subject_id` list;
- ECG windows contributed by every patient;
- maximum and median contribution;
- top-one and top-five patient share;
- patient-concentration HHI;
- warning when a class has fewer than 20 patients or one patient contributes
  at least 25% of its samples.

## HDF5 model format

`model_weights.h5` is an HDF5 export of the PyTorch `state_dict`; it is not a
Keras model. It is loaded by this project:

```bash
hypok-mimic3 predict \
  --config configs/mimic3.yaml \
  --input /data/mimic3_selective_windows/example.npz
```

## Validation report contents

The report includes:

- dataset names and versions;
- ECG–laboratory matching and exclusions;
- patient-level train/validation/test split;
- preprocessing;
- architecture and hyperparameters;
- native and HDF5 checkpoint paths;
- training duration and environment;
- training/validation ACC and loss curves;
- overall accuracy, balanced accuracy, precision, recall, F1, AUROC and AUPRC;
- per-class sensitivity, specificity, precision, F1, AUROC and AUPRC;
- patient-cluster bootstrap 95% confidence intervals;
- confusion matrix and ROC/PR figures;
- patient concentration audit;
- explicit PASS/NOT MET result for the >0.85 targets.

## Synthetic reporting check

This checks only code and report generation:

```bash
python scripts/make_synthetic_demo.py \
  --output-dir outputs/synthetic_demo
```

Every generated figure and report is visibly marked synthetic.

## Data governance

MIMIC Clinical data, record-level manifests, subject IDs, predictions, and
trained checkpoints must not be pushed to a public repository or sent to an
online AI service. Only source code, configuration, documentation, and safely
aggregated results may be public.

This project is research software, not a medical device.
