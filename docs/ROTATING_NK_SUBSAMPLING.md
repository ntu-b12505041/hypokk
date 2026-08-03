# Rotating NK Subsampling Experiment

This experiment is an ablation after the original full-cohort Lead II baseline.
It does not change the patient-level split and does not subsample validation or
test data.

## Prespecified training-only sampling

For each epoch, the sampler:

1. applies a maximum of 10 windows per patient per class;
2. retains all eligible HypoK and HyperK windows after that cap;
3. selects at most 8,000 NK windows without replacement;
4. changes the selected NK windows deterministically with `seed + epoch`;
5. shuffles all selected training indices.

The loss uses square-root inverse-frequency class weights calculated from the
first epoch's sampled class counts. This is intentionally softer than combining
majority subsampling with the original effective-number weights.

## Run

Keep the completed baseline outputs. The new configuration writes to a separate
directory:

```text
outputs/mimic3_lead2_se_resnet_subsampled
```

Update only the local data roots in `configs/mimic3_subsampled.yaml` if the
server stores the protected data elsewhere, then validate and train:

```bash
hypok-mimic3 validate-config --config configs/mimic3_subsampled.yaml

mkdir -p run_logs
hypok-mimic3 train \
  --config configs/mimic3_subsampled.yaml \
  2>&1 | tee "run_logs/06_train_subsampled_$(date +%Y%m%d_%H%M%S).log"
```

Do not run `evaluate` until the validation results have been reviewed and the
experiment has been selected without looking at the locked test split.

## Additional outputs

The training run writes:

```text
logs/sampling_audit.csv
logs/sampling_manifests/epoch_001.csv
logs/sampling_manifests/epoch_002.csv
...
```

The aggregate audit records, for every epoch and class:

- available and sampled ECG windows;
- independent patients;
- maximum and median windows contributed by one patient;
- top-one and top-five patient contribution shares.

The epoch manifests contain record-level identifiers and are protected derived
data. Keep them local and never commit or publish them.

## Comparison with the baseline

Compare validation macro AUROC, per-class AUROC/AUPRC, sensitivity,
specificity, confusion matrix, and calibration results against the original
baseline. A lower NK prediction rate alone is not evidence of improvement;
threshold-independent validation discrimination must improve.
