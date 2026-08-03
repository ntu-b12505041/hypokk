# Regularized Rotating-Subsampling Experiment

This is the third prespecified experiment. It responds to the second run's
training/validation divergence while keeping its patient-capped rotating NK
sampling strategy unchanged.

## Why this experiment exists

The second run reached its best validation macro AUROC at epoch 8, while
training AUROC continued to improve and validation AUROC later declined. This
pattern is consistent with overfitting. The third run changes only optimization
and regularization settings so that its result remains interpretable as an
ablation.

| Setting | Second run | Third run |
|---|---:|---:|
| Dropout | 0.20 | 0.30 |
| Learning rate | 0.001 | 0.0003 |
| Weight decay | 0.0001 | 0.0005 |
| Maximum epochs | 60 | 40 |
| Warm-up epochs | 3 | 2 |
| Early-stopping patience | 10 | 8 |

All data definitions, patient-disjoint splits, preprocessing, loss heads,
validation calibration, and rotating sampling settings remain unchanged.
Validation and test are never subsampled.

## Run

The configuration writes to a new output directory and therefore does not
overwrite either earlier experiment:

```text
outputs/mimic3_lead2_se_resnet_subsampled_regularized
```

Update only the protected-data roots in
`configs/mimic3_subsampled_regularized.yaml`, then run:

```bash
hypok-mimic3 validate-config \
  --config configs/mimic3_subsampled_regularized.yaml

mkdir -p run_logs
hypok-mimic3 train \
  --config configs/mimic3_subsampled_regularized.yaml \
  2>&1 | tee "run_logs/07_train_regularized_$(date +%Y%m%d_%H%M%S).log"
```

Do not evaluate the locked test split yet. First compare validation macro
AUROC, per-class AUROC/AUPRC, confusion matrix, sensitivity, specificity, and
the train/validation learning curves with the first two runs.
