# GPU runbook

```bash
cd ~/hypok-mimic3
source .venv/bin/activate

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
hypok-mimic3 validate-config --config configs/mimic3.yaml
```

Run stages separately:

```bash
hypok-mimic3 index-waveforms --config configs/mimic3.yaml --workers 16
hypok-mimic3 build-cohort --config configs/mimic3.yaml
hypok-mimic3 materialize-windows --config configs/mimic3.yaml --workers 8
hypok-mimic3 split --config configs/mimic3.yaml
hypok-mimic3 train --config configs/mimic3.yaml
hypok-mimic3 evaluate --config configs/mimic3.yaml
```

Before training, inspect:

```bash
df -h /data
du -sh /data/mimic3_selective_windows
python -c "import pandas as pd; x=pd.read_csv('data/processed/mimic3_materialized_cohort_split.csv'); print(pd.crosstab(x.split,x.label)); print(x.groupby('split').subject_id.nunique())"
```

Do not start formal training if a split lacks a class, a rare class comes from
only a few patients, the GPU is unavailable, or required files are incomplete.
