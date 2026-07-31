PYTHON ?= python
CONFIG ?= configs/mimic3.yaml

.PHONY: install test demo index cohort materialize split train evaluate

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) scripts/make_synthetic_demo.py --output-dir outputs/synthetic_demo

index:
	hypok-mimic3 index-waveforms --config $(CONFIG)

cohort:
	hypok-mimic3 build-cohort --config $(CONFIG)

materialize:
	hypok-mimic3 materialize-windows --config $(CONFIG)

split:
	hypok-mimic3 split --config $(CONFIG)

train:
	hypok-mimic3 train --config $(CONFIG)

evaluate:
	hypok-mimic3 evaluate --config $(CONFIG)
