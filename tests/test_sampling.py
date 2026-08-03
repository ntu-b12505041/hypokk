import numpy as np
import pandas as pd

from hypok_mimic3.sampling import RotatingTrainSubsampler, SamplingSettings


def _frame() -> pd.DataFrame:
    rows = []
    study_id = 0
    for label_id, patient_counts in {0: [3, 2], 1: [8, 8, 8], 2: [2, 2]}.items():
        for patient_offset, count in enumerate(patient_counts):
            subject_id = label_id * 100 + patient_offset
            for _ in range(count):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "label_id": label_id,
                    }
                )
                study_id += 1
    return pd.DataFrame(rows)


def _sampler() -> RotatingTrainSubsampler:
    return RotatingTrainSubsampler(
        _frame(),
        SamplingSettings(
            class_name="NK",
            label_id=1,
            windows_per_epoch=6,
            max_windows_per_subject_per_class_per_epoch=3,
            seed=123,
        ),
    )


def test_rotating_sampler_is_deterministic_and_caps_patients():
    first = _sampler()
    second = _sampler()
    assert np.array_equal(first.indices_for_epoch(0), second.indices_for_epoch(0))

    first.set_epoch(0)
    list(iter(first))
    manifest = first.manifest_frame()
    counts = manifest.groupby("label_id").size().to_dict()
    assert counts == {0: 5, 1: 6, 2: 4}
    assert manifest.groupby(["label_id", "subject_id"]).size().max() <= 3


def test_majority_windows_rotate_between_epochs():
    sampler = _sampler()
    epoch_zero = set(
        sampler.frame.iloc[sampler.indices_for_epoch(0)]
        .query("label_id == 1")["study_id"]
        .tolist()
    )
    epoch_one = set(
        sampler.frame.iloc[sampler.indices_for_epoch(1)]
        .query("label_id == 1")["study_id"]
        .tolist()
    )
    assert epoch_zero != epoch_one


def test_sampling_audit_contains_patient_concentration_fields():
    sampler = _sampler()
    sampler.set_epoch(2)
    list(iter(sampler))
    audit = sampler.audit_frame(["HypoK", "NK", "HyperK"])
    assert set(audit["class_name"]) == {"HypoK", "NK", "HyperK"}
    assert set(
        [
            "sampled_windows",
            "independent_patients",
            "max_windows_one_patient",
            "top1_patient_share",
            "top5_patient_share",
        ]
    ).issubset(audit.columns)
