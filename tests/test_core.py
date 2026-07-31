from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hypok_mimic3.calibration import tune_ordered_thresholds
from hypok_mimic3.config import load_config
from hypok_mimic3.labels import PotassiumLabeler
from hypok_mimic3.metrics import classification_metrics, target_is_met
from hypok_mimic3.mimic3 import (
    _drop_near_duplicate_windows,
    _remote_record_locator,
    _subject_from_record_path,
    build_potassium_cohort,
)
from hypok_mimic3.preprocess import ECGPreprocessor
from hypok_mimic3.splits import make_patient_level_splits, write_split_audit


class LabelTests(unittest.TestCase):
    def test_clinical_boundaries(self):
        labeler = PotassiumLabeler(hypokalemia_upper=3.5, hyperkalemia_lower=5.5)
        result = labeler.transform([2.9, 3.499, 3.5, 5.499, 5.5, 7.0])
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2, 2])


class SplitTests(unittest.TestCase):
    def test_patient_leakage_is_impossible(self):
        rows = []
        for patient in range(120):
            label = patient % 3
            for study in range(2):
                rows.append(
                    {
                        "subject_id": patient,
                        "study_id": patient * 10 + study,
                        "label_id": label,
                    }
                )
        split, _ = make_patient_level_splits(pd.DataFrame(rows), seed=7)
        counts = split.groupby("subject_id")["split"].nunique()
        self.assertTrue((counts == 1).all())
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})


class PreprocessingTests(unittest.TestCase):
    def test_resample_shape_and_finite_values(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(0, 0.1, size=(5000, 12))
        preprocessor = ECGPreprocessor(target_sampling_rate=250, duration_seconds=10)
        result = preprocessor(signal, 500)
        self.assertEqual(result.shape, (12, 2500))
        self.assertTrue(np.isfinite(result).all())

    def test_single_lead_bedside_shape(self):
        rng = np.random.default_rng(4)
        signal = rng.normal(0, 0.1, size=(1250, 1))
        preprocessor = ECGPreprocessor(
            target_sampling_rate=125,
            duration_seconds=10,
            bandpass_low_hz=0.5,
            bandpass_high_hz=40,
            normalization="none",
            profile="standard",
        )
        result = preprocessor(signal, 125)
        self.assertEqual(result.shape, (1, 1250))
        self.assertTrue(np.isfinite(result).all())


class ConfigurationTests(unittest.TestCase):
    def test_full_and_demo_configs_are_valid(self):
        root = Path(__file__).resolve().parents[1]
        full = load_config(root / "configs" / "mimic3.yaml")
        demo = load_config(root / "configs" / "mimic3_demo.yaml")
        self.assertEqual(full["data"]["mimic_clinical_version"], "1.4")
        self.assertEqual(demo["model"]["input_leads"], 1)


class MIMIC3Tests(unittest.TestCase):
    def test_subject_id_is_parsed_from_matched_path(self):
        path = "p04/p044083/p044083-2112-05-04-19-50"
        self.assertEqual(_subject_from_record_path(path), 44083)

    def test_nested_physionet_locator(self):
        name, directory = _remote_record_locator(
            "p04/p044083/p044083-2112-05-04-19-50",
            "mimic3wdb-matched/1.0",
        )
        self.assertEqual(name, "p044083-2112-05-04-19-50")
        self.assertEqual(directory, "mimic3wdb-matched/1.0/p04/p044083")

    def test_near_duplicate_windows_are_removed(self):
        frame = pd.DataFrame(
            {
                "waveform_record": ["a", "a", "a", "b"],
                "subject_id": [1, 1, 1, 2],
                "labevent_id": [1, 2, 3, 4],
                "ecg_anchor_time": pd.to_datetime(
                    [
                        "2100-01-01 00:00:00",
                        "2100-01-01 00:00:10",
                        "2100-01-01 00:00:45",
                        "2100-01-01 00:00:05",
                    ]
                ),
            }
        )
        result = _drop_near_duplicate_windows(frame, 30)
        self.assertEqual(set(result["labevent_id"]), {1, 3, 4})

    def test_split_audit_reports_patient_concentration(self):
        rows = []
        for patient in range(30):
            for label in range(3):
                rows.append(
                    {
                        "split": ("train", "validation", "test")[patient % 3],
                        "label_id": label,
                        "subject_id": patient,
                    }
                )
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            audit = write_split_audit(pd.DataFrame(rows), directory)
            self.assertIn("HypoK", audit["train"])
            self.assertTrue(Path(directory, "patient_contributions.csv").exists())

    def test_synthetic_mimic3_tables_pair_to_waveform_coverage(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clinical = root / "clinical"
            clinical.mkdir()
            pd.DataFrame(
                {
                    "ROW_ID": [1, 2, 3],
                    "SUBJECT_ID": [100001, 100002, 100003],
                    "HADM_ID": [200001, 200002, 200003],
                    "ITEMID": [50971, 50971, 50971],
                    "CHARTTIME": [
                        "2100-01-01 00:05:00",
                        "2100-01-01 00:05:00",
                        "2100-01-01 00:05:00",
                    ],
                    "VALUENUM": [3.2, 4.2, 6.0],
                    "VALUEUOM": ["mEq/L", "mEq/L", "mEq/L"],
                    "FLAG": ["abnormal", "", "abnormal"],
                }
            ).to_csv(clinical / "LABEVENTS.csv", index=False)
            pd.DataFrame(
                {
                    "ITEMID": [50971],
                    "LABEL": ["Potassium"],
                    "FLUID": ["Blood"],
                    "CATEGORY": ["Chemistry"],
                }
            ).to_csv(clinical / "D_LABITEMS.csv", index=False)
            index_path = root / "waveforms.csv"
            pd.DataFrame(
                {
                    "subject_id": [100001, 100002, 100003],
                    "waveform_record": ["a", "b", "c"],
                    "record_start_time": ["2100-01-01 00:00:00"] * 3,
                    "record_end_time": ["2100-01-01 00:10:00"] * 3,
                    "sampling_rate": [125.0] * 3,
                    "signal_length": [75000] * 3,
                    "lead_names": ["II"] * 3,
                    "index_error": [""] * 3,
                }
            ).to_csv(index_path, index=False)
            output = root / "cohort.csv"
            config = {
                "data": {
                    "waveform_index_csv": str(index_path),
                    "clinical_root": str(clinical),
                    "cohort_manifest_csv": str(output),
                    "mimic_waveform_version": "1.0",
                    "mimic_clinical_version": "1.4",
                    "lab_window_minutes": 60,
                    "minimum_window_separation_seconds": 30,
                    "include_whole_blood": False,
                    "potassium_itemids": {"serum": [50971], "whole_blood": [50822]},
                    "min_potassium": 1.5,
                    "max_potassium": 10.0,
                },
                "labels": {
                    "hypokalemia_upper": 3.5,
                    "hyperkalemia_lower": 5.5,
                    "hypokalemia_inclusive": False,
                    "hyperkalemia_inclusive": True,
                    "names": ["HypoK", "NK", "HyperK"],
                },
                "preprocess": {"duration_seconds": 10},
            }
            cohort, summary = build_potassium_cohort(config)
            self.assertEqual(len(cohort), 3)
            self.assertEqual(set(cohort["label"]), {"HypoK", "NK", "HyperK"})
            self.assertTrue((cohort["abs_delta_minutes"] == 0).all())
            self.assertEqual(summary["subjects"], 3)


class MetricsTests(unittest.TestCase):
    def test_specificity_and_strict_target(self):
        y_true = np.repeat(np.arange(3), 20)
        y_pred = y_true.copy()
        metrics = classification_metrics(y_true, y_pred)
        self.assertTrue(target_is_met(metrics, 0.85, 0.85))
        for values in metrics["per_class"].values():
            values["recall"] = 0.85
        self.assertFalse(target_is_met(metrics, 0.85, 0.85))

    def test_ordered_threshold_tuning(self):
        y_true = np.repeat(np.arange(3), 25)
        scores = np.concatenate(
            [np.linspace(0.0, 0.4, 25), np.linspace(0.8, 1.2, 25), np.linspace(1.6, 2.0, 25)]
        )
        result = tune_ordered_thresholds(y_true, scores, grid_size=41)
        self.assertTrue(result["low_threshold"] < result["high_threshold"])
        self.assertTrue(result["target_met"])


if __name__ == "__main__":
    unittest.main()
