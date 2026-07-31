from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .labels import PotassiumLabeler
from .preprocess import ECGAugmenter, ECGPreprocessor


class MIMIC3ECGPotassiumDataset:
    """Lazy dataset backed by selectively materialized ECG windows."""

    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessor: ECGPreprocessor,
        labeler: PotassiumLabeler,
        augmenter: ECGAugmenter | None = None,
        seed: int = 20260723,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.preprocessor = preprocessor
        self.labeler = labeler
        self.augmenter = augmenter
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _load(cache_path: str) -> tuple[np.ndarray, float]:
        path = Path(cache_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing materialized ECG window: {path}")
        with np.load(path, allow_pickle=False) as payload:
            signal = np.asarray(payload["signal"], dtype=np.float32)
            sampling_rate = float(payload["sampling_rate"])
        return signal, sampling_rate

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        signal, fs = self._load(str(row["waveform_cache_path"]))
        signal = self.preprocessor(signal, fs)
        if self.augmenter is not None:
            signal = self.augmenter(signal, np.random.default_rng(self.seed + index))
        label_id = int(row["label_id"])
        return {
            "ecg": signal,
            "label": np.int64(label_id),
            "ordinal": self.labeler.ordinal_targets([label_id])[0],
            "potassium": np.float32(row["potassium"]),
            "subject_id": np.int64(row["subject_id"]),
            "study_id": np.int64(row["study_id"]),
        }


def load_split_datasets(config: dict) -> dict[str, MIMIC3ECGPotassiumDataset]:
    frame = pd.read_csv(config["data"]["split_csv"])
    required = {"split", "waveform_cache_path", "subject_id", "study_id", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Split file is missing {sorted(missing)}. Run materialize-windows before split."
        )
    labeler = PotassiumLabeler.from_config(config)
    preprocessor = ECGPreprocessor.from_config(config)
    augmenter = (
        ECGAugmenter.from_config(config) if config.get("augmentation", {}).get("enabled") else None
    )
    common = {
        "preprocessor": preprocessor,
        "labeler": labeler,
        "seed": int(config["project"]["seed"]),
    }
    return {
        "train": MIMIC3ECGPotassiumDataset(
            frame[frame["split"] == "train"], augmenter=augmenter, **common
        ),
        "validation": MIMIC3ECGPotassiumDataset(
            frame[frame["split"] == "validation"], augmenter=None, **common
        ),
        "test": MIMIC3ECGPotassiumDataset(
            frame[frame["split"] == "test"], augmenter=None, **common
        ),
    }
