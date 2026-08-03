from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingSettings:
    class_name: str
    label_id: int
    windows_per_epoch: int
    max_windows_per_subject_per_class_per_epoch: int
    seed: int


class RotatingTrainSubsampler:
    """Deterministic epoch-wise sampler that only downsamples one training class.

    Minority classes are retained subject to the same per-patient cap. The selected
    majority-class windows rotate by epoch, so no permanent reduced cohort is made.
    Validation and test loaders never use this sampler.
    """

    def __init__(self, frame: pd.DataFrame, settings: SamplingSettings) -> None:
        required = {"subject_id", "study_id", "label_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Training frame is missing sampling columns: {sorted(missing)}")
        if settings.windows_per_epoch <= 0:
            raise ValueError("sampling.windows_per_epoch must be positive")
        if settings.max_windows_per_subject_per_class_per_epoch < 0:
            raise ValueError(
                "sampling.max_windows_per_subject_per_class_per_epoch must be non-negative"
            )
        labels = frame["label_id"].astype(int)
        if settings.label_id not in set(labels):
            raise ValueError(
                f"Sampling class {settings.class_name!r} (label {settings.label_id}) "
                "is absent from the training split"
            )
        self.frame = frame.reset_index(drop=True)
        self.settings = settings
        self.epoch = 0
        self.last_indices = np.empty(0, dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _subject_capped_indices(
        self, indices: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        cap = self.settings.max_windows_per_subject_per_class_per_epoch
        if cap == 0:
            return indices.copy()
        selected: list[np.ndarray] = []
        subset = self.frame.iloc[indices]
        for _, group in subset.groupby("subject_id", sort=True):
            group_indices = group.index.to_numpy(dtype=np.int64)
            rng.shuffle(group_indices)
            selected.append(group_indices[:cap])
        if not selected:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(selected)

    def indices_for_epoch(self, epoch: int | None = None) -> np.ndarray:
        epoch = self.epoch if epoch is None else int(epoch)
        rng = np.random.default_rng(self.settings.seed + epoch)
        labels = self.frame["label_id"].to_numpy(dtype=int)
        selected: list[np.ndarray] = []
        for label_id in sorted(np.unique(labels)):
            candidates = np.flatnonzero(labels == label_id).astype(np.int64)
            candidates = self._subject_capped_indices(candidates, rng)
            if label_id == self.settings.label_id:
                target = min(self.settings.windows_per_epoch, len(candidates))
                candidates = rng.choice(candidates, size=target, replace=False)
            selected.append(np.asarray(candidates, dtype=np.int64))
        indices = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
        rng.shuffle(indices)
        return indices

    def __iter__(self):
        self.last_indices = self.indices_for_epoch()
        return iter(self.last_indices.tolist())

    def __len__(self) -> int:
        return int(len(self.indices_for_epoch()))

    def manifest_frame(self) -> pd.DataFrame:
        indices = self.last_indices if len(self.last_indices) else self.indices_for_epoch()
        columns = ["subject_id", "study_id", "label_id"]
        manifest = self.frame.iloc[indices][columns].copy()
        manifest.insert(0, "dataset_index", indices)
        manifest.insert(0, "epoch", self.epoch + 1)
        return manifest.reset_index(drop=True)

    def audit_frame(self, class_names: list[str]) -> pd.DataFrame:
        manifest = self.manifest_frame()
        rows = []
        available = self.frame.groupby("label_id").size().to_dict()
        for label_id, group in manifest.groupby("label_id", sort=True):
            contributions = group.groupby("subject_id").size().sort_values(ascending=False)
            total = int(len(group))
            rows.append(
                {
                    "epoch": self.epoch + 1,
                    "class_name": class_names[int(label_id)],
                    "label_id": int(label_id),
                    "available_windows": int(available.get(int(label_id), 0)),
                    "sampled_windows": total,
                    "independent_patients": int(group["subject_id"].nunique()),
                    "max_windows_one_patient": int(contributions.max()),
                    "median_windows_per_patient": float(contributions.median()),
                    "top1_patient_share": float(contributions.iloc[:1].sum() / total),
                    "top5_patient_share": float(contributions.iloc[:5].sum() / total),
                }
            )
        return pd.DataFrame(rows)
