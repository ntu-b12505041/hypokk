from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve project-relative paths at runtime."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_meta"] = {"config_path": str(config_path)}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("project", "data", "labels", "split", "preprocess", "model", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    ratios = [
        float(config["split"]["train_ratio"]),
        float(config["split"]["validation_ratio"]),
        float(config["split"]["test_ratio"]),
    ]
    if abs(sum(ratios) - 1.0) > 1e-8 or any(r <= 0 for r in ratios):
        raise ValueError(f"Split ratios must be positive and sum to 1; got {ratios}")

    low = float(config["labels"]["hypokalemia_upper"])
    high = float(config["labels"]["hyperkalemia_lower"])
    if low >= high:
        raise ValueError("Hypokalemia threshold must be below hyperkalemia threshold")

    data = config["data"]
    if data.get("mimic_clinical_version") != "1.4":
        raise ValueError("This separate project requires MIMIC-III Clinical v1.4")
    if data.get("mimic_waveform_version") != "1.0":
        raise ValueError("This separate project requires MIMIC-III Matched Waveform v1.0")
    if not data.get("lead_order"):
        raise ValueError("data.lead_order must contain at least one bedside ECG lead")

    model_name = config["model"]["name"]
    if model_name != "se_resnet1d_multitask":
        raise ValueError(f"Unsupported model.name: {model_name}")
    if int(config["model"]["input_leads"]) != len(data["lead_order"]):
        raise ValueError("model.input_leads must equal len(data.lead_order)")
    if float(config["preprocess"]["duration_seconds"]) <= 0:
        raise ValueError("preprocess.duration_seconds must be positive")

    class_weight_method = str(
        config["training"].get("class_weight_method", "effective_number")
    ).lower()
    supported_weights = {"effective_number", "sqrt_inverse_frequency", "none"}
    if class_weight_method not in supported_weights:
        raise ValueError(
            f"training.class_weight_method must be one of {sorted(supported_weights)}"
        )

    sampling = config.get("sampling", {})
    if sampling.get("enabled", False):
        if sampling.get("strategy") != "rotating_train_subsample":
            raise ValueError(
                "Enabled sampling.strategy must be 'rotating_train_subsample'"
            )
        class_name = str(sampling.get("class_name", ""))
        if class_name not in config["labels"]["names"]:
            raise ValueError("sampling.class_name must be present in labels.names")
        if int(sampling.get("windows_per_epoch", 0)) <= 0:
            raise ValueError("sampling.windows_per_epoch must be positive")
        if int(sampling.get("max_windows_per_subject_per_class_per_epoch", 0)) < 0:
            raise ValueError(
                "sampling.max_windows_per_subject_per_class_per_epoch must be non-negative"
            )


def ensure_output_dirs(config: dict[str, Any]) -> Path:
    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve()
    for name in ("checkpoints", "figures", "metrics", "reports", "logs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    return output_dir
