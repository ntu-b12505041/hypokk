from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import apply_calibration
from .model import build_model
from .preprocess import ECGPreprocessor
from .serialization import load_state_dict_h5
from .training import choose_device
from .utils import read_json


def predict_cached_window(
    config: dict,
    input_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> dict:
    import torch

    output_dir = Path(config["project"]["output_dir"]).expanduser().resolve()
    checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path
        else output_dir / "checkpoints" / "model_weights.h5"
    )
    calibration_path = output_dir / "metrics" / "calibration.json"
    if not checkpoint.exists() or not calibration_path.exists():
        raise FileNotFoundError("Missing trained checkpoint or calibration.json")

    model = build_model(config)
    if checkpoint.suffix.lower() == ".h5":
        load_state_dict_h5(model, checkpoint)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model_state_dict", payload))

    with np.load(Path(input_path).expanduser().resolve(), allow_pickle=False) as payload:
        signal = np.asarray(payload["signal"], dtype=np.float32)
        sampling_rate = float(payload["sampling_rate"])
    ecg = ECGPreprocessor.from_config(config)(signal, sampling_rate)
    device = choose_device(config["training"]["device"])
    model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(ecg).unsqueeze(0).to(device))
    logits = output["logits"].cpu().numpy()
    ordinal = output["ordinal_logits"].cpu().numpy()
    potassium = output["potassium"].cpu().numpy()
    prediction, probabilities = apply_calibration(
        logits, ordinal, potassium, read_json(calibration_path)
    )
    names = config["labels"]["names"]
    return {
        "predicted_class_id": int(prediction[0]),
        "predicted_class": names[int(prediction[0])],
        "probabilities": {
            names[index]: float(probabilities[0, index]) for index in range(len(names))
        },
        "predicted_potassium_mmol_l": float(potassium[0]),
        "input_leads": list(config["data"]["lead_order"]),
        "research_only": True,
    }
