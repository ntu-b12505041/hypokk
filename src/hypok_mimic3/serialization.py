from __future__ import annotations

import json
from pathlib import Path


def export_state_dict_h5(model, path: str | Path, metadata: dict | None = None) -> Path:
    """Export a PyTorch state_dict to HDF5.

    The file is intentionally named .h5 for the requested delivery format, but
    it is not a Keras model. Use load_state_dict_h5 or the project inference CLI.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for .h5 model export") from exc

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.attrs["format"] = "hypok-mimic3-pytorch-state-dict-v1"
        handle.attrs["metadata_json"] = json.dumps(metadata or {}, default=str)
        group = handle.create_group("state_dict")
        for name, tensor in model.state_dict().items():
            group.create_dataset(name, data=tensor.detach().cpu().numpy())
    return target


def load_state_dict_h5(model, path: str | Path):
    try:
        import h5py
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py and PyTorch are required to load .h5 weights") from exc

    with h5py.File(Path(path).expanduser().resolve(), "r") as handle:
        if handle.attrs.get("format") != "hypok-mimic3-pytorch-state-dict-v1":
            raise ValueError("Unsupported HDF5 model format")
        state = {
            name: torch.from_numpy(handle["state_dict"][name][()])
            for name in handle["state_dict"].keys()
        }
    model.load_state_dict(state, strict=True)
    return model
