from __future__ import annotations

import math
import platform
import time

import numpy as np
import pandas as pd

from .calibration import calibrate_predictions
from .config import ensure_output_dirs
from .dataset import load_split_datasets
from .losses import build_class_weights, build_multitask_loss
from .metrics import classification_metrics
from .model import build_model
from .sampling import RotatingTrainSubsampler, SamplingSettings
from .serialization import export_state_dict_h5
from .utils import seed_everything, write_json


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the project dependencies before training."
        ) from exc
    return torch, DataLoader


def choose_device(requested: str):
    torch, _ = _torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _build_train_sampler(config: dict, train_dataset):
    section = config.get("sampling", {})
    if not section.get("enabled", False):
        return None
    if section.get("strategy") != "rotating_train_subsample":
        raise ValueError(f"Unsupported sampling.strategy: {section.get('strategy')}")
    class_names = list(config["labels"]["names"])
    class_name = str(section["class_name"])
    settings = SamplingSettings(
        class_name=class_name,
        label_id=class_names.index(class_name),
        windows_per_epoch=int(section["windows_per_epoch"]),
        max_windows_per_subject_per_class_per_epoch=int(
            section.get("max_windows_per_subject_per_class_per_epoch", 0)
        ),
        seed=int(config["project"]["seed"]),
    )
    return RotatingTrainSubsampler(train_dataset.frame, settings)


def _make_loaders(config: dict, datasets: dict):
    torch, DataLoader = _torch()
    section = config["training"]
    common = {
        "batch_size": int(section["batch_size"]),
        "num_workers": int(section["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": int(section["num_workers"]) > 0,
    }
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    train_sampler = _build_train_sampler(config, datasets["train"])
    train_loader = DataLoader(
        datasets["train"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        drop_last=False,
        **common,
    )
    loaders = {
        "train": train_loader,
        "validation": DataLoader(datasets["validation"], shuffle=False, drop_last=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **common),
    }
    return loaders, train_sampler


def _optimizer_and_scheduler(config: dict, model, steps_per_epoch: int):
    torch, _ = _torch()
    section = config["training"]
    lr = float(section["learning_rate"])
    weight_decay = float(section["weight_decay"])
    if section["optimizer"].lower() == "adamw":
        model_cfg = config["model"]
        if hasattr(model, "backbone") and model_cfg.get("backbone_learning_rate"):
            backbone_parameters = list(model.backbone.parameters())
            backbone_ids = {id(parameter) for parameter in backbone_parameters}
            head_parameters = [
                parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": backbone_parameters,
                        "lr": float(model_cfg["backbone_learning_rate"]),
                        "name": "backbone",
                    },
                    {
                        "params": head_parameters,
                        "lr": float(model_cfg.get("head_learning_rate", lr)),
                        "name": "heads",
                    },
                ],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {section['optimizer']}")
    epochs = int(section["epochs"])
    warmup_steps = int(section.get("warmup_epochs", 0)) * steps_per_epoch
    total_steps = max(1, epochs * steps_per_epoch)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    scheduler=None,
    scaler=None,
    gradient_clip_norm: float = 1.0,
) -> dict:
    torch, _ = _torch()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    labels, predictions, probabilities = [], [], []
    components = {"classification_loss": 0.0, "ordinal_loss": 0.0, "regression_loss": 0.0}
    use_amp = scaler is not None and scaler.is_enabled()

    for batch in loader:
        ecg = batch["ecg"].to(device=device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if use_amp
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with context:
            outputs = model(ecg)
            loss, loss_components = loss_fn(outputs, batch)
        if training:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = ecg.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        for key, value in loss_components.items():
            components[key] += float(value.cpu()) * batch_size
        probs = torch.softmax(outputs["logits"].detach(), dim=1).cpu().numpy()
        labels.append(batch["label"].cpu().numpy())
        probabilities.append(probs)
        predictions.append(probs.argmax(axis=1))

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    probs = np.concatenate(probabilities)
    metrics = classification_metrics(y_true, y_pred, probs)
    result = {
        "loss": total_loss / max(1, total_examples),
        **{key: value / max(1, total_examples) for key, value in components.items()},
        **{key: value for key, value in metrics.items() if key != "per_class"},
    }
    return result


def collect_predictions(model, loader, device) -> pd.DataFrame:
    torch, _ = _torch()
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["ecg"].to(device=device, dtype=torch.float32))
            logits = outputs["logits"].cpu().numpy()
            ordinal = outputs["ordinal_logits"].cpu().numpy()
            potassium_pred = outputs["potassium"].cpu().numpy()
            batch_size = len(logits)
            for idx in range(batch_size):
                rows.append(
                    {
                        "subject_id": int(batch["subject_id"][idx]),
                        "study_id": int(batch["study_id"][idx]),
                        "label_id": int(batch["label"][idx]),
                        "potassium": float(batch["potassium"][idx]),
                        "predicted_potassium": float(potassium_pred[idx]),
                        **{f"logit_{j}": float(logits[idx, j]) for j in range(3)},
                        **{f"ordinal_logit_{j}": float(ordinal[idx, j]) for j in range(2)},
                    }
                )
    return pd.DataFrame(rows)


def _grad_scaler(torch, use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch compatibility fallback
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_model(config: dict) -> dict:
    torch, _ = _torch()
    import scipy
    import sklearn
    import wfdb

    seed = int(config["project"]["seed"])
    seed_everything(seed)
    output_dir = ensure_output_dirs(config)
    datasets = load_split_datasets(config)
    loaders, train_sampler = _make_loaders(config, datasets)

    if train_sampler is None:
        weighting_labels = datasets["train"].frame["label_id"].to_numpy()
    else:
        initial_indices = train_sampler.indices_for_epoch(0)
        weighting_labels = datasets["train"].frame.iloc[initial_indices]["label_id"].to_numpy()
    class_weights = build_class_weights(config, weighting_labels)

    model = build_model(config)
    freeze_backbone_epochs = int(config["model"].get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        if not hasattr(model, "freeze_backbone"):
            raise ValueError("freeze_backbone_epochs requires a model with freeze_backbone()")
        model.freeze_backbone()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    initially_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    device = choose_device(config["training"]["device"])
    model.to(device)
    loss_fn = build_multitask_loss(config, class_weights)
    optimizer, scheduler = _optimizer_and_scheduler(config, model, len(loaders["train"]))
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = _grad_scaler(torch, use_amp)

    best_score = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    patience = int(config["training"]["early_stopping_patience"])
    history = []
    sampling_audits = []
    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    sampling_cfg = config.get("sampling", {})
    manifest_dir = output_dir / "logs" / "sampling_manifests"
    if train_sampler is not None and sampling_cfg.get("save_epoch_manifests", True):
        manifest_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            model.unfreeze_backbone()
        train_stats = _run_epoch(
            model,
            loaders["train"],
            loss_fn,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        if train_sampler is not None:
            sampling_audits.append(train_sampler.audit_frame(list(config["labels"]["names"])))
            if sampling_cfg.get("save_epoch_manifests", True):
                train_sampler.manifest_frame().to_csv(
                    manifest_dir / f"epoch_{epoch:03d}.csv", index=False
                )
        with torch.inference_mode():
            val_stats = _run_epoch(model, loaders["validation"], loss_fn, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update(
            {f"train_{key}": value for key, value in train_stats.items() if np.isscalar(value)}
        )
        row.update({f"val_{key}": value for key, value in val_stats.items() if np.isscalar(value)})
        history.append(row)
        score = float(val_stats.get("macro_auroc_ovr", np.nan))
        if not np.isfinite(score):
            score = float(val_stats["balanced_accuracy"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_score": score,
                    "config": config,
                    "class_weights": class_weights.tolist(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    elapsed = time.perf_counter() - started
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "logs" / "training_history.csv", index=False)
    if sampling_audits:
        pd.concat(sampling_audits, ignore_index=True).to_csv(
            output_dir / "logs" / "sampling_audit.csv", index=False
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    h5_checkpoint_path = export_state_dict_h5(
        model,
        output_dir / "checkpoints" / "model_weights.h5",
        {
            "model_name": config["model"]["name"],
            "best_epoch": checkpoint["epoch"],
            "mimic_clinical_version": config["data"]["mimic_clinical_version"],
            "mimic_waveform_version": config["data"]["mimic_waveform_version"],
            "lead_order": config["data"]["lead_order"],
        },
    )
    validation_predictions = collect_predictions(model, loaders["validation"], device)
    logits = validation_predictions[[f"logit_{idx}" for idx in range(3)]].to_numpy()
    ordinal_logits = validation_predictions[[f"ordinal_logit_{idx}" for idx in range(2)]].to_numpy()
    calibration = calibrate_predictions(
        validation_predictions["label_id"].to_numpy(),
        logits,
        ordinal_logits,
        validation_predictions["predicted_potassium"].to_numpy(),
        config,
    )
    write_json(output_dir / "metrics" / "calibration.json", calibration.to_dict())
    validation_predictions.to_csv(
        output_dir / "metrics" / "validation_predictions.csv", index=False
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "wfdb": wfdb.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
    }
    sampled_counts = np.bincount(
        np.asarray(weighting_labels, dtype=int), minlength=int(config["model"]["num_classes"])
    )
    summary = {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "epochs_completed": len(history),
        "checkpoint": str(checkpoint_path),
        "h5_checkpoint": str(h5_checkpoint_path),
        "class_weight_method": config["training"].get("class_weight_method", "effective_number"),
        "class_weights": class_weights.tolist(),
        "weighting_class_counts": sampled_counts.tolist(),
        "sampling": sampling_cfg if train_sampler is not None else {"enabled": False},
        "environment": environment,
        "validation_target_met": calibration.target_met_on_validation,
        "total_parameters": int(total_parameters),
        "initially_trainable_parameters": int(initially_trainable_parameters),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "model_name": config["model"]["name"],
    }
    write_json(output_dir / "logs" / "training_summary.json", summary)
    return summary
