"""Training engine."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.engine.evaluator import evaluate
from src.losses.losses import get_loss_runtime_metadata
from src.utils.checkpoint import save_checkpoint
from src.utils.ema import ModelEMA
from src.utils.metrics import AverageMeter, topk_accuracy
from src.utils.misc import save_json
from src.utils.run_metadata import list_checkpoint_files, write_run_metadata


def build_optimizer(model: torch.nn.Module, optimizer_cfg: Dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    name = optimizer_cfg.get("name", "adamw").lower()
    lr = float(optimizer_cfg.get("lr", 1e-3))
    wd = float(optimizer_cfg.get("weight_decay", 0.0))
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain after staged-training scope configuration.")

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: Dict,
    total_epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build scheduler from config (cosine with warmup)."""
    name = scheduler_cfg.get("name", "cosine").lower()
    if name == "none":
        return None

    if name == "cosine":
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.0))
        total_epochs = max(total_epochs, 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler: {name}")


_SELECTION_METRIC_ALIASES = {
    "acc": "val_acc",
    "acc1": "val_acc",
    "val_acc": "val_acc",
    "val_acc1": "val_acc",
    "macro_recall": "val_macro_recall",
    "val_macro_recall": "val_macro_recall",
    "macro_per_class_acc": "val_macro_recall",
    "nll": "val_nll",
    "val_nll": "val_nll",
    "ece": "val_ece",
    "val_ece": "val_ece",
    "loss": "val_loss",
    "val_loss": "val_loss",
}
_MAXIMIZE_METRICS = {"val_acc", "val_macro_recall"}
_MINIMIZE_METRICS = {"val_loss", "val_nll", "val_ece"}


def _canonical_selection_metric(metric_name: str) -> str:
    normalized = str(metric_name).strip().lower()
    if normalized not in _SELECTION_METRIC_ALIASES:
        raise ValueError(f"Unsupported model-selection metric: {metric_name}")
    return _SELECTION_METRIC_ALIASES[normalized]


def _selection_mode(metric_name: str, configured_mode: str = "auto") -> str:
    mode = str(configured_mode).lower()
    if mode not in {"auto", "min", "max"}:
        raise ValueError(f"Unsupported model-selection mode: {configured_mode}")
    if mode != "auto":
        return mode
    if metric_name in _MAXIMIZE_METRICS:
        return "max"
    if metric_name in _MINIMIZE_METRICS:
        return "min"
    raise ValueError(f"Cannot infer optimization mode for metric: {metric_name}")


def _is_better(candidate: float, best: float, mode: str) -> bool:
    return candidate > best if mode == "max" else candidate < best


def _selection_metric_details(selection_cfg: Dict) -> tuple[str, str]:
    metric_name = _canonical_selection_metric(selection_cfg.get("metric", "val_acc"))
    mode = _selection_mode(metric_name, selection_cfg.get("mode", "auto"))
    return metric_name, mode


def _validation_metric_block(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "val_loss": float(metrics.get("loss", 0.0)),
        "val_acc": float(metrics.get("acc1", 0.0)),
        "val_macro_recall": float(metrics.get("macro_recall", 0.0)),
        "val_nll": float(metrics.get("nll", 0.0)),
        "val_ece": float(metrics.get("ece", 0.0)),
        "val_acc5": float(metrics.get("acc5", 0.0)),
    }


def _select_metric_block(
    raw_metrics: Dict[str, float],
    ema_metrics: Optional[Dict[str, float]],
    selection_cfg: Dict,
) -> tuple[str, Dict[str, float], str, str, float]:
    source = str(selection_cfg.get("source", "auto")).lower()
    if source not in {"auto", "raw", "ema"}:
        raise ValueError(f"Unsupported model-selection source: {source}")

    metric_name = _canonical_selection_metric(selection_cfg.get("metric", "val_acc"))
    mode = _selection_mode(metric_name, selection_cfg.get("mode", "auto"))

    if source == "raw":
        selected_source = "raw"
        selected_metrics = raw_metrics
    elif source == "ema":
        if ema_metrics is None:
            raise ValueError("Model-selection source 'ema' requested but EMA metrics are unavailable.")
        selected_source = "ema"
        selected_metrics = ema_metrics
    else:
        if ema_metrics is not None:
            selected_source = "ema"
            selected_metrics = ema_metrics
        else:
            selected_source = "raw"
            selected_metrics = raw_metrics

    if metric_name not in selected_metrics:
        raise ValueError(f"Selected metric '{metric_name}' is unavailable in validation metrics.")
    return selected_source, selected_metrics, metric_name, mode, float(selected_metrics[metric_name])


def _retain_top_k_checkpoints(
    records: list[Dict],
    keep_top_k: int,
    mode: str,
) -> list[Dict]:
    if keep_top_k <= 0 or len(records) <= keep_top_k:
        return records

    reverse = mode == "max"
    ranked = sorted(records, key=lambda item: (item["selection_metric_value"], -item["epoch"]), reverse=reverse)
    keep = ranked[:keep_top_k]
    keep_paths = {record["path"] for record in keep}
    for record in records:
        if record["path"] not in keep_paths:
            path = Path(record["path"])
            if path.exists():
                path.unlink()
    return sorted(keep, key=lambda item: item["epoch"])


def _rand_bbox(size, lam):
    _, _, h, w = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    return x1, y1, x2, y2


def _apply_mixup_or_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mix_cfg: Dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    if not bool(mix_cfg.get("enabled", False)):
        return images, targets, targets, 1.0, False

    if random.random() > float(mix_cfg.get("prob", 0.5)):
        return images, targets, targets, 1.0, False

    mixup_alpha = float(mix_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(mix_cfg.get("cutmix_alpha", 0.0))

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)
    indices = torch.randperm(images.size(0), device=images.device)
    targets_a = targets
    targets_b = targets[indices]

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        images = lam * images + (1.0 - lam) * images[indices]

    return images, targets_a, targets_b, float(lam), True


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    train_cfg: Dict,
    mix_cfg: Dict,
    scaler: torch.amp.GradScaler,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    log_interval = int(train_cfg.get("log_interval", 50))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    amp = bool(train_cfg.get("amp", True))

    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{total_epochs}", leave=False)
    for step, (images, targets) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        images, targets_a, targets_b, lam, is_mixed = _apply_mixup_or_cutmix(images, targets, mix_cfg)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = amp and device.type == "cuda"
        with torch.amp.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
        ):
            logits = model(images)
            if is_mixed:
                loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        metric_targets = targets if not is_mixed else targets_a
        acc1 = topk_accuracy(logits.detach(), metric_targets, topk=(1,))[0]

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(acc1, n=images.size(0))

        if step % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc1": f"{acc_meter.avg:.2f}"})

    return {"loss": float(loss_meter.avg), "acc1": float(acc_meter.avg)}


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    config: Dict,
    run_dir: Path,
    logger,
) -> Dict:
    """Train and validate model, save checkpoints, and return summary."""
    train_cfg = config["train"]
    mix_cfg = config.get("mixup_cutmix", {})
    total_epochs = int(train_cfg.get("epochs", config.get("scheduler", {}).get("epochs", 20)))
    num_classes = int(config["model"]["num_classes"])
    selection_cfg = dict(train_cfg.get("model_selection", {}))
    checkpoint_cfg = dict(train_cfg.get("checkpointing", {}))
    staged_runtime = config.get("staged_training_runtime", {})

    selection_metric_name, selection_mode_name = _selection_metric_details(selection_cfg)
    selection_best_value = float("-inf") if selection_mode_name == "max" else float("inf")

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda",
    )
    use_ema = bool(train_cfg.get("ema", {}).get("enabled", False))
    ema = ModelEMA(model, decay=float(train_cfg["ema"].get("decay", 0.9998))) if use_ema else None

    best_epoch = -1
    best_selected_acc = -1.0
    best_metric_source = "raw"
    best_metrics_selected: Dict[str, float] = {}

    best_raw_acc = -1.0
    best_raw_epoch = -1
    best_raw_metrics: Dict[str, float] = {}

    best_ema_acc = -1.0
    best_ema_epoch = -1
    best_ema_metrics: Dict[str, float] = {}

    patience = int(train_cfg.get("early_stopping", {}).get("patience", 5))
    early_stop_enabled = bool(train_cfg.get("early_stopping", {}).get("enabled", False))
    patience_counter = 0

    history = []
    checkpoint_dir = run_dir / "checkpoints"
    best_ckpt_path = checkpoint_dir / "best.ckpt"
    best_raw_ckpt_path = checkpoint_dir / "best_raw.ckpt"
    best_ema_ckpt_path = checkpoint_dir / "best_ema.ckpt"
    last_ckpt_path = checkpoint_dir / "last.ckpt"
    retained_epoch_checkpoints: list[Dict] = []

    for epoch in range(total_epochs):
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        loss_runtime = get_loss_runtime_metadata(config.get("loss", {}), criterion)
        drw_runtime = loss_runtime.get("deferred_reweighting", {})

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=total_epochs,
            train_cfg=train_cfg,
            mix_cfg=mix_cfg,
            scaler=scaler,
            ema=ema,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            amp=bool(train_cfg.get("amp", True)),
            num_classes=num_classes,
            desc="Validation",
        )
        raw_metric_block = _validation_metric_block(val_metrics)

        ema_metric_block = None
        if ema is not None:
            ema_metrics = evaluate(
                model=ema.ema_model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                amp=bool(train_cfg.get("amp", True)),
                num_classes=num_classes,
                desc="Validation (EMA)",
            )
            ema_metric_block = _validation_metric_block(ema_metrics)

        if scheduler is not None:
            scheduler.step()

        is_best_raw = raw_metric_block["val_acc"] > best_raw_acc
        if is_best_raw:
            best_raw_acc = raw_metric_block["val_acc"]
            best_raw_epoch = epoch
            best_raw_metrics = dict(raw_metric_block)

        is_best_ema = ema_metric_block is not None and ema_metric_block["val_acc"] > best_ema_acc
        if is_best_ema and ema_metric_block is not None:
            best_ema_acc = ema_metric_block["val_acc"]
            best_ema_epoch = epoch
            best_ema_metrics = dict(ema_metric_block)

        (
            selection_source,
            selected_metric_block,
            selection_metric_name,
            selection_mode_name,
            selection_metric_value,
        ) = _select_metric_block(
            raw_metrics=raw_metric_block,
            ema_metrics=ema_metric_block,
            selection_cfg=selection_cfg,
        )

        is_best_selected = _is_better(selection_metric_value, selection_best_value, selection_mode_name)
        if is_best_selected:
            selection_best_value = selection_metric_value
            best_epoch = epoch
            best_selected_acc = selected_metric_block["val_acc"]
            best_metric_source = selection_source
            best_metrics_selected = dict(selected_metric_block)
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc1": train_metrics["acc1"],
            "val_loss": raw_metric_block["val_loss"],
            "val_acc": raw_metric_block["val_acc"],
            "val_acc1": raw_metric_block["val_acc"],
            "val_acc5": raw_metric_block["val_acc5"],
            "val_macro_recall": raw_metric_block["val_macro_recall"],
            "val_macro_per_class_acc": float(val_metrics.get("macro_per_class_acc", 0.0)),
            "val_nll": raw_metric_block["val_nll"],
            "val_ece": raw_metric_block["val_ece"],
            "val_acc_ema": ema_metric_block["val_acc"] if ema_metric_block is not None else None,
            "val_acc1_ema": ema_metric_block["val_acc"] if ema_metric_block is not None else None,
            "val_acc5_ema": ema_metric_block["val_acc5"] if ema_metric_block is not None else None,
            "val_macro_recall_ema": (
                ema_metric_block["val_macro_recall"] if ema_metric_block is not None else None
            ),
            "val_nll_ema": ema_metric_block["val_nll"] if ema_metric_block is not None else None,
            "val_ece_ema": ema_metric_block["val_ece"] if ema_metric_block is not None else None,
            "model_selection_metric": selection_metric_name,
            "model_selection_source": selection_source,
            "model_selection_value": selection_metric_value,
            "best_metric_so_far": selection_best_value,
            "best_acc_so_far": best_selected_acc,
            "loss_name": loss_runtime.get("loss_name"),
            "loss_family": loss_runtime.get("loss_family"),
            "loss_class_weight_source": loss_runtime.get("class_weight_source"),
            "loss_active_class_weight_min": loss_runtime.get("active_class_weight_min"),
            "loss_active_class_weight_max": loss_runtime.get("active_class_weight_max"),
            "loss_drw_enabled": drw_runtime.get("enabled", False),
            "loss_drw_active": drw_runtime.get("active", False),
            "loss_drw_start_epoch": drw_runtime.get("start_epoch"),
        }
        history.append(epoch_record)

        checkpoint_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_selected_acc,
            "best_metric_name": selection_metric_name,
            "best_metric_mode": selection_mode_name,
            "best_metric_value": selection_best_value,
            "current_metric_source": selection_source,
            "current_metric_value": selection_metric_value,
            "current_metrics": selected_metric_block,
            "config": config,
            "history": history,
            "staged_training_runtime": staged_runtime,
            "loss_runtime": loss_runtime,
        }
        if ema is not None:
            checkpoint_state["ema_state_dict"] = ema.state_dict()

        save_checkpoint(checkpoint_state, last_ckpt_path)
        if is_best_selected:
            save_checkpoint(checkpoint_state, best_ckpt_path)
        if is_best_raw:
            save_checkpoint(checkpoint_state, best_raw_ckpt_path)
        if is_best_ema:
            save_checkpoint(checkpoint_state, best_ema_ckpt_path)

        if bool(checkpoint_cfg.get("save_every_epoch", False)):
            epoch_ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.ckpt"
            save_checkpoint(checkpoint_state, epoch_ckpt_path)
            retained_epoch_checkpoints.append(
                {
                    "epoch": epoch + 1,
                    "path": str(epoch_ckpt_path),
                    "selection_metric_name": selection_metric_name,
                    "selection_metric_source": selection_source,
                    "selection_metric_value": selection_metric_value,
                }
            )
            retained_epoch_checkpoints = _retain_top_k_checkpoints(
                records=retained_epoch_checkpoints,
                keep_top_k=int(checkpoint_cfg.get("keep_top_k", 0)),
                mode=selection_mode_name,
            )

        raw_log = (
            f"raw[val_acc={raw_metric_block['val_acc']:.2f} "
            f"macro_recall={raw_metric_block['val_macro_recall']:.2f} "
            f"nll={raw_metric_block['val_nll']:.4f} "
            f"ece={raw_metric_block['val_ece']:.4f}]"
        )
        if ema_metric_block is not None:
            ema_log = (
                f"ema[val_acc={ema_metric_block['val_acc']:.2f} "
                f"macro_recall={ema_metric_block['val_macro_recall']:.2f} "
                f"nll={ema_metric_block['val_nll']:.4f} "
                f"ece={ema_metric_block['val_ece']:.4f}]"
            )
        else:
            ema_log = "ema[disabled]"
        loss_log = (
            f"loss[name={loss_runtime.get('loss_name')} "
            f"class_weight_source={loss_runtime.get('class_weight_source')} "
            f"drw_active={drw_runtime.get('active', False)}"
        )
        if drw_runtime.get("enabled", False):
            loss_log += f" start_epoch={drw_runtime.get('start_epoch')}]"
        else:
            loss_log += "]"

        logger.info(
            "Epoch %d/%d | lr=%.6f | train_loss=%.4f | train_acc=%.2f | %s | %s | %s | "
            "select[%s/%s]=%.4f | best=%.4f",
            epoch + 1,
            total_epochs,
            lr,
            train_metrics["loss"],
            train_metrics["acc1"],
            loss_log,
            raw_log,
            ema_log,
            selection_source,
            selection_metric_name,
            selection_metric_value,
            selection_best_value,
        )

        if early_stop_enabled and patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d) using %s/%s.",
                epoch + 1,
                patience,
                selection_source,
                selection_metric_name,
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    save_json({"history": history}, run_dir / "history.json")

    checkpoint_files = list_checkpoint_files(checkpoint_dir)
    loss_runtime = get_loss_runtime_metadata(config.get("loss", {}), criterion)
    summary = {
        "best_acc1": float(best_selected_acc),
        "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else None,
        "best_checkpoint": str(best_ckpt_path),
        "best_metric_name": selection_metric_name,
        "best_metric_mode": selection_mode_name,
        "best_metric_source": best_metric_source,
        "best_metric_value": float(selection_best_value),
        "best_metrics_selected": best_metrics_selected,
        "best_raw_acc1": float(best_raw_acc),
        "best_raw_epoch": int(best_raw_epoch + 1) if best_raw_epoch >= 0 else None,
        "best_raw_checkpoint": str(best_raw_ckpt_path),
        "best_raw_metrics": best_raw_metrics,
        "best_ema_acc1": float(best_ema_acc) if best_ema_acc >= 0 else None,
        "best_ema_epoch": int(best_ema_epoch + 1) if best_ema_epoch >= 0 else None,
        "best_ema_checkpoint": str(best_ema_ckpt_path) if best_ema_acc >= 0 else None,
        "best_ema_metrics": best_ema_metrics if best_ema_metrics else None,
        "last_checkpoint": str(last_ckpt_path),
        "history_path": str(run_dir / "history.csv"),
        "checkpoint_files": checkpoint_files,
        "retained_epoch_checkpoints": retained_epoch_checkpoints,
        "checkpointing": {
            "save_every_epoch": bool(checkpoint_cfg.get("save_every_epoch", False)),
            "keep_top_k": int(checkpoint_cfg.get("keep_top_k", 0)),
        },
        "loss_runtime": loss_runtime,
        "staged_training": staged_runtime,
    }
    save_json(summary, run_dir / "summary.json")
    write_run_metadata(
        run_dir=run_dir,
        config=config,
        summary=summary,
        checkpoint_files=checkpoint_files,
    )
    return summary
