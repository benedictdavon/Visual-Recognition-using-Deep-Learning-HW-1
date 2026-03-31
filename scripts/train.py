"""Training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (  # noqa: E402
    FlexibleImageDataset,
    compute_class_counts,
    prepare_dataframes,
)
from src.data.samplers import build_train_sampler  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.engine.trainer import build_optimizer, build_scheduler, fit  # noqa: E402
from src.losses.losses import build_loss, get_loss_runtime_metadata  # noqa: E402
from src.models.builder import build_model, configure_trainable_scope  # noqa: E402
from src.utils.checkpoint import initialize_model_from_checkpoint  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.misc import create_run_dir, merge_yaml_configs, save_json, save_yaml  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.staged_training import (  # noqa: E402
    build_stage_runtime,
    infer_run_dir_from_checkpoint,
    normalize_stage_list,
    normalize_stage_name,
    normalize_trainable_scope,
    resolve_optional_path,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train ResNet classifier for HW1.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--aug-config", type=str, default=None)
    parser.add_argument("--inference-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None, help="Reserved for future resume support."
    )
    parser.add_argument(
        "--init-ckpt", type=str, default=None, help="Initialize model from checkpoint."
    )
    parser.add_argument(
        "--init-use-ema",
        action="store_true",
        help="Initialize from EMA weights when --init-ckpt checkpoint contains them.",
    )
    return parser.parse_args()


def _build_dataloader(
    dataset, batch_size: int, num_workers: int, shuffle: bool, cfg: dict, sampler=None
):
    use_pin_memory = bool(cfg.get("pin_memory", True)) and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=bool(cfg.get("persistent_workers", True)) and num_workers > 0,
        drop_last=shuffle or sampler is not None,
    )


def _resolve_staged_training_config(config: dict, args: argparse.Namespace) -> dict:
    train_cfg = config.setdefault("train", {})
    staged_cfg = dict(config.get("staged_training", {}))

    stage_name = normalize_stage_name(staged_cfg.get("stage_name"))
    default_scope = "classifier_only" if stage_name == "classifier_rebalance" else "full_model"
    trainable_scope = normalize_trainable_scope(staged_cfg.get("trainable_scope") or default_scope)
    require_parent_checkpoint = bool(
        staged_cfg.get("require_parent_checkpoint", stage_name == "classifier_rebalance")
    )
    expected_parent_stages = normalize_stage_list(staged_cfg.get("expected_parent_stages"))

    parent_checkpoint = (
        args.init_ckpt or staged_cfg.get("parent_checkpoint") or train_cfg.get("init_checkpoint")
    )
    parent_checkpoint = resolve_optional_path(parent_checkpoint)
    parent_use_ema = bool(
        args.init_use_ema
        or staged_cfg.get("parent_use_ema", False)
        or train_cfg.get("init_use_ema", False)
    )

    if stage_name == "classifier_rebalance" and trainable_scope != "classifier_only":
        raise ValueError(
            "classifier_rebalance stage must use staged_training.trainable_scope=classifier_only."
        )
    if trainable_scope == "classifier_only":
        require_parent_checkpoint = True
    if require_parent_checkpoint and parent_checkpoint is None:
        raise ValueError(
            f"Staged training stage '{stage_name}' requires a valid parent checkpoint. "
            "Set staged_training.parent_checkpoint or pass --init-ckpt."
        )
    if parent_checkpoint is not None and not Path(parent_checkpoint).exists():
        raise FileNotFoundError(f"Parent checkpoint not found: {parent_checkpoint}")

    lineage_cfg = staged_cfg.get("lineage", {})
    if not isinstance(lineage_cfg, dict):
        raise ValueError("staged_training.lineage must be a mapping when provided.")

    resolved = {
        "stage_name": stage_name,
        "trainable_scope": trainable_scope,
        "require_parent_checkpoint": require_parent_checkpoint,
        "parent_checkpoint": parent_checkpoint,
        "parent_use_ema": parent_use_ema if parent_checkpoint is not None else False,
        "parent_run_dir": resolve_optional_path(staged_cfg.get("parent_run_dir"))
        or infer_run_dir_from_checkpoint(parent_checkpoint),
        "expected_parent_stages": expected_parent_stages,
        "lineage": {
            "base_checkpoint": resolve_optional_path(lineage_cfg.get("base_checkpoint")),
            "base_run_dir": resolve_optional_path(lineage_cfg.get("base_run_dir")),
            "rebalance_checkpoint": resolve_optional_path(lineage_cfg.get("rebalance_checkpoint")),
            "rebalance_run_dir": resolve_optional_path(lineage_cfg.get("rebalance_run_dir")),
        },
    }

    train_cfg["init_checkpoint"] = parent_checkpoint
    train_cfg["init_use_ema"] = resolved["parent_use_ema"]
    config["staged_training"] = resolved

    if parent_checkpoint:
        config["model"]["pretrained"] = False
        config["model"]["custom_pretrained_init"] = None

    return resolved


def _validate_parent_checkpoint_context(
    config: dict, staged_cfg: dict, parent_checkpoint: dict
) -> None:
    if not staged_cfg.get("parent_checkpoint"):
        return

    parent_config = parent_checkpoint.get("config", {})
    if not isinstance(parent_config, dict):
        parent_config = {}

    parent_model_name = parent_config.get("model", {}).get("name")
    current_model_name = config.get("model", {}).get("name")
    if parent_model_name and current_model_name and parent_model_name != current_model_name:
        raise ValueError(
            "Parent checkpoint model does not match staged recipe: "
            f"expected '{current_model_name}', got '{parent_model_name}'."
        )

    expected_parent_stages = normalize_stage_list(staged_cfg.get("expected_parent_stages"))
    if expected_parent_stages:
        parent_staged_cfg = parent_config.get("staged_training", {})
        if not isinstance(parent_staged_cfg, dict):
            parent_staged_cfg = {}
        parent_stage_name = normalize_stage_name(parent_staged_cfg.get("stage_name"))
        if parent_stage_name not in expected_parent_stages:
            raise ValueError(
                "Parent checkpoint stage does not match staged recipe: "
                f"expected one of {expected_parent_stages}, got '{parent_stage_name}'."
            )


def main() -> None:
    """Build the training stack, fit the model, and persist run artifacts."""
    args = parse_args()
    extra_cfgs = [
        p
        for p in [args.model_config, args.train_config, args.aug_config, args.inference_config]
        if p
    ]
    config = merge_yaml_configs(args.config, extra_cfgs)

    if args.output_dir is not None:
        config["project"]["output_root"] = args.output_dir

    staged_cfg = _resolve_staged_training_config(config, args)
    init_ckpt = staged_cfg.get("parent_checkpoint")
    init_use_ema = bool(staged_cfg.get("parent_use_ema", False))

    set_seed(
        seed=int(config["project"].get("seed", 42)),
        deterministic=bool(config["project"].get("deterministic", False)),
    )

    output_root = Path(config["project"]["output_root"])
    exp_name = config["project"].get("experiment_name", "exp")
    run_dir = create_run_dir(output_root, exp_name)
    logger = setup_logger("train", run_dir / "train.log")

    save_yaml(config, run_dir / "config.yaml")
    logger.info("Run directory: %s", run_dir)

    bundle = prepare_dataframes(config)
    num_classes = len(bundle.label_to_idx)
    if int(config["model"].get("num_classes", num_classes)) != num_classes:
        logger.warning(
            "Overriding model.num_classes from %s to detected %d.",
            config["model"].get("num_classes"),
            num_classes,
        )
    config["model"]["num_classes"] = num_classes

    train_tfms, eval_tfms = build_transforms(config)
    train_ds = FlexibleImageDataset(bundle.train_df, transform=train_tfms, is_test=False)
    val_ds = FlexibleImageDataset(bundle.val_df, transform=eval_tfms, is_test=False)
    train_sampler, sampler_meta = build_train_sampler(
        bundle.train_df,
        sampler_cfg=config.get("sampler", {}),
        num_classes=num_classes,
    )

    dl_cfg = config["dataloader"]
    train_loader = _build_dataloader(
        train_ds,
        batch_size=int(dl_cfg["batch_size"]),
        num_workers=int(dl_cfg["num_workers"]),
        shuffle=train_sampler is None,
        cfg=dl_cfg,
        sampler=train_sampler,
    )
    val_loader = _build_dataloader(
        val_ds,
        batch_size=int(dl_cfg.get("val_batch_size", dl_cfg["batch_size"])),
        num_workers=int(dl_cfg["num_workers"]),
        shuffle=False,
        cfg=dl_cfg,
    )

    model, model_meta = build_model(config["model"], num_classes=num_classes)
    loaded_parent_checkpoint = None
    if init_ckpt:
        try:
            loaded_parent_checkpoint = initialize_model_from_checkpoint(
                model=model,
                checkpoint_path=init_ckpt,
                map_location="cpu",
                strict=True,
                use_ema=init_use_ema,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Parent checkpoint is incompatible with the requested staged recipe: {init_ckpt}"
            ) from exc
        _validate_parent_checkpoint_context(
            config=config,
            staged_cfg=staged_cfg,
            parent_checkpoint=loaded_parent_checkpoint,
        )

    trainable_scope_meta = configure_trainable_scope(model, staged_cfg["trainable_scope"])
    model_meta["trainable_params"] = trainable_scope_meta["trainable_params"]
    model_meta["frozen_params"] = trainable_scope_meta["frozen_params"]
    model_meta["trainable_scope"] = trainable_scope_meta["trainable_scope"]
    model_meta["trainable_parameter_names"] = trainable_scope_meta["trainable_parameter_names"]

    config["staged_training_runtime"] = build_stage_runtime(
        staged_cfg=staged_cfg,
        parent_checkpoint=loaded_parent_checkpoint,
        trainable_param_names=trainable_scope_meta["trainable_parameter_names"],
        trainable_param_count=trainable_scope_meta["trainable_params"],
        frozen_param_count=trainable_scope_meta["frozen_params"],
    )
    config["staged_training_runtime"]["experiment_name"] = config["project"].get("experiment_name")

    logger.info(
        "Model=%s | params=%.2fM (limit %.2fM) | trainable_scope=%s | trainable=%.2fM",
        config["model"]["name"],
        model_meta["total_params_million"],
        model_meta["param_limit_million"],
        trainable_scope_meta["trainable_scope"],
        trainable_scope_meta["trainable_params"] / 1_000_000,
    )
    build_info = model_meta.get("build_info", {})
    if build_info:
        logger.info("Model build info: %s", build_info)
    logger.info("Staged training runtime: %s", config["staged_training_runtime"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if init_ckpt:
        logger.info("Initialized model from checkpoint: %s | use_ema=%s", init_ckpt, init_use_ema)

    logger.info("Device: %s", device)
    logger.info(
        "Torch=%s | torch.version.cuda=%s | cuda_available=%s | device_count=%d",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.info(
        "Data: train=%d val=%d test=%d classes=%d type=%s",
        len(bundle.train_df),
        len(bundle.val_df),
        len(bundle.test_df),
        num_classes,
        bundle.dataset_type,
    )
    logger.info("Train sampler: %s", sampler_meta)

    class_weights = config["loss"].get("class_weights")
    class_weights_tensor = (
        torch.tensor(class_weights, dtype=torch.float, device=device)
        if class_weights is not None
        else None
    )
    class_counts = compute_class_counts(bundle.train_df, num_classes=num_classes)
    class_counts_tensor = torch.tensor(class_counts, dtype=torch.float, device=device)
    criterion = build_loss(
        config["loss"],
        num_classes=num_classes,
        class_weights=class_weights_tensor,
        class_counts=class_counts_tensor,
        total_epochs=int(config["train"]["epochs"]),
        current_epoch=0,
    ).to(device)
    config["loss_runtime"] = get_loss_runtime_metadata(config["loss"], criterion)
    save_yaml(config, run_dir / "config.yaml")
    logger.info("Loss runtime: %s", config.get("loss_runtime", {}))
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(
        optimizer,
        config["scheduler"],
        total_epochs=int(config["train"]["epochs"]),
    )

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        run_dir=run_dir,
        logger=logger,
    )

    save_json(bundle.idx_to_label, run_dir / "idx_to_label.json")
    save_json({k: int(v) for k, v in bundle.label_to_idx.items()}, run_dir / "label_to_idx.json")
    with (run_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    logger.info(
        "Training finished. Best acc=%.4f at epoch %s",
        summary["best_acc1"],
        summary["best_epoch"],
    )
    logger.info(
        "Model selection: source=%s metric=%s mode=%s value=%.4f",
        summary.get("best_metric_source"),
        summary.get("best_metric_name"),
        summary.get("best_metric_mode"),
        summary.get("best_metric_value", 0.0),
    )
    logger.info("Best checkpoint: %s", summary["best_checkpoint"])


if __name__ == "__main__":
    main()
