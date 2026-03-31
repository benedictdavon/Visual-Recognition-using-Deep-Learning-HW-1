"""Validation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import (  # noqa: E402
    FlexibleImageDataset,
    compute_class_counts,
    prepare_dataframes,
)
from src.data.transforms import build_transforms  # noqa: E402
from src.engine.evaluator import evaluate  # noqa: E402
from src.losses.losses import build_loss, get_loss_runtime_metadata  # noqa: E402
from src.models.builder import build_model  # noqa: E402
from src.utils.branch_provenance import infer_artifact_provenance  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.misc import ensure_dir, merge_yaml_configs, save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--aug-config", type=str, default=None)
    parser.add_argument("--inference-config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=str, default="outputs/validate")
    parser.add_argument("--use-ema", action="store_true", help="Evaluate EMA weights if present.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k predictions to export in analysis CSV.")
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable detailed validation analysis exports.",
    )
    return parser.parse_args()


def _maybe_numeric_label(label_name: str):
    return int(label_name) if str(label_name).isdigit() else str(label_name)


def main() -> None:
    args = parse_args()
    extra_cfgs = [p for p in [args.model_config, args.train_config, args.aug_config, args.inference_config] if p]
    config = merge_yaml_configs(args.config, extra_cfgs)
    set_seed(
        seed=int(config["project"].get("seed", 42)),
        deterministic=bool(config["project"].get("deterministic", False)),
    )

    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("validate", output_dir / "validate.log")

    bundle = prepare_dataframes(config)
    config["model"]["num_classes"] = len(bundle.label_to_idx)
    config["model"]["pretrained"] = False
    config["model"]["custom_pretrained_init"] = None
    _, eval_tfms = build_transforms(config)
    val_ds = FlexibleImageDataset(bundle.val_df, transform=eval_tfms, is_test=False)

    dl_cfg = config["dataloader"]
    val_loader = DataLoader(
        val_ds,
        batch_size=int(dl_cfg.get("val_batch_size", dl_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=bool(dl_cfg.get("persistent_workers", True)) and int(dl_cfg["num_workers"]) > 0,
    )

    model, model_meta = build_model(config["model"], num_classes=len(bundle.label_to_idx))
    build_info = model_meta.get("build_info", {})
    if build_info:
        logger.info("Model build info: %s", build_info)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    ckpt_name = Path(args.ckpt).stem.lower()
    auto_use_ema = (
        (not args.use_ema)
        and ("ema_state_dict" in ckpt)
        and (ckpt_name == "best_ema")
    )
    if auto_use_ema:
        logger.warning(
            "Checkpoint name is best_ema.ckpt but --use-ema was not set. "
            "Auto-loading EMA weights."
        )

    use_ema = bool(args.use_ema or auto_use_ema)
    if use_ema:
        if "ema_state_dict" not in ckpt:
            raise ValueError(
                "--use-ema was requested but checkpoint does not contain ema_state_dict."
            )
        model.load_state_dict(ckpt["ema_state_dict"]["ema_state_dict"], strict=True)
        logger.info("Loaded EMA weights from checkpoint.")
    else:
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded standard model weights from checkpoint.")
    artifact_provenance = infer_artifact_provenance(args.ckpt, use_ema=use_ema)
    logger.info("Artifact provenance: %s", artifact_provenance)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = config["loss"].get("class_weights")
    class_weights_tensor = (
        torch.tensor(class_weights, dtype=torch.float, device=device) if class_weights is not None else None
    )
    class_counts = compute_class_counts(bundle.train_df, num_classes=len(bundle.label_to_idx))
    class_counts_tensor = torch.tensor(class_counts, dtype=torch.float, device=device)
    total_epochs = int(config["train"].get("epochs", config.get("scheduler", {}).get("epochs", 1)))
    criterion = build_loss(
        config["loss"],
        num_classes=len(bundle.label_to_idx),
        class_weights=class_weights_tensor,
        class_counts=class_counts_tensor,
        total_epochs=total_epochs,
        current_epoch=max(total_epochs - 1, 0),
    ).to(device)
    logger.info("Loss runtime: %s", get_loss_runtime_metadata(config["loss"], criterion))

    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        criterion=criterion,
        amp=bool(config["train"].get("amp", True)),
        num_classes=len(bundle.label_to_idx),
        desc="Validate",
        return_predictions=not args.no_analysis,
    )

    serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in metrics.items() if k not in {"targets", "preds", "probs"}
    }
    serializable["val_acc"] = float(metrics["acc1"])
    serializable["val_macro_recall"] = float(metrics.get("macro_recall", 0.0))
    serializable["val_nll"] = float(metrics.get("nll", 0.0))
    serializable["val_ece"] = float(metrics.get("ece", 0.0))
    serializable["checkpoint"] = str(Path(args.ckpt))
    serializable["used_ema_weights"] = bool(use_ema)
    serializable["artifact_provenance"] = artifact_provenance
    if not args.no_analysis and "targets" in metrics:
        targets = metrics["targets"]
        preds = metrics["preds"]
        probs = metrics["probs"]
        topk = max(1, min(int(args.topk), probs.shape[1] if probs.size else 1))

        class_idx = list(range(len(bundle.idx_to_label)))
        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            preds,
            labels=class_idx,
            average=None,
            zero_division=0,
        )
        macro_f1 = float(f1_score(targets, preds, average="macro"))
        weighted_f1 = float(f1_score(targets, preds, average="weighted"))
        serializable["macro_f1"] = macro_f1
        serializable["weighted_f1"] = weighted_f1

        class_acc = metrics.get("per_class_accuracy", {})
        class_rows = []
        for c in class_idx:
            class_name = bundle.idx_to_label[c]
            class_rows.append(
                {
                    "class_idx": c,
                    "class_id": _maybe_numeric_label(class_name),
                    "precision": float(precision[c]),
                    "recall": float(recall[c]),
                    "f1": float(f1[c]),
                    "support": int(support[c]),
                    "class_acc": float(class_acc.get(c, 0.0)),
                }
            )
        class_df = pd.DataFrame(class_rows)
        class_df.to_csv(output_dir / "per_class_report.csv", index=False)
        class_df.sort_values(["class_acc", "support"], ascending=[True, True]).head(20).to_csv(
            output_dir / "hardest_classes.csv",
            index=False,
        )

        # Per-sample predictions and failures.
        topk_idx = np.argsort(-probs, axis=1)[:, :topk]
        topk_prob = np.take_along_axis(probs, topk_idx, axis=1)

        true_label_ids = [
            _maybe_numeric_label(bundle.idx_to_label[int(i)]) for i in targets.tolist()
        ]
        pred_label_ids = [
            _maybe_numeric_label(bundle.idx_to_label[int(i)]) for i in preds.tolist()
        ]

        pred_df = pd.DataFrame(
            {
                "sample_id": bundle.val_df["sample_id"].astype(str).tolist(),
                "path": bundle.val_df["path"].astype(str).tolist(),
                "true_idx": targets.tolist(),
                "true_label": true_label_ids,
                "pred_idx": preds.tolist(),
                "pred_label": pred_label_ids,
                "correct": (targets == preds).tolist(),
                "confidence": probs.max(axis=1).tolist(),
                "top1_prob": probs[np.arange(len(preds)), preds].tolist(),
            }
        )
        pred_df["topk_pred_idx"] = [",".join(map(str, row.tolist())) for row in topk_idx]
        pred_df["topk_pred_label"] = [
            ",".join(str(_maybe_numeric_label(bundle.idx_to_label[int(i)])) for i in row.tolist())
            for row in topk_idx
        ]
        pred_df["topk_prob"] = [",".join(f"{v:.6f}" for v in row.tolist()) for row in topk_prob]
        pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
        pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False).to_csv(
            output_dir / "val_misclassified.csv",
            index=False,
        )
        np.savez_compressed(
            output_dir / "val_probs_with_ids.npz",
            sample_ids=np.array(bundle.val_df["sample_id"].astype(str).tolist()),
            targets=targets,
            preds=preds,
            probs=probs,
            branch=np.array([artifact_provenance["branch"]]),
        )

        logger.info(
            "Analysis exported: per_class_report.csv, hardest_classes.csv, "
            "val_predictions.csv, val_misclassified.csv, val_probs_with_ids.npz",
        )

    serializable["model_metadata"] = model_meta
    save_json(serializable, output_dir / "validate_metrics.json")
    save_json(artifact_provenance, output_dir / "artifact_provenance.json")
    if "confusion_matrix" in metrics:
        np.save(output_dir / "confusion_matrix.npy", metrics["confusion_matrix"])
        conf_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[_maybe_numeric_label(bundle.idx_to_label[i]) for i in range(len(bundle.idx_to_label))],
            columns=[_maybe_numeric_label(bundle.idx_to_label[i]) for i in range(len(bundle.idx_to_label))],
        )
        conf_df.to_csv(output_dir / "confusion_matrix.csv")

    logger.info(
        "Validation acc1=%.4f, macro_recall=%.4f, nll=%.4f, ece=%.4f, acc5=%.4f, loss=%.4f",
        metrics["acc1"],
        metrics.get("macro_recall", 0.0),
        metrics.get("nll", 0.0),
        metrics.get("ece", 0.0),
        metrics.get("acc5", 0.0),
        metrics["loss"],
    )
    logger.info("Saved results to %s", output_dir)


if __name__ == "__main__":
    main()
