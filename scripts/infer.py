"""Inference entrypoint for test prediction and submission CSV generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import FlexibleImageDataset, prepare_dataframes  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.engine.inference import predict_probs  # noqa: E402
from src.models.builder import build_model  # noqa: E402
from src.submission.make_submission import (  # noqa: E402
    build_prediction_dataframe,
    save_prediction_csv,
)
from src.utils.branch_provenance import infer_artifact_provenance  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.utils.misc import ensure_dir, merge_yaml_configs  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference and submission export."""
    parser = argparse.ArgumentParser(description="Inference and prediction.csv generation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--aug-config", type=str, default=None)
    parser.add_argument("--inference-config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=str, default="outputs/infer")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available.")
    parser.add_argument("--tta", action="store_true", help="Override config to enable TTA.")
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="Save per-image class probabilities for soft-vote ensemble (.npy + .npz).",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Also save per-image logits (requires an additional forward output buffer).",
    )
    return parser.parse_args()


def main() -> None:
    """Run test-time inference and save prediction artifacts."""
    args = parse_args()
    extra_cfgs = [
        p
        for p in [args.model_config, args.train_config, args.aug_config, args.inference_config]
        if p
    ]
    config = merge_yaml_configs(args.config, extra_cfgs)
    set_seed(
        seed=int(config["project"].get("seed", 42)),
        deterministic=bool(config["project"].get("deterministic", False)),
    )
    output_dir = ensure_dir(args.output_dir)
    logger = setup_logger("infer", output_dir / "infer.log")

    bundle = prepare_dataframes(config)
    config["model"]["num_classes"] = len(bundle.label_to_idx)
    config["model"]["pretrained"] = False
    config["model"]["custom_pretrained_init"] = None
    _, eval_tfms = build_transforms(config)
    test_ds = FlexibleImageDataset(bundle.test_df, transform=eval_tfms, is_test=True)

    dl_cfg = config["dataloader"]
    test_loader = DataLoader(
        test_ds,
        batch_size=int(dl_cfg.get("val_batch_size", dl_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=bool(dl_cfg.get("persistent_workers", True))
        and int(dl_cfg["num_workers"]) > 0,
    )

    model, _ = build_model(config["model"], num_classes=len(bundle.label_to_idx))
    model_meta = getattr(model, "_build_info", None)
    if model_meta:
        logger.info("Model build info: %s", model_meta)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    ckpt_name = Path(args.ckpt).stem.lower()
    auto_use_ema = (not args.use_ema) and ("ema_state_dict" in ckpt) and (ckpt_name == "best_ema")
    if auto_use_ema:
        logger.warning(
            "Checkpoint name is best_ema.ckpt but --use-ema was not set. Auto-loading EMA weights."
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

    tta_cfg = dict(config.get("inference", {}).get("tta", {}))
    if args.tta:
        tta_cfg["enabled"] = True

    need_logits = bool(args.save_logits)
    if need_logits:
        sample_ids, probs, logits = predict_probs(
            model=model,
            loader=test_loader,
            device=device,
            amp=bool(config["train"].get("amp", True)),
            tta_cfg=tta_cfg,
            return_logits=True,
            desc="Infer",
        )
    else:
        sample_ids, probs = predict_probs(
            model=model,
            loader=test_loader,
            device=device,
            amp=bool(config["train"].get("amp", True)),
            tta_cfg=tta_cfg,
            return_logits=False,
            desc="Infer",
        )
    pred_indices = probs.argmax(axis=1).tolist()

    id_source = config["dataset"].get("test_id_from", "sample_id")
    if id_source not in {"sample_id", "filename"}:
        raise ValueError("dataset.test_id_from must be 'sample_id' or 'filename'.")
    if id_source in bundle.test_df.columns:
        output_ids = bundle.test_df[id_source].astype(str).tolist()
    else:
        output_ids = sample_ids

    output_cfg = config.get("inference", {}).get("output", {})
    use_label_name = bool(output_cfg.get("use_label_name", False))
    if not use_label_name and all(str(v).isdigit() for v in bundle.label_to_idx.keys()):
        logger.warning(
            "output.use_label_name is False while class names are numeric. "
            "This writes internal class indices, which may hurt competition score."
        )
    submission_df = build_prediction_dataframe(
        sample_ids=output_ids,
        pred_indices=pred_indices,
        id_column=output_cfg.get("id_column", "id"),
        target_column=output_cfg.get("target_column", "label"),
        idx_to_label=bundle.idx_to_label,
        use_label_name=use_label_name,
    )
    pred_csv_path = save_prediction_csv(
        submission_df, output_dir=output_dir, filename="prediction.csv"
    )

    with (output_dir / "inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.ckpt,
                "num_samples": len(sample_ids),
                "prediction_csv": str(pred_csv_path),
                "tta": tta_cfg,
                "used_ema_weights": use_ema,
                "artifact_provenance": artifact_provenance,
                "model_build_info": model_meta,
            },
            f,
            indent=2,
        )
    with (output_dir / "artifact_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(artifact_provenance, f, indent=2)

    if args.save_probs:
        np.save(output_dir / "test_probs.npy", probs)
        npz_path = output_dir / "test_probs_with_ids.npz"
        if need_logits:
            np.save(output_dir / "test_logits.npy", logits)
            np.savez_compressed(
                npz_path,
                sample_ids=np.array(output_ids),
                probs=probs,
                logits=logits,
                branch=np.array([artifact_provenance["branch"]]),
            )
            logger.info("Saved logits to %s", output_dir / "test_logits.npy")
        else:
            np.savez_compressed(
                npz_path,
                sample_ids=np.array(output_ids),
                probs=probs,
                branch=np.array([artifact_provenance["branch"]]),
            )

        with (output_dir / "idx_to_label.json").open("w", encoding="utf-8") as f:
            json.dump({int(k): v for k, v in bundle.idx_to_label.items()}, f, indent=2)
        logger.info("Saved probabilities to %s and %s", output_dir / "test_probs.npy", npz_path)
    elif args.save_logits:
        # Save logits-only artifact if explicitly requested without save-probs.
        np.save(output_dir / "test_logits.npy", logits)
        np.savez_compressed(
            output_dir / "test_probs_with_ids.npz",
            sample_ids=np.array(output_ids),
            probs=probs,
            logits=logits,
            branch=np.array([artifact_provenance["branch"]]),
        )
        with (output_dir / "idx_to_label.json").open("w", encoding="utf-8") as f:
            json.dump({int(k): v for k, v in bundle.idx_to_label.items()}, f, indent=2)
        logger.info("Saved logits to %s", output_dir / "test_logits.npy")

    logger.info("Saved prediction CSV: %s", pred_csv_path)


if __name__ == "__main__":
    main()
