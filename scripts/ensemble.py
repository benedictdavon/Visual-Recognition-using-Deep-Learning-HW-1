"""Simple ensemble utility for multiple prediction CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission.make_submission import save_prediction_csv  # noqa: E402
from src.utils.misc import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hard or soft ensembling."""
    parser = argparse.ArgumentParser(description="Ensemble multiple prediction CSV files.")
    parser.add_argument("--mode", choices=["hard", "soft"], default="hard", help="Ensemble mode.")
    parser.add_argument(
        "--preds", nargs="*", default=None, help="Prediction CSV paths for hard voting."
    )
    parser.add_argument(
        "--prob-files",
        nargs="*",
        default=None,
        help="Probability artifacts (.npz from infer --save-probs) for soft voting.",
    )
    parser.add_argument(
        "--prob-key",
        choices=["probs", "logits"],
        default="probs",
        help="Which array to fuse in soft mode.",
    )
    parser.add_argument(
        "--weights", nargs="*", type=float, default=None, help="Optional per-file weights."
    )
    parser.add_argument(
        "--idx-to-label", type=str, default=None, help="Path to idx_to_label.json for soft mode."
    )
    parser.add_argument("--id-column", type=str, default="image_name")
    parser.add_argument("--target-column", type=str, default="pred_label")
    parser.add_argument("--output-dir", type=str, default="outputs/ensemble")
    parser.add_argument(
        "--use-label-name",
        dest="use_label_name",
        action="store_true",
        help="Map predicted class index back to label names in soft mode (default: enabled).",
    )
    parser.add_argument(
        "--no-use-label-name",
        dest="use_label_name",
        action="store_false",
        help="Use raw class indices in soft mode.",
    )
    parser.set_defaults(use_label_name=True)
    return parser.parse_args()


def weighted_vote(row_values, weights):
    """Return the label with the largest accumulated vote weight."""
    score = {}
    for value, w in zip(row_values, weights):
        score[value] = score.get(value, 0.0) + w
    return max(score, key=score.get)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_idx_to_label(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


def _resolve_idx_to_label(
    prob_paths: list[Path], override_path: str | None
) -> dict[int, str] | None:
    if override_path:
        p = Path(override_path)
        if not p.exists():
            raise FileNotFoundError(f"idx_to_label not found: {p}")
        return _load_idx_to_label(p)

    candidate = prob_paths[0].parent / "idx_to_label.json"
    if candidate.exists():
        return _load_idx_to_label(candidate)
    return None


def _load_prob_artifact(path: Path, prob_key: str) -> tuple[list[str], np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    if "sample_ids" not in arr:
        raise ValueError(f"{path} is missing 'sample_ids' in npz artifact.")
    sample_ids = arr["sample_ids"].astype(str).tolist()

    if prob_key == "probs":
        if "probs" in arr:
            values = arr["probs"].astype(np.float64)
        elif "logits" in arr:
            values = _softmax_numpy(arr["logits"].astype(np.float64))
        else:
            raise ValueError(f"{path} is missing both 'probs' and 'logits'.")
    else:
        if "logits" not in arr:
            raise ValueError(f"{path} is missing 'logits' while --prob-key=logits was requested.")
        values = arr["logits"].astype(np.float64)
    return sample_ids, values


def main() -> None:
    """Fuse predictions and export a competition-style submission file."""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.mode == "hard":
        if not args.preds:
            raise ValueError("--preds is required when --mode=hard")
        csv_paths = [Path(p) for p in args.preds]

        frames = []
        for path in csv_paths:
            df = pd.read_csv(path)
            if args.id_column not in df.columns or args.target_column not in df.columns:
                raise ValueError(
                    f"{path} missing required columns '{args.id_column}' and '{args.target_column}'."
                )
            frames.append(df[[args.id_column, args.target_column]].copy())

        base_ids = frames[0][args.id_column].astype(str).tolist()
        for idx, frame in enumerate(frames[1:], start=1):
            ids = frame[args.id_column].astype(str).tolist()
            if ids != base_ids:
                raise ValueError(
                    f"ID order mismatch in prediction file index {idx}: {csv_paths[idx]}"
                )

        weights = args.weights if args.weights else [1.0] * len(frames)
        if len(weights) != len(frames):
            raise ValueError("weights length must match number of prediction files.")

        target_matrix = np.column_stack([f[args.target_column].astype(str).values for f in frames])
        ensemble_targets = []
        if any(w != 1.0 for w in weights):
            for row in target_matrix:
                ensemble_targets.append(weighted_vote(row, weights))
        else:
            for row in target_matrix:
                ensemble_targets.append(Counter(row).most_common(1)[0][0])

        out_df = pd.DataFrame(
            {
                args.id_column: base_ids,
                args.target_column: ensemble_targets,
            }
        )
        output_path = save_prediction_csv(out_df, output_dir=output_dir, filename="prediction.csv")
        print(f"Hard-vote ensemble of {len(frames)} files -> {output_path}")
        return

    if not args.prob_files:
        raise ValueError("--prob-files is required when --mode=soft")
    prob_paths = [Path(p) for p in args.prob_files]
    weights = args.weights if args.weights else [1.0] * len(prob_paths)
    if len(weights) != len(prob_paths):
        raise ValueError("weights length must match number of probability files.")

    base_ids, first_values = _load_prob_artifact(prob_paths[0], args.prob_key)
    fused = first_values * float(weights[0])
    for idx, path in enumerate(prob_paths[1:], start=1):
        ids, values = _load_prob_artifact(path, args.prob_key)
        if ids != base_ids:
            raise ValueError(f"ID order mismatch in probability file index {idx}: {path}")
        if values.shape != first_values.shape:
            raise ValueError(f"Shape mismatch in probability file index {idx}: {path}")
        fused += values * float(weights[idx])

    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    fused = fused / weight_sum
    if args.prob_key == "logits":
        fused = _softmax_numpy(fused)

    pred_indices = np.argmax(fused, axis=1).tolist()
    idx_to_label = _resolve_idx_to_label(prob_paths, args.idx_to_label)
    if args.use_label_name:
        if idx_to_label is None:
            raise ValueError(
                "Soft mode with label-name output requires idx_to_label mapping. "
                "Pass --idx-to-label or place idx_to_label.json beside the first prob file."
            )
        mapped = [idx_to_label[int(i)] for i in pred_indices]
        if all(str(v).isdigit() for v in mapped):
            targets = [int(v) for v in mapped]
        else:
            targets = mapped
    else:
        targets = [int(i) for i in pred_indices]

    out_df = pd.DataFrame(
        {
            args.id_column: base_ids,
            args.target_column: targets,
        }
    )
    output_path = save_prediction_csv(out_df, output_dir=output_dir, filename="prediction.csv")
    print(f"Soft-vote ensemble of {len(prob_paths)} files -> {output_path}")


if __name__ == "__main__":
    main()
