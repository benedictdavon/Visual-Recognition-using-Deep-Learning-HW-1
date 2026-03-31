"""Inspect dataset structure and class distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import inspect_dataset_layout, prepare_dataframes  # noqa: E402
from src.utils.misc import merge_yaml_configs, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect homework dataset structure.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root directory.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dataset_inspection.json",
        help="Where to save inspection summary.",
    )
    return parser.parse_args()


def main() -> None:
    """Inspect the configured dataset and export a summary report."""
    args = parse_args()
    if args.config:
        cfg = merge_yaml_configs(args.config)
        if args.data_dir is not None:
            cfg["dataset"]["data_dir"] = args.data_dir
    elif args.data_dir is not None:
        cfg = {
            "dataset": {"data_dir": args.data_dir, "dataset_type": "auto"},
            "project": {"seed": 42},
        }
    else:
        raise ValueError("Provide either --data-dir or --config.")

    data_dir = cfg["dataset"]["data_dir"]
    layout = inspect_dataset_layout(data_dir)
    report = {"layout": layout}

    print("=== Dataset Layout ===")
    print(json.dumps(layout, indent=2))

    try:
        bundle = prepare_dataframes(cfg)
        class_dist = (
            bundle.train_df["label_idx"].value_counts().sort_index().to_dict()
            if "label_idx" in bundle.train_df.columns
            else {}
        )
        report["prepared"] = {
            "dataset_type": bundle.dataset_type,
            "num_train": int(len(bundle.train_df)),
            "num_val": int(len(bundle.val_df)),
            "num_test": int(len(bundle.test_df)),
            "num_classes": int(len(bundle.label_to_idx)),
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        }
        print("\n=== Parsed Dataset Summary ===")
        print(pd.Series(report["prepared"]).to_string())
    except (FileNotFoundError, KeyError, OSError, ValueError, pd.errors.EmptyDataError) as exc:
        report["prepare_error"] = str(exc)
        print(f"\nFailed to parse dataset with current config: {exc}")

    save_json(report, args.output)
    print(f"\nSaved inspection report to: {args.output}")


if __name__ == "__main__":
    main()
