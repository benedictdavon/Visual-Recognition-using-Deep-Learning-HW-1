"""Run multiple training jobs sequentially from a batch YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.misc import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for queued training jobs."""
    parser = argparse.ArgumentParser(description="Run multiple training jobs in sequence.")
    parser.add_argument(
        "--batch-config", type=str, required=True, help="YAML file describing queued jobs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs if one fails.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch scripts/train.py.",
    )
    return parser.parse_args()


def _resolve_config_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _checkpoint_from_run_dir(run_dir: str, checkpoint_name: str) -> str:
    ckpt_path = Path(_resolve_config_path(run_dir)) / "checkpoints" / f"{checkpoint_name}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return str(ckpt_path.resolve())


def _build_train_command(
    python_exe: str,
    default_config: str,
    job_cfg: dict,
) -> list[str]:
    cmd = [python_exe, str((ROOT / "scripts" / "train.py").resolve())]
    cmd.extend(["--config", _resolve_config_path(job_cfg.get("config", default_config))])

    init_ckpt = job_cfg.get("init_ckpt")
    init_from_run_dir = job_cfg.get("init_from_run_dir")
    init_checkpoint_name = job_cfg.get("init_checkpoint_name")
    if init_ckpt and (init_from_run_dir or init_checkpoint_name):
        raise ValueError(
            "Use either 'init_ckpt' or 'init_from_run_dir' + 'init_checkpoint_name', not both."
        )
    if init_from_run_dir and not init_checkpoint_name:
        raise ValueError("'init_checkpoint_name' is required when using 'init_from_run_dir'.")
    if init_checkpoint_name and not init_from_run_dir and not init_ckpt:
        raise ValueError("'init_from_run_dir' is required when using 'init_checkpoint_name'.")
    if init_from_run_dir:
        init_ckpt = _checkpoint_from_run_dir(init_from_run_dir, init_checkpoint_name)

    optional_paths = {
        "--model-config": job_cfg.get("model_config"),
        "--train-config": job_cfg.get("train_config"),
        "--aug-config": job_cfg.get("aug_config"),
        "--inference-config": job_cfg.get("inference_config"),
        "--output-dir": job_cfg.get("output_dir"),
        "--resume": job_cfg.get("resume"),
        "--init-ckpt": init_ckpt,
    }
    for flag, value in optional_paths.items():
        if value:
            cmd.extend([flag, _resolve_config_path(value)])

    if bool(job_cfg.get("init_use_ema", False)):
        cmd.append("--init-use-ema")
    return cmd


def main() -> int:
    """Execute training jobs from a batch YAML manifest."""
    args = parse_args()
    batch_cfg = load_yaml(args.batch_config)
    jobs = batch_cfg.get("jobs", [])
    if not jobs:
        raise ValueError("Batch config must contain a non-empty 'jobs' list.")

    default_config = batch_cfg.get("base_config", "configs/config.yaml")
    continue_on_error = bool(batch_cfg.get("continue_on_error", False)) or args.continue_on_error

    failures = []
    for idx, job_cfg in enumerate(jobs, start=1):
        job_name = job_cfg.get("name", f"job_{idx:02d}")
        cmd = _build_train_command(args.python, default_config, job_cfg)

        print(f"\n=== [{idx}/{len(jobs)}] {job_name} ===")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            failures.append((job_name, result.returncode))
            print(f"Job failed: {job_name} (exit code {result.returncode})")
            if not continue_on_error:
                return result.returncode

    if failures:
        print("\nCompleted with failures:")
        for job_name, returncode in failures:
            print(f"- {job_name}: exit code {returncode}")
        return 1

    print("\nAll queued training jobs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
