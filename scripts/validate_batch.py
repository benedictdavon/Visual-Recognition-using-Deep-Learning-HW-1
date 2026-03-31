"""Run multiple validation jobs sequentially from a batch YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.misc import ensure_dir, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple validation jobs in sequence.")
    parser.add_argument("--batch-config", type=str, required=True, help="YAML file describing queued jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs if one fails.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch scripts/validate.py.",
    )
    return parser.parse_args()


def _resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _checkpoint_from_run_dir(run_dir: str, checkpoint_name: str) -> str:
    ckpt_path = Path(_resolve_path(run_dir)) / "checkpoints" / f"{checkpoint_name}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return str(ckpt_path.resolve())


def _default_output_dir(job_name: str) -> str:
    return str((ensure_dir(ROOT / "outputs" / "validate_batch") / job_name).resolve())


def _build_command(python_exe: str, default_config: str, job_cfg: dict) -> list[str]:
    cmd = [python_exe, str((ROOT / "scripts" / "validate.py").resolve())]
    cmd.extend(["--config", _resolve_path(job_cfg.get("config", default_config))])

    train_config = job_cfg.get("train_config")
    if train_config:
        cmd.extend(["--train-config", _resolve_path(train_config)])

    optional_paths = {
        "--model-config": job_cfg.get("model_config"),
        "--aug-config": job_cfg.get("aug_config"),
        "--inference-config": job_cfg.get("inference_config"),
    }
    for flag, value in optional_paths.items():
        if value:
            cmd.extend([flag, _resolve_path(value)])

    ckpt = job_cfg.get("ckpt")
    run_dir = job_cfg.get("run_dir")
    checkpoint_name = job_cfg.get("checkpoint_name")
    if ckpt and (run_dir or checkpoint_name):
        raise ValueError("Use either 'ckpt' or 'run_dir' + 'checkpoint_name', not both.")
    if run_dir and not checkpoint_name:
        raise ValueError("'checkpoint_name' is required when 'run_dir' is set.")
    if checkpoint_name and not run_dir and not ckpt:
        raise ValueError("'run_dir' is required when 'checkpoint_name' is set.")
    if ckpt:
        cmd.extend(["--ckpt", _resolve_path(ckpt)])
    elif run_dir and checkpoint_name:
        cmd.extend(["--ckpt", _checkpoint_from_run_dir(run_dir, checkpoint_name)])
    else:
        raise ValueError("Each validation job needs either 'ckpt' or both 'run_dir' and 'checkpoint_name'.")

    output_dir = job_cfg.get("output_dir", _default_output_dir(job_cfg.get("name", "job")))
    cmd.extend(["--output-dir", _resolve_path(output_dir)])

    if bool(job_cfg.get("use_ema", False)):
        cmd.append("--use-ema")
    if bool(job_cfg.get("no_analysis", False)):
        cmd.append("--no-analysis")
    if "topk" in job_cfg:
        cmd.extend(["--topk", str(job_cfg["topk"])])
    return cmd


def main() -> int:
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
        cmd = _build_command(args.python, default_config, job_cfg)

        print(f"\n=== [{idx}/{len(jobs)}] {job_name} ===")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=ROOT)
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

    print("\nAll queued validation jobs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
