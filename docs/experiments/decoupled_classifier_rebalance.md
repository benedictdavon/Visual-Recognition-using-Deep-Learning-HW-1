# Decoupled Classifier Rebalance

This path implements the cRT-style hypothesis for HW1 without changing the legal backbone family:

- keep the strong `ResNet101 + logit_adjusted_ce` base
- warm-start a classifier-only rebalance stage from that saved base
- optionally run a short FixRes continuation after the rebalance stage

The point is not to claim a bigger model or a different family. The point is to test whether the representation is already good enough and the remaining bias is mostly in the classifier.

## Configs

- Base anchor: `configs/experiments/target099/decoupled_classifier_rebalance/base_resnet101_logit_adjusted.yaml`
- Rebalance continuation: `configs/experiments/target099/decoupled_classifier_rebalance/classifier_rebalance_from_base_template.yaml`
- Optional FixRes continuation: `configs/experiments/target099/decoupled_classifier_rebalance/fixres320_from_rebalance_template.yaml`
- Smoke verification: `configs/experiments/target099/decoupled_classifier_rebalance/classifier_rebalance_smoke.yaml`

## Training Contract

- The rebalance stage is explicit: `staged_training.stage_name: classifier_rebalance`
- The rebalance stage is classifier-only: `staged_training.trainable_scope: classifier_only`
- The rebalance stage is fail-fast: it requires a real parent checkpoint and refuses incompatible parent stages or model families before optimization starts
- Ordinary runs stay full-model by default
- Run artifacts now record:
  - current stage
  - active trainable scope
  - parent checkpoint and run
  - inferred base and rebalance ancestry for later stages

## Recommended Commands

Base anchor:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/decoupled_classifier_rebalance/base_resnet101_logit_adjusted.yaml \
  --output-dir outputs/target099_crt
```

Classifier rebalance:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/decoupled_classifier_rebalance/classifier_rebalance_from_base_template.yaml \
  --output-dir outputs/target099_crt \
  --init-ckpt outputs/target099_crt/<base_run>/checkpoints/best.ckpt \
  --init-use-ema
```

Optional FixRes continuation:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/decoupled_classifier_rebalance/fixres320_from_rebalance_template.yaml \
  --output-dir outputs/target099_crt \
  --init-ckpt outputs/target099_crt/<rebalance_run>/checkpoints/best.ckpt
```

## Stage Gates

- Base stage:
  - keep the existing `val_acc` and `val_nll` discipline from the strong logit-adjusted anchor
- Rebalance stage:
  - select checkpoints by `val_nll`
  - inspect `val_macro_recall`, `val_acc`, `val_nll`, and `val_ece` together
- Optional FixRes continuation:
  - treat it as a third stage in the same evidence bundle, not as proof that rebalance worked on its own

## Report-Safe Framing

- The backbone remains `ResNet101`, which stays inside the stated homework allowance.
- The classifier-only stage modifies optimization scope, not the backbone family.
- Pretrained weights remain allowed and builder-based legality checks still run before optimization.
- Any claim should be framed as a staged recipe comparison:
  - base only
  - base + classifier rebalance
  - base + classifier rebalance + optional FixRes

## Verification Notes

The implementation should be considered healthy only if all of the following remain true:

- rebalance fails immediately when the parent checkpoint is missing
- rebalance fails before optimization when the parent model is incompatible
- rebalance summaries and `run_metadata.json` show `trainable_scope: classifier_only`
- rebalance trainable parameter names are only `classifier.weight` and `classifier.bias`

## Verification Evidence

Base anchor used for staged verification:

- `outputs/target099_stage0/20260325_213235_target099_s0_r1_resnet101_logit_adjusted`

Positive smoke rebalance run:

- `outputs/target099_crt_verification/20260329_223053_target099_crt_classifier_rebalance_smoke`
- observed trainable parameter names:
  - `classifier.weight`
  - `classifier.bias`
- observed lineage fields:
  - parent experiment: `target099_s0_r1_resnet101_logit_adjusted`
  - parent stage: `single_stage`
  - inferred base checkpoint and base run recorded in both `summary.json` and `run_metadata.json`
- observed smoke metrics:
  - `val_acc`: `88.33`
  - `val_macro_recall`: `88.33`
  - `val_nll`: `0.5053`
  - `val_ece`: `0.0752`

Negative verification runs:

- missing parent checkpoint:
  - command failed before run initialization with `FileNotFoundError`
- incompatible parent checkpoint:
  - `outputs/target099_crt_verification_negative_incompatible_v2/20260329_224013_target099_crt_classifier_rebalance_smoke_resnet50_incompatible`
  - command failed before optimization with `ValueError: Parent checkpoint is incompatible with the requested staged recipe`

Bundle recommendation:

- promote the implementation as ready for full experiments
- run the real base + rebalance bundle next:
  - base: `base_resnet101_logit_adjusted.yaml`
  - rebalance: `classifier_rebalance_from_base_template.yaml`
- only run the optional FixRes continuation if the full rebalance stage improves `val_macro_recall` or `val_nll` without a clearly unacceptable `val_acc` drop
