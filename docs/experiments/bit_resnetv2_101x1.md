# BiT ResNetV2-101x1 Experiment Path

This path adds a stronger residual-family transfer prior without leaving the homework's legal model family.

Chosen backbone:

- `resnetv2_101x1_bit.goog_in21k_ft_in1k`

Why this path was added:

- still clearly ResNet-family
- under the `100M` parameter cap
- stronger pretraining than the earlier torchvision-style residual anchors
- intended as a serious single-model upside bet, not just a diversity probe

## Support Added

Runtime support now exists for this backbone through `timm`.

Relevant files:

- `src/models/resnet_variants.py`
- `requirements.txt`
- `configs/model/resnetv2_101x1_bit_goog_in21k_ft_in1k.yaml`
- `configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml`

Important limitation:

- the `timm`-backed BiT path should not be combined with the repo's torchvision-only backbone mutations such as `resnetd`, `attention`, `se_mode`, `drop_path_rate`, or `custom_pretrained_init`

## Commands

Strong `target099` run:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml \
  --output-dir outputs/bit_resnetv2_101x1
```

Validate the EMA branch:

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml \
  --ckpt outputs/bit_resnetv2_101x1/20260331_120013_target099_bit_resnetv2_101x1_strong/checkpoints/best_ema.ckpt \
  --use-ema \
  --output-dir outputs/validate_bit_resnetv2_101x1/20260331_120013_best_ema
```

Infer from the EMA branch:

```bash
python scripts/infer.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml \
  --ckpt outputs/bit_resnetv2_101x1/20260331_120013_target099_bit_resnetv2_101x1_strong/checkpoints/best_ema.ckpt \
  --use-ema \
  --output-dir outputs/infer_bit_resnetv2_101x1/20260331_120013_best_ema_tta \
  --tta
```

## Actual Results

### Weak baseline recipe

The first BiT run used a weak baseline recipe and is not the right reference:

- run:
  - `outputs/bit_resnetv2_101x1/20260331_000827_baseline_resnet50`
- best result:
  - `val_acc = 85.67`

Interpretation:

- this was a recipe mismatch
- it should not be used as the verdict on the BiT backbone

### Strong `target099` recipe

Main run:

- `outputs/bit_resnetv2_101x1/20260331_120013_target099_bit_resnetv2_101x1_strong`

Training progressed cleanly through epoch 24, then crashed while writing `epoch_025.ckpt`.

Crash note:

- the crash happened during checkpoint serialization
- earlier checkpoints were preserved
- the partial `epoch_025.ckpt` should be ignored

Best preserved branch to use:

- `checkpoints/best_ema.ckpt`

Validated result from that branch:

- `val_acc = 92.33`
- `val_macro_recall = 92.33`
- `val_nll = 0.3806`
- `val_ece = 0.0791`
- `val_acc5 = 98.67`

Validation artifact:

- `outputs/validate_bit_resnetv2_101x1/20260331_120013_best_ema/validate_metrics.json`

## Comparison To Existing Anchors

Earlier Stage 0 FixRes anchor:

- selected checkpoint:
  - `val_acc = 91.00`
  - `val_nll = 0.3986`
- best EMA accuracy checkpoint:
  - `val_acc = 92.00`
  - `val_nll = 0.4842`

CRT FixRes-from-rebalance anchor:

- selected checkpoint:
  - `val_acc = 90.33`
  - `val_nll = 0.3587`

BiT interpretation:

- higher validation accuracy than the earlier anchors
- stronger overall balance than the Stage 0 FixRes EMA branch
- not as calibration-lean as the CRT checkpoint, but much stronger in top-1 accuracy

## Verdict

This is the current best local single-model checkpoint.

Practical decision:

- do not retrain from scratch just to recover the missing final epochs
- use the preserved `best_ema.ckpt`
- treat the crash as a storage/checkpointing issue, not as evidence against the backbone
