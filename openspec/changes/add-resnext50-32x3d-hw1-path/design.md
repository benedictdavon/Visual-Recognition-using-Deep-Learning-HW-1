# Design: ResNeXt-50 32x3d HW1 Path

## Overview

This design adds a custom `resnext50_32x3d` backbone to the existing model builder and defines a conservative experiment path around it.

The design is shaped by three facts from the current repo:

- the best single-model path is still `resnet101` plus a short FixRes stage
- the heavy `resnext101_64x4d` trial failed under a copied recipe with a much smaller batch size
- the hidden-test plateau appears to be driven by correlation and noisy ranking rather than obvious under-capacity

Because of that, `resnext50_32x3d` is treated first as a low-cost grouped-cardinality probe and potential diversity model, not as an automatically stronger single-model replacement.

## Current Repo Facts That Matter

- The builder path is `scripts/* -> src/models/builder.py -> src/models/resnet_variants.py`.
- The parameter limit is enforced centrally in `src/models/builder.py`.
- Current supported grouped residual backbones are `resnext101_64x4d` and `resnext101_32x8d`.
- The repo already supports residual-family modifications such as ResNet-D, SE, CBAM, bottleneck SE, and stochastic depth.
- Stronger current experimentation is centered on:
  - `val_acc`
  - `val_macro_recall`
  - `val_nll`
  - `val_ece`
  - dense checkpoint retention
  - explicit run metadata

## Architecture Definition

The proposed model is:

- name: `resnext50_32x3d`
- depth profile: ResNet-50 bottleneck stage counts `[3, 4, 6, 3]`
- groups: `32`
- width per group: `3`
- approximate parameter count in this repo's 100-class classifier setup: `17.88M`

Implementation should construct it through the same torchvision `ResNet` / `Bottleneck` family used for existing residual backbones so the residual lineage remains clear.

## Legality And Reportability

This model should be treated as a Tier 2 residual-family extension under the current repo policy:

- it remains a grouped-bottleneck residual network in the ResNeXt family
- it must be explained in the report as a narrower ResNeXt-style grouped-convolution variant
- the report must cite ResNet and ResNeXt
- the report must state that the model is custom-configured inside the ResNeXt family rather than implying it is a standard torchvision checkpoint

## Initialization Strategy

This is the highest-risk part of the design because `32x3d` does not have a standard torchvision pretrained checkpoint.

The preferred implementation path is:

1. Build the custom `resnext50_32x3d` module.
2. Add an explicit initialization mode for this model only.
3. Support a conservative warm-start from `resnext50_32x4d` ImageNet weights by shape-aware channel slicing where dimensions shrink.
4. Log the initialization source clearly in the run output and metadata.

If that warm-start proves too brittle in implementation or produces unstable training, the fallback is:

- keep `resnext50_32x3d` support
- allow `pretrained: false`
- mark it as a lower-confidence path
- prioritize `resnext50_32x4d` as the next cleaner cardinality baseline

The repo must not silently pretend that `32x3d` uses a standard pretrained checkpoint when it does not.

## Training Recipe

The architecture should reuse the current strongest repo training stack as much as possible:

- pretrained initialization when available through the explicit warm-start path
- strong augmentation stack
- AdamW
- cosine decay
- EMA
- AMP
- the newer validation metrics and run metadata

The recipe should not simply clone the old `resnext101_64x4d` config.

Required experiment stages:

1. Smoke
   - model build
   - parameter count
   - one short run to confirm forward/backward, checkpointing, and inference compatibility
2. Short base run at `256`
   - enough epochs to see whether optimization is healthy
   - no FixRes yet
3. Full base run at `256`
   - only if the short run passes the gate
4. Short FixRes refresh at `320`
   - low LR
   - mixup/cutmix disabled
   - only if the full base run is already competitive
5. Diversity and ensemble evaluation
   - compare against current ResNet101 anchor using saved validation/test probabilities

## Continuation Gates

### Gate 1: Post-smoke

- build succeeds
- param count is under `100M`
- train, validate, and infer paths all run

### Gate 2: Post-short-run

Continue only if at least one of the following is true:

- selected `val_acc` is within about `1.5` to `2.0` points of the current ResNet101 base anchor
- selected `val_nll` is close enough to suggest the model is still viable after full training
- disagreement and rescue behavior suggest meaningful ensemble diversity potential

Stop early if:

- the model is clearly worse than the current anchor with no diversity upside
- training is unstable
- overfit appears faster than the already fragile FixRes path

### Gate 3: Post-full-run

Continue to FixRes only if:

- the base run is single-model competitive
- or the base run is slightly weaker but materially more diverse than the current model pool

## Inference And Ensemble Plan

If the model passes the full-run gate:

- run standard validation and test inference through existing scripts
- save probability artifacts
- compare against the best ResNet101 FixRes anchor with the existing diversity and ensemble search tooling

The likely highest-ROI use case is:

- `resnext50_32x3d` as an ensemble partner for the current ResNet101 path

The least defensible use case is:

- replacing the current best ResNet101 baseline without evidence

## Scientific Support

- ResNet: residual learning makes deep optimization practical.
- ResNeXt: cardinality is a meaningful scaling axis distinct from depth and width.
- Bag of Tricks: recipe details matter enough to swamp naive architecture comparisons.
- Revisiting ResNets: training and scaling choices can matter more than architectural novelty.
- FixRes: short resolution-refresh fine-tuning is useful only after the base model is already strong.

## Risks

- `32x3d` may simply be too small to beat the current `resnet101` path as a single model.
- Custom warm-start logic may introduce implementation complexity or brittle assumptions.
- Tiny validation means real gains may be hard to rank.
- A lower-parameter model may only pay off as an ensemble-diversity branch, not a headline single model.

## Mitigations

- Keep the implementation narrow and builder-centered.
- Fail fast if initialization mode is ambiguous.
- Use the newer metric stack and diversity tooling rather than only top-1 accuracy.
- Define a hard fallback to `resnext50_32x4d` if `32x3d` is not competitive enough.
