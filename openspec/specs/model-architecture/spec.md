# Model Architecture Spec

## Overview

This spec defines what "ResNet-family" means in this repository, how supported architectures are registered, and how architecture legality should be evaluated for homework use.
It is derived from [src/models/builder.py](../../../src/models/builder.py), [src/models/resnet_variants.py](../../../src/models/resnet_variants.py), [src/models/modules.py](../../../src/models/modules.py), [configs/config.yaml](../../../configs/config.yaml), and the current homework policy captured in [../homework-requirements/spec.md](../homework-requirements/spec.md).

## Requirements

### Observed current behavior

- The model config is centered on `model.name`.
- `build_model` always routes through `src/models/builder.py`, which:
  - calls `build_resnet_variant`
  - measures parameter count
  - rejects models whose parameter count is greater than or equal to `model.param_limit_million`
- `build_resnet_variant` currently supports these base names:
  - `resnet50`
  - `resnet101`
  - `resnet152`
  - `resnext50_32x3d`
  - `resnext50_32x4d`
  - `wide_resnet50_2`
  - `resnext101_64x4d`
  - `resnext101_32x8d`
- It also supports registered profile names:
  - `resnet101d_bse_sd`
  - `resnext101_32x8d_d_bse_sd`
- The current code supports these architectural modification axes:
  - ResNet-D stem/downsample changes through `model.resnetd`
  - stage attention through `model.attention: none|se|cbam`
  - bottleneck-level SE through `model.se_mode: bottleneck`
  - stochastic depth through `model.drop_path_rate`
  - classifier-head dropout through `model.dropout`
- Unsupported model names raise `ValueError`.

### Required preserved behavior

- In this repo, "ResNet-family" means either:
  - a standard residual architecture registered in the current builder, or
  - a clearly residual-derived extension whose backbone lineage remains recognizable as ResNet or a ResNet-derived family
- Architecture legality for homework use is tiered:
  - `Tier 1: safest`
    - `resnet50`
    - `resnet101`
    - `resnet152`
  - `Tier 2: allowed with report-visible explanation and citation`
    - `resnext50_32x3d`
    - `resnext50_32x4d`
    - `wide_resnet50_2`
    - `resnext101_64x4d`
    - `resnext101_32x8d`
    - ResNet-D style stem/downsample changes
    - SE / CBAM additions to residual stages
    - bottleneck SE and stochastic depth additions
    - registered profiles built from those ingredients
  - `Tier 3: requires explicit justification before homework use`
    - newly added model names whose residual lineage is not obvious
    - structural changes that stop looking like a clear residual-family extension
    - any architecture that code can build but the report cannot defend cleanly as ResNet-family
- Code support does not automatically make a model Tier 1.
- Custom residual-family variants that do not have a standard torchvision pretrained checkpoint must expose their initialization path explicitly in config or fail fast.
- Future architecture additions must be registered explicitly in `build_resnet_variant`. Silent name aliases or hidden model construction paths are not allowed.
- Homework-facing architecture changes must remain explainable from:
  - the config
  - the model name
  - the report write-up

### Must not break

- `src/models/builder.py` must remain the parameter-budget enforcement boundary.
- Unsupported `model.name` values must continue to fail fast.
- The strict `< 100M` parameter interpretation must remain intact.
- Training, validation, and inference scripts must continue to build models through the same builder path.
- Future architectures must not bypass legality review just because they share some residual blocks with an allowed model.

## Scenarios

### Scenario: Vanilla ResNet stays in Tier 1

- Given `model.name: resnet101`
- When `build_model` is called
- Then the model is Tier 1 on architecture policy grounds
- And it remains valid only if the parameter budget is satisfied

### Scenario: ResNeXt or Wide-ResNet is allowed with explanation

- Given `model.name: resnext50_32x3d`, `model.name: resnext50_32x4d`, `model.name: resnext101_64x4d`, or `model.name: wide_resnet50_2`
- When the model is used for homework-facing work
- Then the model is treated as an allowed ResNet-family extension under Tier 2
- And the report must cite and explain the architectural family and why it remains residual-based

### Scenario: Custom grouped-width variant requires explicit init policy

- Given `model.name: resnext50_32x3d`
- And `model.pretrained: true`
- When no explicit custom initialization mode is configured
- Then model construction must fail
- And the repo must not silently imply standard pretrained support

### Scenario: ResNet-D plus SE plus stochastic depth is allowed with explanation

- Given a configuration that enables `model.resnetd`, bottleneck SE, or non-zero `drop_path_rate`
- When the model is used in homework-facing work
- Then it remains allowable only if the report makes the residual-family lineage and exact code-side changes explicit

### Scenario: Unsupported model name fails fast

- Given `model.name` is not registered in `build_resnet_variant`
- When training or inference builds the model
- Then the command must fail with an explicit error

### Scenario: Parameter-limit violation blocks the architecture

- Given a model that reaches or exceeds `model.param_limit_million`
- When `build_model` is called
- Then model construction must fail before training begins

### Scenario: Code-supported but undocumented modification is not homework-ready

- Given a code-supported Tier 2 architecture
- And the report does not explain or cite the modification
- When homework readiness is evaluated
- Then the run is not homework-ready even though the code executed successfully

### Scenario: New ambiguous backbone addition is not auto-approved

- Given a future contributor adds a new `model.name`
- When that name is only weakly related to ResNet-family lineage
- Then the addition is Tier 3 by default
- And it must not be treated as homework-safe until explicitly justified

## Known ambiguities or gaps

- Legality tier is not stored in config or model metadata today.
- The repo now records custom initialization metadata for supported custom grouped-width variants, but legality tier is still not encoded in config.
- The repo does not programmatically require citations for ResNet-family extensions.
- The builder enforces parameter budget but not report defensibility.

## Non-goals

- This spec does not require architecture search or auto-registration.
- This spec does not define training recipe policy.
- This spec does not guarantee that every Tier 2 model will outperform Tier 1 baselines.
