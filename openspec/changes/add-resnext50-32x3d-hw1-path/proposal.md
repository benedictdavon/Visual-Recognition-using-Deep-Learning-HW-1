# Proposal: Add ResNeXt-50 32x3d HW1 Path

## Summary

Add a legal ResNet-family extension path for `resnext50_32x3d` and the minimum experiment scaffolding needed to judge it against the current HW1 anchor under the existing repo constraints.

The goal is not to promise a public leaderboard jump to `0.98`. The goal is to implement a disciplined path that can answer whether a narrower grouped-bottleneck ResNeXt variant can improve generalization or ensemble diversity beyond the current `ResNet101 + strong recipe + short FixRes320` baseline.

## Why This Change

The current repo evidence says:

- `resnet101` is the best plain backbone path so far.
- `resnet152` did not improve over `resnet101`.
- `wide_resnet50_2` underperformed in Stage 0.
- `resnext101_64x4d` underperformed heavily, but it was trained with a much smaller batch size than the best `resnet101` recipe and therefore was not a clean cardinality test.
- FixRes helps but overfits quickly.
- The hidden-test plateau at `0.95` appears to be driven by correlated errors, tiny noisy validation, and class imbalance rather than one obvious implementation bug.

This makes a lower-cost cardinality probe reasonable. `resnext50_32x3d` is a custom but clearly residual-family variant with about `17.88M` parameters in this repo's 100-class setup, so it is far under the homework limit and is practical for a single GPU.

## Scope

This change proposes:

- adding `model.name: resnext50_32x3d` to the repo model builder
- documenting the model as a Tier 2 ResNet-family extension that requires report-visible explanation and citation
- adding a disciplined experiment path:
  - smoke test
  - short 256-resolution base train
  - full base train only if the short run is competitive
  - short FixRes refresh only if the full base train is competitive
  - ensemble evaluation against the current ResNet101 anchor
- adding explicit stop rules so a weak custom architecture is killed early

## Constraints And Guardrails

- The backbone must remain clearly in the ResNet family.
- Total parameters must remain strictly below `100M`.
- No external data may be introduced.
- The inference path must keep producing `prediction.csv`.
- The default workflow must remain practical on a single GPU.
- This change must not silently replace the current ResNet101 baseline.
- Because `resnext50_32x3d` is not a standard torchvision pretrained model, initialization behavior must be explicit and reproducible.

## Success Criteria

### Implementation Success

- The repo can build `resnext50_32x3d` through the same builder path used by training, validation, and inference.
- Parameter-limit enforcement remains active.
- The new model is available from config without code edits per run.
- The run artifacts remain compatible with the current validation, inference, and ensemble scripts.

### Experiment Success

- Short-run continuation gate:
  - the model trains stably
  - selected validation metrics are not catastrophically below the current ResNet101 anchor
  - there is no immediate overfit signature worse than the current FixRes path
- Full-run continuation gate:
  - either the model is competitive enough as a single model
  - or it provides materially better disagreement/rescue behavior for ensemble use
- Stretch target:
  - approach or exceed `95%` local validation accuracy after the full path and short FixRes refresh
  - show a realistic path to challenge the current `0.95` public leaderboard plateau

The `0.98` public target is treated as a stretch outcome, not an acceptance criterion for implementation.

## Non-Goals

- This change does not guarantee a leaderboard score.
- This change does not bless every custom grouped-convolution variant as homework-safe.
- This change does not replace the current tail-aware ResNet101 path unless it earns that status through evidence.

## Scientific Basis

- ResNet establishes the residual-family baseline and motivates residual scaling.
- ResNeXt motivates cardinality as a scaling axis that can outperform simple depth or width increases under similar complexity.
- Bag of Tricks supports disciplined training-recipe refinements rather than relying on architecture alone.
- Revisiting ResNets argues that training and scaling choices can matter as much as architecture changes.
- FixRes supports short resolution-refresh fine-tuning once a base model is already competitive.

These papers justify trying a grouped-bottleneck residual variant, but they do not justify skipping the repo's normal evidence gates.
