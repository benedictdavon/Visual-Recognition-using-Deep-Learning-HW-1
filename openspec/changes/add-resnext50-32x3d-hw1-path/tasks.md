# Tasks: ResNeXt-50 32x3d HW1 Path

- [x] Add `resnext50_32x3d` model-name support in `src/models/resnet_variants.py` using the existing residual-family builder path.
- [x] Keep parameter-count enforcement unchanged in `src/models/builder.py` and verify the new model remains under the strict `< 100M` rule.
- [x] Add explicit initialization handling for `resnext50_32x3d` so the repo does not silently claim standard pretrained support when using a custom grouped-width variant.
- [x] Ensure training, validation, and inference entrypoints can build and load the new model from config without special-case script edits.
- [x] Add one smoke config for `resnext50_32x3d` that proves end-to-end train/validate/infer compatibility on a single GPU.
- [x] Add one short-run config at the current base resolution to judge whether the model is competitive enough to continue.
- [x] Add one full-run config only if the short-run config passes its gate.
- [x] Add one short FixRes refresh config that warm-starts from the best `resnext50_32x3d` base checkpoint and disables mixup/cutmix.
- [x] Add documentation that explains the legality tier for this model.
- [x] Add documentation that explains the custom nature of `32x3d` and the initialization path.
- [x] Add documentation that explains the stop and continue criteria for smoke, short-run, full-run, and FixRes stages.
- [x] Add documentation that lists the exact train, validate, infer, diversity, and ensemble commands.
- [x] Add report-oriented notes describing how to cite and explain the custom ResNeXt-family variant without overstating its status.
- [x] Run at least smoke-level verification and record whether the custom initialization path behaves as designed.
- [x] Document the redirect condition to `resnext50_32x4d` if the custom `32x3d` path is clearly underpowered or the initialization path is too brittle.
