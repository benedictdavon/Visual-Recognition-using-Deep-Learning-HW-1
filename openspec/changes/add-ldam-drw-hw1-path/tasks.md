## 1. Loss And Schedule Support

- [x] 1.1 Add an explicit LDAM loss option to the training loss builder with fail-fast validation for required class-count inputs.
- [x] 1.2 Add deferred reweighting controls so class-balanced weighting can remain inactive early and activate only after a configured late-stage boundary.
- [x] 1.3 Ensure ordinary non-LDAM recipes keep their current behavior when the new loss and DRW controls are not requested.

## 2. Experiment Path And Metadata

- [x] 2.1 Add `ResNet101` experiment configs for the LDAM-DRW path, including the late-stage activation boundary and any required weighting settings.
- [x] 2.2 Persist LDAM-DRW metadata in run artifacts so completed runs record the loss family, whether deferred reweighting was enabled, and when it activated.
- [x] 2.3 Keep the staged evidence clear enough to compare LDAM-DRW runs directly against the current non-LDAM anchors.

## 3. Verification And Documentation

- [x] 3.1 Run smoke-level verification that an LDAM-DRW recipe launches correctly with valid class counts and an explicit late-stage boundary.
- [x] 3.2 Verify that LDAM runs fail fast when class-count inputs or schedule settings are invalid instead of silently degrading to an existing loss path.
- [x] 3.3 Add experiment-facing notes that explain the LDAM-DRW hypothesis, the late-stage reweighting behavior, and the criteria for keeping or killing this path.
