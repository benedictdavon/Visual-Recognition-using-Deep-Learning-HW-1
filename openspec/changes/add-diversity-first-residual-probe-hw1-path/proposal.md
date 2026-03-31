## Why

The current plateau looks correlation-limited, not simply under-parameterized. A new residual-family architecture probe only makes sense if it changes the error pattern enough to help an ensemble, so the repo needs a disciplined path that treats one legal Tier 2 residual-family variant as an ensemble candidate first and only promotes it if its diversity evidence is real.

## What Changes

- Add a conservative experiment path for one already-supported Tier 2 residual-family candidate, such as a tuned `ResNeXt101` branch or an `SE-ResNet101` branch.
- Treat raw and EMA branches as distinct evaluation candidates when both are available, instead of assuming EMA is automatically the better ensemble branch.
- Add diversity-first continuation gates that prioritize disagreement, rescue count, and branch-specific validation evidence against the current `ResNet101` anchor.
- Add reproducibility requirements so candidate promotion records branch provenance and the diversity evidence used to justify ensemble inclusion.
- Keep the path explicitly framed as an ensemble-candidate probe rather than a new default anchor.

## Capabilities

### New Capabilities
- `diversity-first-residual-probe`: A legal experiment path for one already-supported Tier 2 residual-family variant whose continuation and adoption are based on diversity evidence against the current anchor.

### Modified Capabilities
- `evaluation-and-validation`: Candidate promotion changes to require explicit branch-specific validation evidence for raw and EMA branches when both are available.
- `experimentation-and-reproducibility`: Diversity-first probes change the evidence contract so branch provenance and diversity artifacts are part of the adoption bundle.
- `ensemble-and-search`: Candidate admission changes to treat branch-labeled raw and EMA artifacts as distinct inputs only when their provenance and diversity evidence are explicit.

## Impact

- Affected work is expected to center on experiment configs, validation / inference export conventions, diversity manifests, and run / branch metadata rather than on adding a brand-new backbone family.
- Existing supported Tier 2 residual-family variants remain the legal envelope for this path.
- No new external dependency is required.
- The backbone family rule, parameter budget, and no-external-data homework guardrails remain unchanged.
