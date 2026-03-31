## 1. Candidate Path Setup

- [x] 1.1 Choose one already-supported Tier 2 residual-family candidate for the probe and add or update the experiment configs needed to run it conservatively against the current `ResNet101` anchor.
- [x] 1.2 Add branch-aware validation and inference conventions so raw and EMA branches can be exported or referenced as distinct candidate branches when both are credible.
- [x] 1.3 Keep the candidate path framed as an ensemble probe rather than an automatic anchor replacement in configs and experiment-facing docs.

## 2. Diversity Evidence Scaffolding

- [x] 2.1 Add or update manifests and naming conventions so diversity tooling can compare the probe against the current anchor with explicit raw / EMA branch provenance.
- [x] 2.2 Define continuation gates that combine acceptable local-metric gap with disagreement or rescue evidence rather than relying only on single-model rank.
- [x] 2.3 Define stop rules that kill the probe when branch-specific diversity value is weak even if the architecture itself is legal.

## 3. Verification And Documentation

- [x] 3.1 Run smoke-level verification that the chosen probe can produce branch-specific validation artifacts suitable for later diversity comparison.
- [x] 3.2 Verify that raw and EMA branches remain distinguishable throughout the validation and ensemble-candidate workflow.
- [x] 3.3 Document the final evidence bundle needed to justify ensemble inclusion of the probe without overstating it as the new default anchor.
