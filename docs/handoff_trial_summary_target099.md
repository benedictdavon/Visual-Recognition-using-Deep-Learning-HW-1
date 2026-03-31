# VRDL HW1 Trial Summary and 0.99 Gap Analysis

Last updated: 2026-03-25 (Asia/Taipei)  
Owner context: single-GPU workflow, ResNet-only homework constraint, no external data

---

## 1) Executive summary

- Current public leaderboard plateau: **0.95**.
- This score has been reached by multiple pipelines:
  - single-model FixRes320 EMA
  - 3-seed base-model ensemble
  - base+fixres mixed ensembles
- New ensemble sweep attempts on 2026-03-25 (2 submitted so far) also returned **0.95**.
- Main diagnosis: **prediction correlation ceiling**, not a single obvious code bug.
  - Across 12 saved `.npz` probability models, mean pairwise disagreement is only **~5.04%**.
  - Majority vote confidence is very high, so extra similar models mostly repeat the same predictions.
- Internal val metrics improved into low-90s on tiny val (300 images), but leaderboard does not move above 0.95.  
  This is consistent with a mismatch between tiny balanced val and hidden test behavior + insufficient ensemble diversity.

Target requested: **0.99**.

---

## 2) Constraints and task frame

- Task: 100-class image classification.
- Dataset:
  - train: 20,724
  - val: 300
  - test: 2,344
  - class folders: `0..99`
- Hard constraints (homework):  
  1. ResNet-only backbone family  
  2. `< 100M` params  
  3. no external data  
  4. submission file must remain `prediction.csv`

---

## 3) Data profile and why it matters

From `outputs/data_analysis_summary.json`:

- Train imbalance is strong:
  - min class count: **20**
  - max class count: **450**
  - max/min ratio: **22.5**
- Val set is tiny and balanced:
  - exactly **3 images per class** (100 classes total, 300 images).

Implication:
- Small architecture deltas are hard to rank reliably on official val.
- Leaderboard probing can disagree with val.
- Long-tail and calibration effects can dominate late-stage gains.

---

## 4) Chronological training run ledger

### 4.1 Completed runs with summary

| Run dir | Main config intent | Model | Params | Best val acc (selected metric) | Notes |
|---|---|---:|---:|---:|---|
| `20260311_235832_baseline_resnet50` | baseline | resnet50 | 23.71M | 86.33 | first full pipeline baseline |
| `20260312_002956_resnet101_strong_v1` | stronger recipe v1 | resnet101 | 42.71M | 86.33 | no gain vs baseline locally; later LB was better after fixes |
| `20260312_134125_resnet101_strong_v2_fast` | strong v2 fast | resnet101 | 42.71M | 89.67 | major local jump |
| `20260312_153549_resnet152_strong_v1` | depth scale test | resnet152 | 58.35M | 89.67 | no meaningful gain over resnet101 v2 |
| `20260313_122717_resnet101_strong_v2_320` | larger train/eval size | resnet101 | 42.71M | 88.33 | worse than 256 strong v2 fast |
| `20260313_225258_resnet101_fixres_finetune_320` | fixres stage | resnet101 | 42.71M | 91.00 | fast overfit dynamics appear |
| `20260314_001338_resnet101_strong_v2_fast_seed42` | seed sweep | resnet101 | 42.71M | 88.67 | seed variance present |
| `20260314_013534_resnet101_strong_v2_fast_seed3407` | seed sweep | resnet101 | 42.71M | 89.00 | seed variance present |
| `20260314_030251_resnet101_strong_v2_fast_seed2026` | seed sweep | resnet101 | 42.71M | 90.00 | best of base3 seeds |
| `20260315_232045_resnext101_64x4d_v1` | cardinality test | resnext101_64x4d | 81.61M | 85.67 | underperformed heavily |
| `20260316_181147_resnet101_fixres_finetune_320_seed42` | fixres seed sweep | resnet101 | 42.71M | 91.33 | very fast peak then drift |
| `20260316_184045_resnet101_fixres_finetune_320_seed3407` | fixres seed sweep | resnet101 | 42.71M | 91.00 | very fast peak then drift |
| `20260316_190430_resnet101_fixres_finetune_320_seed2026` | fixres seed sweep | resnet101 | 42.71M | 92.00 | best local val among fixres runs |
| `20260324_221550_fixres320_seed42_short3` | short fixres refresh | resnet101 | 42.71M | 91.00 | short 3-epoch variant |
| `20260324_224423_resnet101d_bse_sd_smoke` | architecture smoke | resnet101d_bse_sd | 47.55M | 73.33 | smoke-only sanity |

### 4.2 Incomplete/aborted runs

| Run dir | Model | Last logged epoch | Last logged best | Notes |
|---|---:|---:|---:|---|
| `20260313_002400_resnet101_resnetd_se_288` | resnet101 (resnetd+stage se) | 32/45 | 82.00 (EMA-selected metric in log) | interrupted before summary finalized |
| `20260313_003301_resnet101_plain_288_control` | resnet101 plain 288 | 33/45 | 89.00 | interrupted before summary finalized |
| `20260313_235111_resnet101_fixres_finetune_288` | resnet101 fixres288 | 5/8 | 90.33 | interrupted early |
| `20260324_223343_fixres320_seed3407_short3` | resnet101 | startup only | n/a | stopped before epoch 1 completed |
| `20260324_230548_resnet101d_bse_sd_strong` | resnet101d_bse_sd | 34/40 | 87.33 | plateaued below baseline strong-v2; interrupted before summary |

---

## 5) Submission scoreboard history (from provided screenshot + known artifacts)

| Submission ID | File name | Date | Score | Interpretation |
|---:|---|---|---:|---|
| 649898 | `infer_base3_plus_fixres320_seed2026.zip` | 2026-03-25 19:38 | 0.95 | base3 + fixres style blend, no lift above plateau |
| 649896 | `infer_base3_soft_ema.zip` | 2026-03-25 19:38 | 0.95 | soft EMA blend, still plateau |
| 626419 | `infer_ensemble_renet101_v2_3seeds.zip` | 2026-03-15 23:10 | 0.95 | 3-seed ensemble reached plateau |
| 620302 | `infer_fixres320_ema.zip` | 2026-03-13 23:46 | 0.95 | single fixres EMA reached same plateau |
| 617257 | `prediction.zip` | 2026-03-12 23:49 | 0.93 | intermediate stage |
| 615774 | `infer_resnet101_strong_v2_fast.zip` | 2026-03-12 15:26 | 0.94 | strong v2 fast uplift |
| 615656 | `prediction_resnet101_ama_fix.zip` | 2026-03-12 14:46 | 0.91 | pre-fix stage |
| 615547 | `prediction_resnet101.zip` | 2026-03-12 14:17 | 0.12 | early label mapping issue period |
| 615542 | `prediction_baseline.zip` | 2026-03-12 14:16 | 0.12 | early label mapping issue period |

Note: latest custom ensemble sweep on 2026-03-25 generated new zip files, and user reported 2 tested submissions were also **0.95**.

---

## 6) What changed across phases and observed effect

## Phase A: baseline pipeline + correctness fixes
- Added end-to-end train/validate/infer/submission pipeline.
- Fixed numeric label mapping restoration in submission.
- Outcome: massive practical jump from broken 0.12 era to valid competitive submissions.

## Phase B: stronger training recipe on ResNet101
- Switched from baseline recipe to stronger augmentation and optimization.
- Added EMA and stronger evaluation stack.
- Outcome: local val moved from ~86 to ~89-90; leaderboard improved to ~0.94.

## Phase C: depth scaling to ResNet152
- Objective: use more parameters via depth.
- Outcome: local best ~89.67 (not better than resnet101 strong-v2 best), leaderboard did not improve meaningfully.
- Interpretation: depth-only scaling was not the bottleneck.

## Phase D: FixRes fine-tune (320)
- Objective: reduce train-test resolution mismatch.
- Outcome:
  - local val peaks rose to 91-92 quickly
  - train accuracy rapidly approached ~99.7-99.9
  - later val drifted downward
- Interpretation: FixRes helps, but overfits rapidly under current late-stage recipe.

## Phase E: Seed ensembling
- Base3 seed runs and fixres seed runs were ensembled.
- Outcome: improved robustness but leaderboard remained ~0.95.

## Phase F: cardinality model trial (ResNeXt101-64x4d)
- Objective: use larger residual-family capacity under 100M.
- Outcome: underperformed strongly (85.67 local best), high compute cost.
- Interpretation: recipe/batch/optimization mismatch likely severe for this backbone in current setup.

## Phase G: custom residual architecture (ResNet101-D + bottleneck SE + stochastic depth)
- Implemented as `resnet101d_bse_sd` with ~47.55M params.
- Smoke passed; main run plateaued at ~87.33 by epoch 34/40 and did not match baseline strong-v2.
- Interpretation: current d-stem conversion + regularization combination under this recipe did not improve.

## Phase H: additional ensemble sweep on existing 12 probability files
- Generated candidates:
  - `raw6_equal`
  - `all12_equal`
  - `all12_rawx2`
  - `diverse8_raw_plus2ema`
- Outcome: two submitted today, both reported as 0.95.
- Interpretation: ensembling variants among highly correlated models is not enough.

---

## 7) Quantitative diagnostics behind the 0.95 plateau

## 7.1 Seed variance (local val)

- Base strong-v2 (3 seeds):
  - mean best: **89.222**
  - std: **0.694**
  - min/max: **88.667 / 90.000**

- FixRes320 (3 seeds):
  - mean best: **91.444**
  - std: **0.509**
  - min/max: **91.000 / 92.000**

Interpretation:
- Local val gains are real but modest across seeds.
- This did not translate to >0.95 LB.

## 7.2 Overfitting signature in FixRes

- In fixres runs, train acc reaches ~99.7-99.9 extremely fast.
- Best val typically occurs in first few epochs, then trends down.
- This is consistent with resolution finetune over-specialization.

## 7.3 Ensemble correlation ceiling

Using 12 saved `.npz` test-probability artifacts:
- number of models: **12**
- mean pairwise disagreement: **0.05035**
- min disagreement: **0.02645**
- max disagreement: **0.06954**

Interpretation:
- Models are too similar.
- Additional weighted voting mostly reorders a small subset of uncertain samples.
- This explains repeated 0.95 with many ensemble variants.

## 7.4 Compute cost scaling

Approx average epoch time from logs:
- ResNet101 strong-v2 seed2026: **2.44 min/epoch**
- ResNet152 strong-v1: **4.89 min/epoch**
- ResNeXt101-64x4d: **7.04 min/epoch**
- ResNet101d_bse_sd strong: **3.86 min/epoch**

Interpretation:
- Some higher-capacity tests are expensive without clear gain, so experiment allocation quality matters.

---

## 8) Latest ensemble sweep artifacts (2026-03-25)

Generated zip files:

- `outputs/ensemble_sweep/raw6_equal/infer_ensemble_raw6_equal.zip`
- `outputs/ensemble_sweep/all12_equal/infer_ensemble_all12_equal.zip`
- `outputs/ensemble_sweep/all12_rawx2/infer_ensemble_all12_rawx2.zip`
- `outputs/ensemble_sweep/diverse8_raw_plus2ema/infer_ensemble_diverse8_raw_plus2ema.zip`

Relative prediction differences:
- `raw6_equal` vs previous `infer_fixres320_ema`: ~3.50% label difference
- `all12_equal` vs previous `infer_fixres320_ema`: ~3.03%
- `all12_rawx2` vs previous `infer_fixres320_ema`: ~3.28%
- candidate-to-candidate differences are small (0.30%-1.19%)

User-reported submission outcome:
- two tested candidates today both scored **0.95**.

---

## 9) Why 0.99 is hard from current state

If leaderboard metric is top-1 accuracy on 2,344 test images:
- 0.95 corresponds to about 2,227 correct
- 0.99 corresponds to about 2,321 correct
- required net gain is about **+94 additional correct predictions**

This is a large jump from current plateau.

Given current evidence, +94 is unlikely from:
- minor weighting tweaks over the same model pool
- depth-only scaling
- repeating similar strong-v2/fixres recipes

---

## 10) Attempted vs unattempted optimization axes

## Attempted

- baseline ResNet50 and stronger ResNet101 recipe
- ResNet152 depth scaling
- ResNeXt101-64x4d cardinality trial
- FixRes320 finetune (multiple seeds)
- multiple seed ensembling and soft-vote blending
- custom residual path with bottleneck SE + stochastic depth

## Largely unattempted or not fully settled

- principled long-tail optimization under this dataset:
  - balanced softmax / class-balanced loss / logit adjustment
  - sampler-level class balancing
- robust model selection beyond tiny official val:
  - pseudo-fold CV for model ranking
  - train+val refit after locking hyperparameters
- substantially more diverse model families within legal residual scope:
  - wider residual variants tuned properly
  - alternative grouped settings tuned from scratch for this dataset
- stronger but still legal inference diversity:
  - multi-scale TTA variants
  - snapshot soups from one trajectory
  - disagreement-aware ensemble subset selection

---

## 11) Key reasons each high-level direction stalled

1. **Depth increase (101 -> 152) stalled**
- Added compute/parameters but not enough new representational mechanism.

2. **FixRes stalled at LB 0.95**
- Improved local val peak but caused fast overfit and did not create enough test-generalization margin.

3. **Ensembling stalled**
- Insufficient diversity among component predictions.

4. **Custom bse_sd residual run stalled**
- Current implementation/recipe combo underperformed baseline strong-v2 and did not justify continuation in its current form.

---

## 12) Artifacts to feed downstream LLM

### Core config files
- `configs/config.yaml`
- `configs/experiments/resnet101_strong_v2_fast_seed42.yaml`
- `configs/experiments/resnet101_strong_v2_fast_seed3407.yaml`
- `configs/experiments/resnet101_strong_v2_fast_seed2026.yaml`
- `configs/experiments/resnet101_fixres_finetune_320_seed42.yaml`
- `configs/experiments/resnet101_fixres_finetune_320_seed3407.yaml`
- `configs/experiments/resnet101_fixres_finetune_320_seed2026.yaml`
- `configs/experiments/resnext101_64x4d_strong.yaml`
- `configs/experiments/resnet101d_bse_sd_strong.yaml`

### Representative run folders
- `outputs/20260314_030251_resnet101_strong_v2_fast_seed2026`
- `outputs/20260316_190430_resnet101_fixres_finetune_320_seed2026`
- `outputs/20260312_153549_resnet152_strong_v1`
- `outputs/20260315_232045_resnext101_64x4d_v1`
- `outputs/20260324_230548_resnet101d_bse_sd_strong`

### Inference probability pool
- all `.npz` files under `outputs/infer_batch/*/test_probs_with_ids.npz`

### Ensemble outputs
- all files under `outputs/ensemble_sweep/*`

### Diagnostics
- `outputs/data_analysis_summary.json`
- `outputs/dataset_inspection.json`

---

## 13) Target statement for downstream optimizer LLM

Primary objective:
- Propose a **legal** path (ResNet-only, no external data, <100M params) to move from stable **0.95** to **>=0.99** leaderboard score.

Secondary objective:
- Maximize expected gain per GPU-hour under current single-GPU constraints.

Required from downstream proposal:
1. Explicit ranking of proposed experiment families by expected gain and risk.
2. Concrete run queue with stop criteria (what to kill early, what to continue).
3. A diversity-first ensemble plan that is materially different from current highly correlated pool.
4. Strict compliance checks for homework legality.

---

## 14) Current practical status

- Best observed public LB remains **0.95**.
- Two additional ensemble submissions on 2026-03-25 also returned **0.95**.
- One submission slot remains for today (per user note at the time of writing this document).

