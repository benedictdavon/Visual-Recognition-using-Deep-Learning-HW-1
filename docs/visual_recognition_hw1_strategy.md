# Visual Recognition HW1 — Findings, Papers, and Experiment Plan

## 1. Objective

This document consolidates:

- the current findings from the project status,
- the most relevant image-classification papers and why they matter,
- a prioritized experiment plan to maximize leaderboard performance under the homework rules.

The current problem setup is:

- **Task:** 100-class image classification
- **Constraints:**
  - ResNet-only backbone
  - under **100M** parameters
  - no external data
  - submission format: `prediction.csv`
- **Current best public leaderboard score:** **0.94**
- **Current best practical model:** `resnet101_strong_v2_fast`
- **Current practical parameter usage:** about **42.7M**

---

## 2. Current Findings from Existing Experiments

### 2.1 What is already working

The project is already in a strong state:

- end-to-end training, validation, checkpointing, inference, and submission all work,
- the label-mapping bug has been fixed,
- EMA, stronger augmentation, Mixup/CutMix, label smoothing, and TTA are already integrated,
- diagnostics such as hardest classes, confusion matrix, and misclassification export are available.

This is important because it means the next gains should come from **careful optimization**, not from rebuilding the pipeline.

### 2.2 Main empirical findings so far

1. **Submission correctness mattered a lot.**
   The label restoration / numeric-folder mapping fix was one of the biggest practical improvements.

2. **ResNet101 currently beats ResNet152 in practice.**
   Although ResNet152 uses more parameters, it did not outperform the best ResNet101 run on the public leaderboard.

3. **Depth alone is not enough.**
   The current evidence suggests that simply scaling from 101 to 152 does not guarantee better leaderboard performance.

4. **EMA is helpful.**
   EMA is consistently useful in the current setup and should remain part of the default training recipe.

5. **The validation set is tiny.**
   With only 300 validation images and 3 images per class, the public leaderboard can diverge from validation accuracy. This makes robust model selection more difficult.

6. **The train distribution is imbalanced.**
   Since the class imbalance is strong while validation is balanced, long-tail handling becomes a high-value direction.

### 2.3 Core conclusion

The next improvement is **more likely to come from training strategy, scaling strategy, imbalance handling, and inference refinement** than from blindly increasing model size.

---

## 3. High-Value Papers to Guide the Next Stage

Below are the most relevant papers for this homework, with direct practical implications.

### 3.1 Bag of Tricks for Image Classification with Convolutional Neural Networks

- **Paper:** Tong He et al., 2018 / CVPR 2019
- **Link:** https://arxiv.org/abs/1812.01187
- **Why it matters:** This is one of the most important papers showing that training refinements can dramatically improve standard CNNs without changing the backbone family.
- **Key result:** It raised **ResNet-50** from **75.3%** to **79.29%** top-1 on ImageNet through training refinements and minor recipe changes.
- **Relevance to HW1:** Supports the idea that the next gains may come from recipe improvement rather than a deeper network.

### 3.2 Fixing the Train-Test Resolution Discrepancy (FixRes)

- **Paper:** Hugo Touvron et al., 2019
- **Link:** https://arxiv.org/abs/1906.06423
- **Why it matters:** It shows that train-time and test-time resolution mismatch hurts performance, and that a cheap fine-tuning stage at the target test resolution can improve results.
- **Key result:** The paper reports **79.8%** top-1 for ResNet-50 on ImageNet with a resolution-aware strategy.
- **Relevance to HW1:** This is a very practical next step because the current system already supports inference and TTA cleanly.

### 3.3 ResNet Strikes Back: An Improved Training Procedure in timm

- **Paper:** Ross Wightman et al., 2021
- **Link:** https://arxiv.org/abs/2110.00476
- **Why it matters:** It re-establishes strong vanilla ResNet baselines using modern training procedures.
- **Key result:** The paper reports **80.4%** top-1 for vanilla ResNet-50 at 224 resolution on ImageNet without extra data or distillation.
- **Relevance to HW1:** Strong evidence that **modern training recipe quality** matters at least as much as raw architecture scale.

### 3.4 Revisiting ResNets: Improved Training and Scaling Strategies

- **Paper:** Irwan Bello et al., 2021
- **Link:** https://arxiv.org/abs/2103.07579
- **Why it matters:** This paper explicitly studies whether gains come from architecture, training, or scaling strategy.
- **Key message:** The paper argues that **training and scaling strategies may matter more than architectural changes**, and it recommends more careful scaling of depth, width, and resolution.
- **Relevance to HW1:** This matches the current empirical results: ResNet152 did not automatically beat ResNet101.

### 3.5 Balanced Meta-Softmax / Balanced Softmax for Long-Tailed Visual Recognition

- **Paper:** Jiawei Ren et al., 2020
- **Link:** https://arxiv.org/abs/2007.10740
- **Why it matters:** The training set is imbalanced, while the validation set is balanced and tiny.
- **Relevance to HW1:** This makes Balanced Softmax or class-distribution-aware loss a high-priority direction.

### 3.6 Squeeze-and-Excitation Networks

- **Paper:** Jie Hu et al., 2017
- **Link:** https://arxiv.org/abs/1709.01507
- **Why it matters:** SE blocks improve channel attention with small extra cost.
- **Relevance to HW1:** Only relevant **if** the instructor allows minor ResNet-family architectural modifications such as SE-ResNet.

---

## 4. Parameter-Budget Analysis Under the 100M Limit

The goal should **not** be to spend parameters just because they are available. The goal should be to spend parameters **only when the added capacity is likely to improve leaderboard score**.

### 4.1 Official TorchVision reference models

The following parameter counts and ImageNet top-1 accuracies come from the current TorchVision documentation.

#### Vanilla ResNet options

- **ResNet101**
  - params: **44,549,160**
  - ImageNet top-1 (IMAGENET1K_V2): **81.886**
  - official doc: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html

- **ResNet152**
  - params: **60,192,808**
  - ImageNet top-1 (IMAGENET1K_V2): **82.284**
  - official doc: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html

#### Broader residual-family options (only if legal)

- **Wide-ResNet50-2**
  - params: **68,883,240**
  - ImageNet top-1 (IMAGENET1K_V2): **81.602**
  - official doc: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html

- **ResNeXt101-64x4d**
  - params: **83,455,272**
  - ImageNet top-1: **83.246**
  - official doc: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnext101_64x4d.html

### 4.2 What these numbers imply

1. **ResNet152 is only a moderate jump over ResNet101** in ImageNet top-1, not a dramatic one.
2. **The best under-100M residual-family candidate is likely ResNeXt101-64x4d**, but only if the homework rule allows it.
3. If the rule is interpreted strictly as only **ResNet50 / ResNet101 / ResNet152**, then the safest choice remains vanilla ResNet.
4. Your own experiments already show that **capacity utilization is not the main bottleneck yet**.

### 4.3 Practical interpretation for HW1

- If the instructor means **strict vanilla ResNet**, stay within **ResNet101/152**.
- If the instructor allows the broader residual family, then **ResNeXt101-64x4d** becomes the most attractive “use more of the 100M budget” option.
- However, even in that case, the model switch should come **after** recipe and scaling improvements are tested on the current strong baseline.

---

## 5. Strategy Recommendation

## 5.1 Main recommendation

Do **not** treat unused parameters as a problem.

The best current evidence suggests this order of importance:

1. **training recipe quality**
2. **resolution / scaling strategy**
3. **imbalance handling**
4. **model selection robustness**
5. **inference refinement**
6. **larger backbone**

That means the best next move is not “jump to the biggest model.”
The best next move is:

- keep **ResNet101** as the anchor,
- strengthen the training/scaling strategy,
- only then test larger legal backbones.

### 5.2 Best strict interpretation plan

If “ResNet-only” means only vanilla ResNet variants, then the best path is:

- stay on **ResNet101** first,
- add **FixRes-style fine-tuning**,
- add **long-tail-aware loss or sampling**,
- improve model selection and inference,
- then re-run **ResNet152** with the improved recipe.

### 5.3 Best broader-family plan

If the homework permits the broader residual family, then the best “maximize parameters under 100M” candidate is:

- **ResNeXt101-64x4d**

But even then, it should be tested **after** improving the training and scaling recipe on ResNet101.

---

## 6. Detailed Experiment Plan

This section prioritizes experiments based on expected return, implementation cost, and legality risk.

### Phase A — Highest ROI, lowest rule risk

These experiments should be done first.

#### A1. FixRes-style fine-tuning

**Goal:** reduce train-test resolution mismatch.

**Plan:**
- train with the current best ResNet101 recipe,
- fine-tune the best checkpoint for 5–10 epochs at a higher resolution,
- test inference at the matched higher resolution.

**Suggested variants:**
- train at 224, fine-tune/infer at 288
- train at 224, fine-tune/infer at 320

**Why high priority:**
- supported strongly by FixRes,
- low engineering risk,
- can often improve performance even without changing the base model.

#### A2. Long-tail handling

**Goal:** address the strong class imbalance in the training set.

**Options to test:**
- weighted cross-entropy
- class-aware sampling
- Balanced Softmax
- deferred rebalancing only in the final training stage

**Recommended order:**
1. class-aware sampler
2. Balanced Softmax
3. late-stage rebalancing with the best of the above

**Why high priority:**
- the train set is imbalanced,
- the evaluation split is balanced,
- this mismatch makes long-tail methods particularly relevant.

#### A3. Strong-early / clean-late schedule

**Goal:** use strong regularization early, then let the model specialize late.

**Plan:**
- keep strong augmentation early,
- reduce augmentation strength in the final stage,
- reduce or disable Mixup/CutMix in the final epochs,
- keep EMA active.

**Why high priority:**
- often improves final decision boundaries,
- useful when leaderboard and tiny validation can be noisy.

#### A4. Checkpoint averaging / soup

**Goal:** reduce noise from selecting one possibly lucky checkpoint.

**Plan:**
- average top 3–5 checkpoints from the same training trajectory,
- compare against the single best epoch,
- optionally use EMA checkpoint as one candidate.

**Why high priority:**
- tiny validation makes single-checkpoint selection fragile.

#### A5. Better inference stack

**Goal:** squeeze final gains from inference without retraining.

**Options:**
- horizontal flip TTA
- resize/crop TTA at 2–3 scales
- logit averaging across a few strong checkpoints

**Important caution:**
- keep TTA modest,
- avoid overly complicated test-time pipelines that may overfit validation heuristics.

---

### Phase B — Strong ablation and model-selection upgrades

#### B1. Multi-seed reruns of the best recipe

**Goal:** estimate variance and reduce luck.

**Plan:**
- run the best ResNet101 recipe with at least 3 seeds,
- compare public leaderboard spread,
- select models for ensemble or checkpoint soup.

**Why important:**
- with a tiny validation set, seed variance can be significant.

#### B2. Cross-validation for model selection

**Goal:** reduce over-reliance on the 300-image val split.

**Plan:**
- create stratified folds on train,
- evaluate recipe stability across folds,
- lock the recipe,
- retrain once on train + val.

**Why important:**
- the current val set is too small to serve as the only selector for fine-grained decisions.

#### B3. Per-class error mining

**Goal:** find classes that need targeted help.

**Plan:**
- inspect hardest classes,
- inspect confusion pairs,
- check whether mistakes are caused by:
  - class imbalance,
  - low resolution,
  - visually similar categories,
  - augmentation instability.

**Possible actions:**
- targeted class weights,
- higher-res fine-tuning,
- more conservative crops for confusing categories.

---

### Phase C — Backbone scaling inside the strict vanilla-ResNet rule

Only move here after Phase A has been tested well.

#### C1. Re-run ResNet152 using the improved recipe

**Goal:** test whether the deeper backbone becomes useful when the recipe is stronger.

**Plan:**
- do not reuse the old ResNet152 recipe unchanged,
- copy the best ResNet101 training recipe exactly,
- add FixRes-style fine-tune if it worked on ResNet101,
- compare both validation and public leaderboard.

**Reason:**
Your current ResNet152 result only proves that **naive scaling did not help in that run**. It does not prove that ResNet152 is fundamentally worse for this dataset.

#### C2. Resolution scaling for ResNet152

**Goal:** test whether the larger model benefits more from higher resolution.

**Plan:**
- start with 224 base training,
- fine-tune at 288 or 320,
- keep EMA and modest TTA.

**Reason:**
Larger models often benefit more when the optimization and scaling strategy are also adapted.

---

### Phase D — Only if broader residual-family models are allowed

This phase should only be attempted if the rule interpretation clearly permits it.

#### D1. ResNeXt101-64x4d

**Why this is the best candidate:**
- under the 100M limit,
- official TorchVision model,
- strongest ImageNet top-1 among the listed residual-family candidates.

**Plan:**
- initialize from pretrained weights,
- use the best ResNet101 recipe,
- add FixRes if it works well,
- compare against the best vanilla ResNet.

#### D2. Wide-ResNet50-2

**Why test it:**
- it spends more parameters on width rather than depth,
- some datasets benefit more from width scaling.

**Plan:**
- use as a secondary broader-family candidate,
- compare against ResNet101 and ResNeXt101.

#### D3. SE-ResNet or similar minor modifications

**Only if legal.**

**Why test it:**
- SE blocks often provide a good accuracy/parameter trade-off.

**Risk:**
- depending on the course interpretation, this may be considered a modified architecture rather than a pure ResNet.

---

## 7. Prioritized Run List

Below is the recommended run order.

### Tier 1 — Do these first

1. **ResNet101 + current best recipe + FixRes fine-tune at 288**
2. **ResNet101 + current best recipe + FixRes fine-tune at 320**
3. **ResNet101 + Balanced Softmax**
4. **ResNet101 + Balanced Softmax + FixRes**
5. **ResNet101 + best above + reduced late-stage Mixup/CutMix**
6. **ResNet101 + checkpoint soup / checkpoint averaging**
7. **ResNet101 + best above + 3 seeds**

### Tier 2 — After the best recipe is identified

8. **ResNet152 + best ResNet101 recipe copied exactly**
9. **ResNet152 + best recipe + FixRes**
10. **ResNet152 + best recipe + improved inference stack**

### Tier 3 — Only if broader family is legal

11. **ResNeXt101-64x4d + best recipe**
12. **ResNeXt101-64x4d + best recipe + FixRes**
13. **Wide-ResNet50-2 + best recipe**

---

## 8. Final Recommendation

### 8.1 Most likely best path to improve the leaderboard

The strongest next move is:

- **keep ResNet101 as the main backbone**,
- apply **FixRes-style fine-tuning**,
- test **Balanced Softmax or class-aware rebalancing**,
- improve **late-stage training and checkpoint selection**,
- then re-evaluate whether ResNet152 is actually worth the extra cost.

### 8.2 Best “maximize 100M budget” answer

If the rule allows the broader residual family, the most attractive under-100M candidate is:

- **ResNeXt101-64x4d**

If the rule is strict vanilla ResNet only, then the best use of the budget is:

- **a better-trained ResNet152**, not a naive deeper rerun.

### 8.3 Final conclusion

At this point, the biggest opportunity is **not** “use more parameters at all costs.”
The biggest opportunity is:

- smarter scaling,
- better training recipe,
- better handling of class imbalance,
- more robust model selection,
- better inference.

Only after that should parameter scaling become the main lever.

---

## 9. Source References

### Papers

- Bag of Tricks for Image Classification with Convolutional Neural Networks  
  https://arxiv.org/abs/1812.01187

- Fixing the Train-Test Resolution Discrepancy  
  https://arxiv.org/abs/1906.06423

- ResNet Strikes Back: An Improved Training Procedure in timm  
  https://arxiv.org/abs/2110.00476

- Revisiting ResNets: Improved Training and Scaling Strategies  
  https://arxiv.org/abs/2103.07579

- Balanced Meta-Softmax for Long-Tailed Visual Recognition  
  https://arxiv.org/abs/2007.10740

- Squeeze-and-Excitation Networks  
  https://arxiv.org/abs/1709.01507

### Official model references

- TorchVision ResNet101  
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html

- TorchVision ResNet152  
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html

- TorchVision Wide-ResNet50-2  
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html

- TorchVision ResNeXt101-64x4d  
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnext101_64x4d.html
