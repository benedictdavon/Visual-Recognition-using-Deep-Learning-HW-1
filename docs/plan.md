# HW1 Rank-1 Strategy Plan

## Goal
The goal of this project is to maximize the private leaderboard score under the homework constraints and realistically compete for Rank 1.

## Core Strategic Principle
Rank 1 is most likely achieved not by a single exotic idea, but by combining:

1. a **strong legal ResNet backbone**
2. a **modern high-performance training recipe**
3. **careful validation to optimize for private leaderboard generalization**
4. **strong inference with TTA and ensembling**

The central philosophy is:

> Build an extremely strong baseline first, then improve it through legal, high-yield changes, and only then use ensembles to maximize final performance.

## Non-Negotiable Constraints
All experiments must obey:
- only **ResNet** as the backbone
- **< 100M parameters**
- **no external data**
- pretrained weights are allowed
- any backbone modification must be clearly described in the report

## What “Rank-1” Means in Practice
To aim for rank 1, the pipeline must:
- generalize to the **private leaderboard**
- avoid overfitting to a single public split
- produce multiple independently strong models
- use legal diversity through architecture depth, lightweight ResNet modifications, augmentation, loss, and resolution changes

## Overall Development Stages
1. Build a clean and reproducible baseline
2. Build a strong training recipe
3. Scale to deeper ResNet variants
4. Add safe ResNet modifications
5. Optimize resolution and inference
6. Ensemble best models
7. Package report, code, and submission cleanly

---

## Stage 1 — Build a Clean Baseline

### Objective
Establish a working training / validation / inference / submission pipeline quickly.

### Recommended First Baseline
- `resnet50`
- pretrained weights enabled
- standard classification head for 100 classes
- input resolution `224`
- cross entropy loss
- AdamW or SGD with momentum
- cosine learning rate schedule with warmup
- simple augmentation only at first

### Why This Matters
The first baseline is not meant to win.
It is meant to:
- validate the codebase
- verify dataset loading
- verify label mapping
- verify parameter count
- verify train/val metrics make sense
- verify submission generation works

### Success Criteria for Baseline
The baseline must:
- train stably
- reach reasonable validation performance
- save checkpoints correctly
- load checkpoints correctly
- generate legal `prediction.csv`
- be fully reproducible with fixed seed

---

## Stage 2 — Strong Modern Training Recipe

### Objective
Convert the baseline into a serious model without changing the backbone family.

### High-Priority Techniques
Apply these incrementally and evaluate carefully:

#### 1. Strong Augmentation
Use a training augmentation pipeline such as:
- `RandomResizedCrop`
- `HorizontalFlip`
- `RandAugment`
- `ColorJitter` if useful
- `RandomErasing`
- normalization with pretrained model statistics

#### 2. Mixup / CutMix
Try:
- Mixup only
- CutMix only
- mixed probability of using either one

These often improve generalization and help reduce overconfidence.

#### 3. Label Smoothing
Use moderate label smoothing, for example:
- `0.05`
- `0.1`

#### 4. EMA
Maintain an exponential moving average of model weights and evaluate both:
- normal weights
- EMA weights

#### 5. Cosine Schedule with Warmup
Good default:
- warmup for a few epochs
- then cosine decay

#### 6. Better Optimizer / Regularization
Try:
- SGD + momentum + weight decay
- AdamW + weight decay

Compare based on validation performance, not just training loss.

### Strong Recipe Baseline Candidate
A strong recipe baseline can look like:
- ResNet-50 pretrained
- resolution 224 or 256
- RandAugment
- Mixup or CutMix
- label smoothing
- Random Erasing
- cosine + warmup
- EMA

### Important Note
The report says merely tuning LR, batch size, and optimizer is not enough as “additional experiments.”
Therefore, these should be part of the base training recipe, not your main novelty.

---

## Stage 3 — Strong Backbone Candidates

### Objective
Scale model capacity while staying safely within the ResNet family.

### Priority Order
1. `ResNet-50`
2. `ResNet-101`
3. `ResNet-152`

### Why These
- all are clearly legal under the homework rule
- all are standard and easy to justify in the report
- depth scaling often improves classification performance if training is stable

### What to Measure
For each backbone:
- parameter count
- validation accuracy
- training speed
- memory usage
- robustness across folds or splits
- whether TTA helps

### Key Principle
A bigger backbone is only worth it if:
- it improves validation
- it remains stable
- the compute cost is still manageable

---

## Stage 4 — Safe ResNet Modifications

### Objective
Add legal modifications that improve performance while keeping the model unmistakably ResNet-based.

### Candidate A — ResNet-D Style Improvements
Safe changes:
- improved stem
- modified downsampling path
- anti-aliasing-like downsample changes if still clearly ResNet-based

This is one of the best “safe” upgrade paths because it remains structurally close to standard ResNet.

### Candidate B — SE-ResNet
Add **Squeeze-and-Excitation** blocks into the ResNet stages.

Why try it:
- lightweight
- common
- often improves classification quality
- still easy to explain as a ResNet modification

### Candidate C — CBAM-ResNet
Add **CBAM** channel + spatial attention modules into the ResNet backbone.

Why try it:
- lightweight
- more expressive than plain channel attention
- may help if the dataset has fine-grained or cluttered visual structure

### Recommended Priority
1. ResNet-D
2. SE-ResNet
3. CBAM-ResNet

### Report Requirement
For every modification:
- explain what changed
- where it was inserted
- why it may help
- how much it changed parameter count
- how performance changed

---

## Stage 5 — Resolution Strategy

### Objective
Improve performance through better train/eval resolution choices.

### Resolutions to Try
- `224`
- `256`
- `288`
- `320` if compute permits

### Recommended Strategy
Use progressive resizing:
1. train at lower resolution
2. fine-tune at a higher resolution
3. evaluate at matching or slightly optimized evaluation resolution

### Why It Matters
Better resolution can improve accuracy, but it also increases:
- memory usage
- training time
- overfitting risk

### Practical Guideline
Use high resolution only on the strongest candidates, not on every experiment.

---

## Stage 6 — Validation Strategy for Private Leaderboard

### Objective
Make decisions that generalize to the hidden private leaderboard.

### Recommended Validation Design
- Use **stratified split**
- Prefer **3-fold or 5-fold cross-validation** for promising models
- Track both:
  - mean accuracy
  - variance across folds

### Why This Matters
Public leaderboard score may not reliably reflect private leaderboard rank.
A model with stable fold performance is often safer than a model that wins one split by chance.

### What to Log
For every run, track:
- seed
- backbone
- modifications
- input resolution
- loss
- augmentation recipe
- optimizer
- LR schedule
- batch size
- epochs
- best validation accuracy
- EMA accuracy
- fold results
- notes

### Decision Rule
Select final candidates primarily by:
1. strong cross-validation mean
2. low fold variance
3. complementary error patterns

---

## Stage 7 — Loss Function Experiments

### Objective
Satisfy the “meaningful additional experiments” requirement and possibly improve performance.

### Recommended Losses to Try
1. Cross Entropy
2. Cross Entropy + Label Smoothing
3. Focal Loss
4. Weighted Cross Entropy if class imbalance is present

### When to Use Weighted Loss
Only use class weighting if the training distribution is clearly imbalanced.
Do not assume weighting helps without evidence.

### What to Compare
- overall top-1 accuracy
- per-class accuracy
- confusion matrix
- calibration / confidence behavior

### Good Experiment Framing for Report
Example:
- **Hypothesis:** Focal Loss may help hard classes and reduce domination by easy examples.
- **Why it may fail:** If the dataset is already balanced and clean, Focal Loss may hurt optimization.
- **Result:** Compare against cross entropy under the same backbone and training schedule.

---

## Stage 8 — Data-Centric Analysis

### Objective
Improve score through understanding the dataset, not only architecture.

### Things to Check
- class imbalance
- suspiciously small classes
- duplicate images
- near-duplicates
- corrupted images
- mislabeled-looking outliers
- image size distribution
- aspect ratio distribution

### Why This Matters
High-ranking competition entries often come from better understanding of the dataset, not only more complex models.

### Safe Improvements
- class-aware sampler
- better resize / crop strategy
- per-class confusion analysis
- targeted augmentation for weak classes

---

## Stage 9 — Inference Optimization

### Objective
Maximize final accuracy after training.

### TTA Strategy
Start simple:
- original evaluation crop
- horizontal flip

If helpful, extend carefully to:
- multi-crop
- scale TTA

### Important Note
More TTA is not always better.
Use only TTA settings that consistently help validation.

### Prediction Aggregation
Average:
- logits, or
- probabilities

Keep the method consistent across all models in the ensemble.

---

## Stage 10 — Ensembling

### Objective
Combine multiple strong but diverse legal models to maximize private leaderboard performance.

### Best Ensemble Diversity Sources
- different backbone depths
- different legal ResNet modifications
- different resolutions
- different loss functions
- different seeds

### Recommended Final Ensemble Candidates
- `ResNet101-D`
- `SE-ResNet101-D`
- `ResNet152`
- optionally `CBAM-ResNet101-D`

### Ensemble Principles
Do not ensemble weak models just to have more models.
Prefer:
- fewer but stronger models
- diversity with similar accuracy
- models whose errors are not identical

### Final Ensemble Selection Rule
Choose the ensemble that offers the best balance of:
- high validation mean
- low variance
- consistent TTA gain
- complementary confusion behavior

---

## Recommended Experiment Ladder

### Phase A — Pipeline Validation
1. `ResNet50` baseline
2. `ResNet50` + stronger augmentation
3. `ResNet50` + strong recipe

### Phase B — Backbone Scaling
4. `ResNet101` + strong recipe
5. `ResNet152` + strong recipe

### Phase C — Backbone Modifications
6. `ResNet50-D` + strong recipe
7. `ResNet101-D` + strong recipe
8. `SE-ResNet50-D` + strong recipe
9. `SE-ResNet101-D` + strong recipe
10. `CBAM-ResNet50-D` + strong recipe
11. `CBAM-ResNet101-D` + strong recipe

### Phase D — Resolution and Loss
12. best candidate + higher resolution fine-tune
13. best candidate + Focal Loss
14. best candidate + class weighting if needed

### Phase E — Final Inference
15. best single model + TTA
16. best 2-model ensemble
17. best 3-model ensemble

---

## Final Recommended Priority Order
If time or compute is limited, prioritize:

1. `ResNet50` + strong recipe
2. `ResNet101` + strong recipe
3. `ResNet50-D`
4. `ResNet101-D`
5. `SE-ResNet101-D`
6. best model at higher resolution
7. best 2–3 model ensemble with TTA

---

## Suggested Folder / Experiment Structure

```text
project_root/
├── configs/
│   ├── model/
│   ├── train/
│   ├── aug/
│   ├── loss/
│   └── inference/
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── engine/
│   ├── utils/
│   └── inference/
├── scripts/
│   ├── train.py
│   ├── validate.py
│   ├── infer.py
│   ├── make_submission.py
│   └── ensemble.py
├── experiments/
│   ├── logs/
│   ├── oof/
│   ├── submissions/
│   └── reports/
├── README.md
└── requirements.txt
```

---

## What Must Be Recorded for the Report
For each serious experiment, store:
- experiment name
- backbone
- modification
- parameter count
- resolution
- training recipe
- loss
- optimizer
- schedule
- fold results
- best checkpoint
- confusion matrix
- final comments

This directly supports the report sections:
- Method
- Results
- Additional experiments

---

## Final Candidate Submission Philosophy
The final competition submission should **not** simply be:
- the model with the best training score
- the model with the best public leaderboard score

It should be:
- the model or ensemble with the strongest evidence of private-set generalization

---

## Minimum Success Outcome
A successful project should deliver:
- one strong legal baseline
- at least several meaningful additional experiments
- at least one safe ResNet modification
- strong report evidence
- one serious final single model
- one serious final ensemble

## Best-Case Outcome
The best-case final submission is:
- an ensemble of 2–4 high-performing legal ResNet-based models
- each trained with a strong modern recipe
- each validated carefully
- combined with modest TTA
- fully reproducible
- fully documented for the report