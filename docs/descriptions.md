# Visual Recognition using Deep Learning — Homework 1

## Purpose
This document is the single source of truth for the homework requirements, constraints, deliverables, penalties, and practical implementation boundaries.
It is intended to guide implementation, experimentation, reporting, and final submission.

## Assignment Information
- **Course:** Visual Recognition using Deep Learning
- **Semester:** 2026 Spring
- **Homework:** HW1
- **Release Date:** 2026/03/10 17:30
- **Deadline:** 2026/03/31 23:59 (Tuesday)

## Core Task
- **Task Type:** Image classification
- **Input:** RGB images
- **Prediction Target:** Object category ID
- **Number of Categories:** 100

## Dataset
- **Training / Validation:** 21,024 images
- **Test:** 2,344 images

## Grading Breakdown
- **Competition:** 85%
  - **Competition Performance:** 80%
  - **Code Reliability & Quality:** 5%
- **Report and Code Submission:** 15%

## Competition Platform
The competition is hosted on **CodaBench**.

## Competition Participation Steps
1. Register a CodaBench account.
2. Do **not** use two accounts for the competition.
3. Open the competition link.
4. Go to **My Submissions**.
5. Join the competition.
6. Enter your **student ID**.
7. Upload the submission file.
8. Add the submission to the leaderboard.
9. Check your result on the leaderboard.

## Competition Submission File Rule
- There is **no restriction** on the outer zip filename uploaded to CodaBench.
- However, the prediction file **inside the zip** must be named exactly:

`prediction.csv`

## Competition Grading Rule
- Competition grading uses the **private (hidden) leaderboard**.
- The **public leaderboard** is only for reference.

### Score Logic
- If accuracy is **below the weak baseline**, the competition score is **0**.
- If accuracy is **between weak baseline and strong baseline**, the score is interpolated from **60 to 80**.
- If accuracy is **between strong baseline and Rank 3**, the score is interpolated from **80 to 100**.
- **Rank 1 / 2 / 3** receive **100**.

### Approximate Anchors Mentioned in Slides
- **Weak baseline:** ~0.84
- **Strong baseline:** ~0.94
- **Rank 3:** unknown

## Hard Constraints and Limitations
Each violation below results in a **15-point penalty**.

1. **Pretrained weights are allowed.**
2. **External data is not allowed.**
3. **Model size must be less than 100M parameters.**
4. **Only ResNet may be used as the backbone.**
5. **Backbone modifications are allowed, but must be clearly explained in the report.**

## Safe Interpretation of the Backbone Rule
To minimize rule risk, the implementation should stay within models that are unmistakably in the ResNet family.

### Safe Choices
- ResNet-50
- ResNet-101
- ResNet-152
- ResNet-D style stem / downsampling modifications
- SE or CBAM modules inserted into a ResNet backbone, with full explanation in the report

### Risky Choices to Avoid
- Any non-ResNet backbone
- Architectures only loosely inspired by ResNet
- Any model whose legality under “ResNet backbone only” could be debated
- Any external pretraining source that is unclear or not standard

## Environment Recommendations
- **Recommended Python Version:** 3.9 or higher
- Use a virtual environment

### Suggested Environment Tools
- Poetry
- Conda
- Virtualenv

## Libraries and Frameworks
You are free to use libraries, packages, modules, and functions.
This includes using existing frameworks and public codebases, as long as you do not violate the homework rules.

## Code Reliability and Quality Requirements (5%)
1. Follow **PEP8**
2. Lint the code
3. Push the code to **GitHub**

### Required `README.md` Sections
Your GitHub repository must include a `README.md` with:
- Introduction
- Environment Setup
- Usage
- Performance Snapshot

## Report Requirements (15%)
The report must:
- Be in **PDF** format
- Be written in **English**

### Penalty
- **-5 points** if the report is not an English PDF

### Required Report Sections
1. **Introduction**
   - Introduce the task
   - Explain the core idea of the method
2. **Method**
   - Data preprocessing
   - Model architecture
   - Hyperparameter settings
   - Any modifications or adjustments
3. **Results**
   - Findings
   - Model performance
   - Example artifacts such as training curves and confusion matrix
4. **References**
   - Papers
   - GitHub references if used
   - Use a clean citation format

## Additional Experiment Requirement
The report is expected to include **meaningful additional experiments**.

### Important Rule
Merely tuning common hyperparameters is **not enough**.
This includes:
- Batch size
- Learning rate
- Optimizer choice

### Acceptable Additional Experiment Directions
- Add layers
- Remove layers
- Try a different loss function
- Add a meaningful ResNet-based modification
- Compare a legal architectural variant against baseline
- Compare augmentation or training strategies in a principled way

### For Each Additional Experiment, Include
1. **Hypothesis** — why the change might help
2. **Reasoning** — why it may or may not work
3. **Results and Implications** — what happened and what it means

## Final Submission to E3
Submit one zip file named:

`<STUDENT_ID>_HW1.zip`

It must contain:
- Code files and folders
- PDF report named:

`<STUDENT_ID>_HW1.pdf`

### Invalid Report Formats
- `.doc`
- `.docx`
- Any non-PDF format

## Submission Packaging Rule
Do **not** include the following inside the E3 zip:
- Dataset files
- `train.csv`
- `test.csv`
- image files such as `.jpg`
- model checkpoints

### Penalty
- **-5 points** if these are included

## GitHub Requirement
- Push the code to GitHub
- Include the GitHub repository link in the PDF report

## Late Policy
- **-20 points per additional late day**

### Example
If the raw score is 90 and the homework is submitted 2 days late, the final score becomes 50.

## Academic Integrity
The homework must be completed independently.

### Severe Penalties for Plagiarism
- Assignment score of **-100**
- Failing the course
- Report to academic integrity office

### Allowed
- Libraries and frameworks
- Torchvision model zoo
- GitHub repositories of published work
- AI assistance for implementation and troubleshooting

### Not Allowed
- External data
- Copying from classmates
- Reusing another student’s code or report

## FAQ-Relevant Practical Notes
### If GPU OOM Happens
- Reduce batch size
- Use a smaller model
- Use memory-saving methods

### If You Do Not Have a Local GPU
- Google Colab is acceptable

### If You Have Course Questions
- Ask on the **E3 forum first**

## Primary Optimization Target
Since the final competition score is based on the **private leaderboard**, the project must prioritize:
- generalization
- robustness
- repeatable validation
- low overfitting risk

## Practical Checklist
### Before Training
- Confirm dataset paths
- Confirm label format
- Confirm number of classes
- Confirm parameter count is under 100M
- Confirm chosen model is clearly ResNet-based
- Confirm no external data is used

### During Development
- Keep experiment logs
- Track validation performance carefully
- Lint regularly
- Keep code PEP8-compliant
- Commit and push to GitHub often

### Before Competition Submission
- Generate the correct prediction file
- Ensure the file inside the zip is named `prediction.csv`
- Verify the CSV column format matches competition expectations

### Before E3 Submission
- Verify zip filename
- Verify PDF report filename
- Ensure the report is English PDF
- Remove datasets and checkpoints
- Add GitHub link to report
- Submit before deadline