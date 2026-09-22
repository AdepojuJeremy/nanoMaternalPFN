# nanoMaternalPFN

A lightweight tabular foundation model research project for maternal health.

## First milestone: synthetic maternal tasks

The project begins with synthetic PFN-style pretraining tasks. The synthetic
prior is an engineering scaffold, not a validated clinical population model or
clinical decision rule.

The current generator uses six numerical maternal-shaped features and varies
the predictive rule from task to task.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Generate one synthetic task

```python
from nanomaternalpfn import generate_task

task = generate_task(seed=42)

print(task.X_context.shape)  # (100, 6)
print(task.y_context.shape)  # (100,)
print(task.X_query.shape)    # (50, 6)
print(task.y_query.shape)    # (50,)
```

## Train

```bash
python -m nanomaternalpfn.train \
  --steps 10000 \
  --batch-size 8 \
  --save checkpoints/v1_prior_10000steps.pt
```

## UCI Maternal Health Risk evaluation

The first real-data experiment uses UCI Maternal Health Risk dataset 863.

The current model head is binary, so the target is:

```text
high risk = 1
low risk or mid risk = 0
```

The pretrained PFN remains frozen. Each evaluation episode supplies 100
labeled context rows and 50 query rows. Logistic Regression and Random Forest
are fitted only on the same context rows.

Exact duplicate rows are removed, and identical feature rows with conflicting
labels are removed before episode sampling.

```bash
python -m nanomaternalpfn.realdata \
  --checkpoint checkpoints/v1_prior_10000steps.pt \
  --episodes 100
```

Reported metrics include accuracy, balanced accuracy, AUROC, log loss, Brier
score, and expected calibration error.

Dataset source:

- UCI Machine Learning Repository, Maternal Health Risk, dataset 863
- DOI: https://doi.org/10.24432/C5DP5D
- License: CC BY 4.0

## Run tests

```bash
pytest -q
```

## Status

Early research prototype. Not for clinical use.
