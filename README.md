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

### Strict 5-fold frozen-transfer evaluation

For the paper-facing UCI result, use the stricter outer-fold protocol:

```bash
python -m nanomaternalpfn.realdata_cv \
  --checkpoint checkpoints/v1_prior_10000steps.pt \
  --contexts-per-fold 20 \
  --context 100
```

Each outer test fold is never used as context. Repeated contexts are sampled
only from the other four folds. Repeats are averaged within each fold, and the
final report gives mean, sample standard deviation, and a 95% t confidence
interval across the five held-out folds.

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


### Paired fold-wise model comparisons

Use the same strict 5-fold protocol to compare nanoMaternalPFN directly against
each baseline on matched held-out folds:

```bash
python -m nanomaternalpfn.realdata_compare \
  --checkpoint checkpoints/v1_prior_10000steps.pt \
  --contexts-per-fold 20 \
  --context 100
```

The command reports nanoMaternalPFN-minus-baseline mean differences, sample
standard deviation, and paired 95% t confidence intervals across the five outer
folds. Positive differences favor nanoMaternalPFN for accuracy, balanced
accuracy, and AUROC. Negative differences favor nanoMaternalPFN for log loss,
Brier score, and ECE.


## nuMoM2b controlled-data evaluation

nuMoM2b is the next planned external cohort because it is a much closer
clinical match to the maternal PFN setting than aggregate survey indicators.
NICHD DASH requires an approved data request before study files can be
downloaded. Controlled data must remain local and must not be committed.

After approved DASH data have been prepared as a one-row-per-participant
analysis CSV:

```bash
mkdir -p data/numom2b
# place the local analysis CSV at data/numom2b/analysis.csv
```

Inspect the available columns:

```bash
python -m nanomaternalpfn.numom2b inspect \
  --csv data/numom2b/analysis.csv
```

Create a local manifest from the committed template:

```bash
cp config/numom2b.template.json config/numom2b.local.json
```

Edit the local manifest to specify:

- one unique participant ID column
- exactly six numeric predictor columns
- one binary outcome column
- explicit positive and negative target values

Then run the same strict frozen-transfer design used for UCI:

```bash
python -m nanomaternalpfn.numom2b evaluate \
  --manifest config/numom2b.local.json \
  --checkpoint checkpoints/v1_prior_10000steps.pt \
  --contexts-per-fold 20 \
  --context 100
```

The adapter uses complete cases only and rejects duplicate participant IDs.
The local nuMoM2b data directory and local manifest are gitignored.


## MASS birthwt public external benchmark

A second no-approval external benchmark uses the public MASS `birthwt`
dataset (189 pregnancies). The target is low birth weight (<2.5 kg).

Predictors:

- maternal age
- maternal weight at last menstrual period
- race code
- smoking during pregnancy
- previous premature labours
- hypertension history
- uterine irritability
- first-trimester physician visits

The recorded infant birth-weight column is intentionally excluded because it
directly defines the target and would cause leakage.

This benchmark also tests feature-count transfer: nanoMaternalPFN was
pretrained on six synthetic columns, while `birthwt` supplies eight predictor
columns.

Run the same strict five-fold protocol:

```bash
python -m nanomaternalpfn.birthwt \
  --checkpoint checkpoints/v1_prior_10000steps.pt \
  --contexts-per-fold 20 \
  --context 100
```

The dataset is downloaded automatically from the public Rdatasets mirror of
the MASS package; no account or data-access approval is required.
