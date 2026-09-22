# nanoMaternalPFN

A lightweight tabular foundation model research project for maternal health.

## First milestone: synthetic maternal tasks

The first implementation is a minimal synthetic task generator for PFN-style pretraining.

It does **not** attempt to simulate a validated clinical population. The feature bounds and outcome rules in V0 are engineering priors used to test the learning pipeline. They must not be interpreted as clinical thresholds or used for clinical decisions.

Each call samples a new small binary classification problem:

```text
maternal-shaped features
        ↓
task-specific linear effects
        ↓
optional pairwise interactions
        ↓
noise + variable class prevalence
        ↓
context patients + query patients
```

The current V0 features are:

- age
- gestational age
- systolic blood pressure
- diastolic blood pressure
- BMI
- glucose

The important PFN property is that the **prediction rule changes between tasks**.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Generate one task

```python
from nanomaternalpfn import generate_task

task = generate_task(seed=42)

print(task.X_context.shape)  # (100, 6)
print(task.y_context.shape)  # (100,)
print(task.X_query.shape)    # (50, 6)
print(task.y_query.shape)    # (50,)
print(task.metadata)
```

Or run the module directly:

```bash
python -m nanomaternalpfn.synthetic
```

## Run tests

```bash
pytest
```

## V0 acceptance criteria

- 150 patients by default
- 6 numerical maternal-shaped features
- 100 context rows and 50 query rows by default
- no missing values
- binary outcome with both classes represented
- reproducible tasks for a fixed seed
- different task rules across seeds
- feature values remain inside explicit generator bounds

## Planned generator progression

```text
V0  Independent numerical features
 ↓
V1  Nonlinear effects and richer interactions
 ↓
V2  Class imbalance controls
 ↓
V3  Missingness
 ↓
V4  Categorical maternal variables
 ↓
V5  Correlated variables
 ↓
V6  Data-grounded maternal priors
```

## Status

Early research prototype.
