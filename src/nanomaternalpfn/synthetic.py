"""Minimal synthetic maternal-task generator for nanoMaternalPFN.

This module intentionally starts simple. It produces *maternal-shaped* numerical
classification tasks for testing a PFN training pipeline. The defaults are
engineering priors, not validated clinical reference ranges or clinical risk
rules.

The generator varies the predictive rule from task to task. That variation is
the key property needed for PFN pretraining: the model should encounter many
small prediction problems rather than one fixed synthetic dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray


FEATURE_NAMES: tuple[str, ...] = (
    "age_years",
    "gestational_age_weeks",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "glucose_mg_dl",
)

# Broad engineering defaults for generator V0.
# They are not clinical thresholds and should later be replaced or calibrated
# using documented maternal-health data sources.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "age_years": (15.0, 45.0),
    "gestational_age_weeks": (4.0, 42.0),
    "systolic_bp": (85.0, 180.0),
    "diastolic_bp": (50.0, 120.0),
    "bmi": (16.0, 45.0),
    "glucose_mg_dl": (60.0, 220.0),
}


@dataclass(frozen=True)
class SyntheticMaternalTask:
    """One small classification task for PFN-style context/query learning."""

    X_context: NDArray[np.float32]
    y_context: NDArray[np.int64]
    X_query: NDArray[np.float32]
    y_query: NDArray[np.int64]
    feature_names: tuple[str, ...]
    seed: int
    metadata: dict[str, Any]


def _sample_features(
    rng: np.random.Generator,
    n_patients: int,
) -> NDArray[np.float64]:
    """Sample independent numerical maternal-shaped features.

    Independence is deliberate in V0. Correlation, missingness, categorical
    variables, longitudinal structure, and data-grounded marginals belong in
    later generator versions.
    """

    columns = [
        rng.uniform(low, high, size=n_patients)
        for low, high in (FEATURE_BOUNDS[name] for name in FEATURE_NAMES)
    ]
    return np.column_stack(columns)


def _sample_task_rule(
    rng: np.random.Generator,
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Sample a new outcome-generating rule for one synthetic task."""

    means = X.mean(axis=0, keepdims=True)
    scales = X.std(axis=0, keepdims=True)
    Xz = (X - means) / np.maximum(scales, 1e-8)

    n_features = X.shape[1]

    # Random sparse-ish linear rule.
    weights = rng.normal(0.0, 1.0, size=n_features)
    active = rng.random(n_features) < 0.7
    if not np.any(active):
        active[rng.integers(0, n_features)] = True
    weights *= active

    score = Xz @ weights

    # Add 0-3 randomly selected pairwise interactions.
    pairs = list(combinations(range(n_features), 2))
    n_interactions = int(rng.integers(0, 4))
    interaction_metadata: list[dict[str, Any]] = []

    if n_interactions:
        selected = rng.choice(len(pairs), size=n_interactions, replace=False)
        for pair_index in np.atleast_1d(selected):
            i, j = pairs[int(pair_index)]
            weight = float(rng.normal(0.0, 0.75))
            score += weight * Xz[:, i] * Xz[:, j]
            interaction_metadata.append(
                {
                    "features": (FEATURE_NAMES[i], FEATURE_NAMES[j]),
                    "weight": weight,
                }
            )

    noise_std = float(rng.uniform(0.2, 1.0))
    score += rng.normal(0.0, noise_std, size=X.shape[0])

    target_prevalence = float(rng.uniform(0.2, 0.8))
    threshold = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= threshold).astype(np.int64)

    metadata = {
        "linear_weights": {
            name: float(weight)
            for name, weight in zip(FEATURE_NAMES, weights, strict=True)
        },
        "interactions": interaction_metadata,
        "noise_std": noise_std,
        "target_positive_prevalence": target_prevalence,
        "positive_prevalence": float(y.mean()),
    }
    return y, metadata


def generate_task(
    seed: int,
    *,
    n_patients: int = 150,
    n_context: int = 100,
) -> SyntheticMaternalTask:
    """Generate one reproducible binary maternal-health-shaped task.

    Parameters
    ----------
    seed:
        Random seed. The same seed and arguments reproduce the same task.
    n_patients:
        Total number of rows in the synthetic task.
    n_context:
        Number of labeled context rows. Remaining rows become query rows.

    Returns
    -------
    SyntheticMaternalTask
        Context/query arrays plus metadata describing the sampled task rule.

    Notes
    -----
    This is a research scaffold for PFN pretraining. It does not simulate a
    validated maternal population and must not be used for clinical decisions.
    """

    if n_patients < 4:
        raise ValueError("n_patients must be at least 4")
    if not 2 <= n_context <= n_patients - 2:
        raise ValueError("n_context must leave at least 2 query rows")

    rng = np.random.default_rng(seed)

    X = _sample_features(rng, n_patients)
    y, metadata = _sample_task_rule(rng, X)

    # Randomize row order before the context/query split.
    order = rng.permutation(n_patients)
    X = X[order]
    y = y[order]

    X_context = X[:n_context].astype(np.float32)
    y_context = y[:n_context].astype(np.int64)
    X_query = X[n_context:].astype(np.float32)
    y_query = y[n_context:].astype(np.int64)

    return SyntheticMaternalTask(
        X_context=X_context,
        y_context=y_context,
        X_query=X_query,
        y_query=y_query,
        feature_names=FEATURE_NAMES,
        seed=seed,
        metadata=metadata,
    )


def main() -> None:
    """Generate one task and print a compact smoke-test summary."""

    task = generate_task(seed=42)
    prevalence = np.concatenate([task.y_context, task.y_query]).mean()

    print("nanoMaternalPFN synthetic task")
    print(f"context: {task.X_context.shape}")
    print(f"query:   {task.X_query.shape}")
    print(f"positive prevalence: {prevalence:.3f}")
    print(f"features: {', '.join(task.feature_names)}")


if __name__ == "__main__":
    main()
