"""Synthetic maternal-task generator for nanoMaternalPFN.

V1 keeps the same public API as V0 but increases task diversity with:

- correlated maternal-shaped numerical features
- nonlinear univariate effects
- stronger pairwise interactions
- gated effects
- heterogeneous task families

These are still engineering priors for PFN research. They are not validated
clinical distributions, clinical thresholds, or clinical decision rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray


GENERATOR_VERSION = "v1"

FEATURE_NAMES: tuple[str, ...] = (
    "age_years",
    "gestational_age_weeks",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "glucose_mg_dl",
)

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


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))


def _scale_to_bounds(
    unit_values: NDArray[np.float64],
    feature_name: str,
) -> NDArray[np.float64]:
    low, high = FEATURE_BOUNDS[feature_name]
    return low + unit_values * (high - low)


def _sample_features(
    rng: np.random.Generator,
    n_patients: int,
) -> NDArray[np.float64]:
    """Sample correlated maternal-shaped numerical features.

    The dependency structure is intentionally simple and only exists to create
    a harder research prior. It is not a fitted model of a real population.
    """

    age_latent = rng.normal(size=n_patients)
    gestational_latent = rng.normal(size=n_patients)
    bp_latent = rng.normal(size=n_patients)
    metabolic_latent = rng.normal(size=n_patients)

    age_z = age_latent
    gestational_z = gestational_latent

    systolic_z = (
        0.75 * bp_latent
        + 0.15 * metabolic_latent
        + np.sqrt(1.0 - 0.75**2 - 0.15**2) * rng.normal(size=n_patients)
    )
    diastolic_z = (
        0.75 * bp_latent
        + 0.10 * metabolic_latent
        + np.sqrt(1.0 - 0.75**2 - 0.10**2) * rng.normal(size=n_patients)
    )
    bmi_z = (
        0.55 * metabolic_latent
        + 0.15 * age_latent
        + np.sqrt(1.0 - 0.55**2 - 0.15**2) * rng.normal(size=n_patients)
    )
    glucose_z = (
        0.55 * metabolic_latent
        + 0.10 * age_latent
        + np.sqrt(1.0 - 0.55**2 - 0.10**2) * rng.normal(size=n_patients)
    )

    latent = np.column_stack(
        [
            age_z,
            gestational_z,
            systolic_z,
            diastolic_z,
            bmi_z,
            glucose_z,
        ]
    )

    unit = _sigmoid(latent)

    columns = [
        _scale_to_bounds(unit[:, i], name)
        for i, name in enumerate(FEATURE_NAMES)
    ]
    return np.column_stack(columns)


def _nonlinear_term(
    x: NDArray[np.float64],
    transform: str,
    threshold: float,
) -> NDArray[np.float64]:
    if transform == "square":
        return x**2
    if transform == "absolute":
        return np.abs(x)
    if transform == "tanh":
        return np.tanh(1.5 * x)
    if transform == "hinge":
        return np.maximum(0.0, x - threshold)
    raise ValueError(f"unknown transform: {transform}")


def _sample_task_rule(
    rng: np.random.Generator,
    X: NDArray[np.float64],
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Sample a heterogeneous nonlinear outcome rule for one task."""

    means = X.mean(axis=0, keepdims=True)
    scales = X.std(axis=0, keepdims=True)
    Xz = (X - means) / np.maximum(scales, 1e-8)

    n_features = X.shape[1]

    family = str(
        rng.choice(
            ["mostly_linear", "mixed_nonlinear", "interaction_heavy"],
            p=[0.20, 0.50, 0.30],
        )
    )

    if family == "mostly_linear":
        active_probability = 0.70
        nonlinear_count = int(rng.integers(1, 3))
        interaction_count = int(rng.integers(0, 3))
        gate_count = int(rng.integers(0, 2))
    elif family == "mixed_nonlinear":
        active_probability = 0.45
        nonlinear_count = int(rng.integers(2, 5))
        interaction_count = int(rng.integers(1, 4))
        gate_count = int(rng.integers(0, 3))
    else:
        active_probability = 0.30
        nonlinear_count = int(rng.integers(2, 5))
        interaction_count = int(rng.integers(2, 5))
        gate_count = int(rng.integers(1, 3))

    linear_weights = rng.normal(0.0, 0.8, size=n_features)
    active = rng.random(n_features) < active_probability
    if not np.any(active):
        active[rng.integers(0, n_features)] = True
    linear_weights *= active

    score = Xz @ linear_weights

    transforms = ("square", "absolute", "tanh", "hinge")
    nonlinear_metadata: list[dict[str, Any]] = []

    for _ in range(nonlinear_count):
        feature_index = int(rng.integers(0, n_features))
        transform = str(rng.choice(transforms))
        weight = float(rng.normal(0.0, 1.1))
        threshold = float(rng.uniform(-0.75, 0.75))

        term = _nonlinear_term(
            Xz[:, feature_index],
            transform,
            threshold,
        )
        score += weight * term

        nonlinear_metadata.append(
            {
                "feature": FEATURE_NAMES[feature_index],
                "transform": transform,
                "weight": weight,
                "threshold": threshold if transform == "hinge" else None,
            }
        )

    pairs = list(combinations(range(n_features), 2))
    interaction_metadata: list[dict[str, Any]] = []

    if interaction_count:
        selected = rng.choice(
            len(pairs),
            size=min(interaction_count, len(pairs)),
            replace=False,
        )
        for pair_index in np.atleast_1d(selected):
            i, j = pairs[int(pair_index)]
            weight = float(rng.normal(0.0, 1.0))
            score += weight * Xz[:, i] * Xz[:, j]
            interaction_metadata.append(
                {
                    "features": (FEATURE_NAMES[i], FEATURE_NAMES[j]),
                    "weight": weight,
                }
            )

    gate_metadata: list[dict[str, Any]] = []

    for _ in range(gate_count):
        gate_feature, effect_feature = rng.choice(
            n_features,
            size=2,
            replace=False,
        )
        gate_feature = int(gate_feature)
        effect_feature = int(effect_feature)
        threshold = float(rng.uniform(-0.75, 0.75))
        weight = float(rng.normal(0.0, 1.0))

        gate = (Xz[:, gate_feature] > threshold).astype(np.float64)
        score += weight * gate * Xz[:, effect_feature]

        gate_metadata.append(
            {
                "gate_feature": FEATURE_NAMES[gate_feature],
                "effect_feature": FEATURE_NAMES[effect_feature],
                "threshold": threshold,
                "weight": weight,
            }
        )

    noise_std = float(rng.uniform(0.25, 0.9))
    score += rng.normal(0.0, noise_std, size=X.shape[0])

    target_prevalence = float(rng.uniform(0.2, 0.8))
    threshold = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= threshold).astype(np.int64)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "task_family": family,
        "linear_weights": {
            name: float(weight)
            for name, weight in zip(FEATURE_NAMES, linear_weights, strict=True)
        },
        "nonlinear_effects": nonlinear_metadata,
        "interactions": interaction_metadata,
        "gated_effects": gate_metadata,
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
    """Generate one reproducible binary maternal-health-shaped task."""

    if n_patients < 4:
        raise ValueError("n_patients must be at least 4")
    if not 2 <= n_context <= n_patients - 2:
        raise ValueError("n_context must leave at least 2 query rows")

    rng = np.random.default_rng(seed)

    X = _sample_features(rng, n_patients)
    y, metadata = _sample_task_rule(rng, X)

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
    task = generate_task(seed=42)
    prevalence = np.concatenate([task.y_context, task.y_query]).mean()

    print("nanoMaternalPFN synthetic task")
    print(f"generator: {task.metadata['generator_version']}")
    print(f"family: {task.metadata['task_family']}")
    print(f"context: {task.X_context.shape}")
    print(f"query:   {task.X_query.shape}")
    print(f"positive prevalence: {prevalence:.3f}")
    print(f"features: {', '.join(task.feature_names)}")


if __name__ == "__main__":
    main()
