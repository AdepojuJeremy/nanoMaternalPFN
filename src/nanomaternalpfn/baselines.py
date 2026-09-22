"""Classical held-out baselines for synthetic maternal tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .synthetic import generate_task


@dataclass(frozen=True)
class BaselineResult:
    name: str
    loss: float
    accuracy: float


def _binary_log_loss(y_true: np.ndarray, p1: np.ndarray) -> float:
    eps = 1e-7
    p1 = np.clip(p1, eps, 1.0 - eps)
    y_true = y_true.astype(np.float64)
    return float(
        -np.mean(y_true * np.log(p1) + (1.0 - y_true) * np.log(1.0 - p1))
    )


def _constant_probabilities(y_context: np.ndarray, n_query: int) -> np.ndarray:
    return np.full(n_query, float(y_context.mean()), dtype=np.float64)


def _predict_logistic(task) -> np.ndarray:
    if np.unique(task.y_context).size < 2:
        return _constant_probabilities(task.y_context, len(task.y_query))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    model.fit(task.X_context, task.y_context)
    return model.predict_proba(task.X_query)[:, 1]


def _predict_random_forest(task, n_trees: int) -> np.ndarray:
    if np.unique(task.y_context).size < 2:
        return _constant_probabilities(task.y_context, len(task.y_query))

    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(task.X_context, task.y_context)
    return model.predict_proba(task.X_query)[:, 1]


def evaluate_baselines(
    *,
    n_tasks: int = 100,
    seed_start: int = 100_000,
    rf_trees: int = 100,
) -> list[BaselineResult]:
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")
    if rf_trees < 1:
        raise ValueError("rf_trees must be at least 1")

    totals = {
        "logistic_regression": {"loss": 0.0, "correct": 0, "n": 0},
        "random_forest": {"loss": 0.0, "correct": 0, "n": 0},
    }

    for seed in range(seed_start, seed_start + n_tasks):
        task = generate_task(seed=seed)

        predictions = {
            "logistic_regression": _predict_logistic(task),
            "random_forest": _predict_random_forest(task, rf_trees),
        }

        for name, p1 in predictions.items():
            y_true = task.y_query
            pred = (p1 >= 0.5).astype(np.int64)

            totals[name]["loss"] += _binary_log_loss(y_true, p1) * len(y_true)
            totals[name]["correct"] += int((pred == y_true).sum())
            totals[name]["n"] += len(y_true)

    return [
        BaselineResult(
            name=name,
            loss=values["loss"] / values["n"],
            accuracy=values["correct"] / values["n"],
        )
        for name, values in totals.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classical baselines on held-out synthetic tasks."
    )
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--rf-trees", type=int, default=100)
    args = parser.parse_args()

    results = evaluate_baselines(
        n_tasks=args.tasks,
        seed_start=args.seed_start,
        rf_trees=args.rf_trees,
    )

    print(f"held-out tasks: {args.tasks}")
    for result in results:
        print(
            f"{result.name:20s} | "
            f"loss {result.loss:.4f} | "
            f"query accuracy {result.accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
