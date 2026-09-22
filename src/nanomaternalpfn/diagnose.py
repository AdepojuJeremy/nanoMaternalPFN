"""Family-level diagnostics for nanoMaternalPFN and classical baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .baselines import _binary_log_loss, _predict_logistic, _predict_random_forest
from .model import NanoMaternalPFN
from .synthetic import generate_task
from .train import choose_device


@dataclass(frozen=True)
class FamilyResult:
    family: str
    model: str
    tasks: int
    loss: float
    accuracy: float


def evaluate_by_family(
    *,
    checkpoint: str,
    n_tasks: int = 500,
    seed_start: int = 100_000,
    rf_trees: int = 100,
    device_name: str = "auto",
) -> list[FamilyResult]:
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")

    device = choose_device(device_name)
    model = NanoMaternalPFN().to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    totals = defaultdict(
        lambda: {"loss": 0.0, "correct": 0, "n": 0, "tasks": 0}
    )

    with torch.no_grad():
        for seed in range(seed_start, seed_start + n_tasks):
            task = generate_task(seed=seed)
            family = task.metadata["task_family"]

            x_context = torch.from_numpy(task.X_context).unsqueeze(0).to(device)
            y_context = torch.from_numpy(task.y_context).unsqueeze(0).to(device)
            x_query = torch.from_numpy(task.X_query).unsqueeze(0).to(device)

            logits = model(x_context, y_context, x_query)
            probabilities = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

            predictions = {
                "nanoMaternalPFN": probabilities,
                "logistic_regression": _predict_logistic(task),
                "random_forest": _predict_random_forest(task, rf_trees),
            }

            for name, p1 in predictions.items():
                y_true = task.y_query
                pred = (p1 >= 0.5).astype(np.int64)
                key = (family, name)

                totals[key]["loss"] += _binary_log_loss(y_true, p1) * len(y_true)
                totals[key]["correct"] += int((pred == y_true).sum())
                totals[key]["n"] += len(y_true)
                totals[key]["tasks"] += 1

    family_order = ["mostly_linear", "mixed_nonlinear", "interaction_heavy"]
    model_order = ["nanoMaternalPFN", "logistic_regression", "random_forest"]

    results = []
    for family in family_order:
        for model_name in model_order:
            values = totals[(family, model_name)]
            if values["n"] == 0:
                continue
            results.append(
                FamilyResult(
                    family=family,
                    model=model_name,
                    tasks=values["tasks"],
                    loss=values["loss"] / values["n"],
                    accuracy=values["correct"] / values["n"],
                )
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare models by synthetic task family."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    args = parser.parse_args()

    results = evaluate_by_family(
        checkpoint=args.checkpoint,
        n_tasks=args.tasks,
        seed_start=args.seed_start,
        rf_trees=args.rf_trees,
        device_name=args.device,
    )

    current_family = None
    for result in results:
        if result.family != current_family:
            current_family = result.family
            print(f"\n{current_family} ({result.tasks} tasks)")
        print(
            f"{result.model:20s} | "
            f"loss {result.loss:.4f} | "
            f"accuracy {result.accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
