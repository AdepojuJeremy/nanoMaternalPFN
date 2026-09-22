"""Research benchmark for calibration and lightweight deployment metrics."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import _binary_log_loss
from .model import NanoMaternalPFN, count_parameters
from .synthetic import generate_task
from .train import choose_device


def brier_score(y_true: np.ndarray, p1: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    p1 = p1.astype(np.float64)
    return float(np.mean((p1 - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    p1: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Equal-width binary ECE for positive-class probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    y_true = y_true.astype(np.float64)
    p1 = np.clip(p1.astype(np.float64), 0.0, 1.0)
    bin_ids = np.minimum((p1 * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    total = len(y_true)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(y_true[mask].mean())
        predicted = float(p1[mask].mean())
        ece += (count / total) * abs(observed - predicted)

    return float(ece)


def _peak_rss_mb() -> float:
    """Return process peak resident memory in MB on macOS/Linux."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _constant_probabilities(y_context: np.ndarray, n_query: int) -> np.ndarray:
    return np.full(n_query, float(y_context.mean()), dtype=np.float64)


def _fit_logistic(task):
    if np.unique(task.y_context).size < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    model.fit(task.X_context, task.y_context)
    return model


def _fit_random_forest(task, n_trees: int):
    if np.unique(task.y_context).size < 2:
        return None
    model = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(task.X_context, task.y_context)
    return model


def _finalize(
    *,
    model_name: str,
    y_true: list[np.ndarray],
    p1: list[np.ndarray],
    adaptation_seconds: float,
    inference_seconds: float,
    n_tasks: int,
    model_size_bytes: float,
    parameters: int | None,
    n_bins: int,
) -> dict[str, Any]:
    y = np.concatenate(y_true)
    probabilities = np.concatenate(p1)
    predictions = (probabilities >= 0.5).astype(np.int64)

    return {
        "model": model_name,
        "tasks": n_tasks,
        "accuracy": float((predictions == y).mean()),
        "log_loss": _binary_log_loss(y, probabilities),
        "brier": brier_score(y, probabilities),
        "ece": expected_calibration_error(y, probabilities, n_bins=n_bins),
        "adaptation_ms_per_task": 1000.0 * adaptation_seconds / n_tasks,
        "inference_ms_per_task": 1000.0 * inference_seconds / n_tasks,
        "model_size_mb": model_size_bytes / (1024.0 * 1024.0),
        "parameters": parameters,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _benchmark_pfn(
    *,
    checkpoint: str,
    n_tasks: int,
    seed_start: int,
    device_name: str,
    n_bins: int,
) -> dict[str, Any]:
    device = choose_device(device_name)
    model = NanoMaternalPFN().to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    warmup = generate_task(seed=seed_start)
    with torch.no_grad():
        model(
            torch.from_numpy(warmup.X_context).unsqueeze(0).to(device),
            torch.from_numpy(warmup.y_context).unsqueeze(0).to(device),
            torch.from_numpy(warmup.X_query).unsqueeze(0).to(device),
        )
        _synchronize(device)

    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    inference_seconds = 0.0

    with torch.no_grad():
        for seed in range(seed_start, seed_start + n_tasks):
            task = generate_task(seed=seed)
            x_context = torch.from_numpy(task.X_context).unsqueeze(0).to(device)
            y_context = torch.from_numpy(task.y_context).unsqueeze(0).to(device)
            x_query = torch.from_numpy(task.X_query).unsqueeze(0).to(device)

            _synchronize(device)
            start = time.perf_counter()
            logits = model(x_context, y_context, x_query)
            _synchronize(device)
            inference_seconds += time.perf_counter() - start

            probabilities = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
            all_y.append(task.y_query)
            all_p.append(probabilities)

    return _finalize(
        model_name="nanoMaternalPFN",
        y_true=all_y,
        p1=all_p,
        adaptation_seconds=0.0,
        inference_seconds=inference_seconds,
        n_tasks=n_tasks,
        model_size_bytes=float(Path(checkpoint).stat().st_size),
        parameters=count_parameters(model),
        n_bins=n_bins,
    )


def _benchmark_classical(
    *,
    model_name: str,
    n_tasks: int,
    seed_start: int,
    rf_trees: int,
    n_bins: int,
) -> dict[str, Any]:
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    adaptation_seconds = 0.0
    inference_seconds = 0.0
    serialized_sizes: list[int] = []

    for seed in range(seed_start, seed_start + n_tasks):
        task = generate_task(seed=seed)

        start = time.perf_counter()
        if model_name == "logistic_regression":
            model = _fit_logistic(task)
        elif model_name == "random_forest":
            model = _fit_random_forest(task, rf_trees)
        else:
            raise ValueError(f"unknown model: {model_name}")
        adaptation_seconds += time.perf_counter() - start

        if model is None:
            start = time.perf_counter()
            probabilities = _constant_probabilities(
                task.y_context,
                len(task.y_query),
            )
            inference_seconds += time.perf_counter() - start
            serialized_sizes.append(0)
        else:
            start = time.perf_counter()
            probabilities = model.predict_proba(task.X_query)[:, 1]
            inference_seconds += time.perf_counter() - start
            serialized_sizes.append(
                len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            )

        all_y.append(task.y_query)
        all_p.append(probabilities)

    return _finalize(
        model_name=model_name,
        y_true=all_y,
        p1=all_p,
        adaptation_seconds=adaptation_seconds,
        inference_seconds=inference_seconds,
        n_tasks=n_tasks,
        model_size_bytes=float(np.mean(serialized_sizes)),
        parameters=None,
        n_bins=n_bins,
    )


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.worker_model == "pfn":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for PFN benchmarking")
        return _benchmark_pfn(
            checkpoint=args.checkpoint,
            n_tasks=args.tasks,
            seed_start=args.seed_start,
            device_name=args.device,
            n_bins=args.ece_bins,
        )

    return _benchmark_classical(
        model_name=args.worker_model,
        n_tasks=args.tasks,
        seed_start=args.seed_start,
        rf_trees=args.rf_trees,
        n_bins=args.ece_bins,
    )


def _run_isolated_worker(
    *,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "nanomaternalpfn.benchmark",
        "--worker-model",
        model_name,
        "--tasks",
        str(args.tasks),
        "--seed-start",
        str(args.seed_start),
        "--rf-trees",
        str(args.rf_trees),
        "--ece-bins",
        str(args.ece_bins),
        "--device",
        args.device,
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return json.loads(completed.stdout)


def _format_parameters(value: int | None) -> str:
    return "-" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark accuracy, calibration, and efficiency."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tasks", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument(
        "--worker-model",
        choices=("pfn", "logistic_regression", "random_forest"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.tasks < 1:
        raise ValueError("--tasks must be at least 1")
    if args.ece_bins < 1:
        raise ValueError("--ece-bins must be at least 1")

    if args.worker_model:
        print(json.dumps(_run_worker(args)))
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required")

    results = [
        _run_isolated_worker(model_name="pfn", args=args),
        _run_isolated_worker(model_name="logistic_regression", args=args),
        _run_isolated_worker(model_name="random_forest", args=args),
    ]

    print(f"held-out tasks: {args.tasks}")
    print(f"ECE bins: {args.ece_bins}")
    print()
    for result in results:
        print(result["model"])
        print(
            f"  accuracy {result['accuracy']:.3f} | "
            f"log loss {result['log_loss']:.4f} | "
            f"Brier {result['brier']:.4f} | "
            f"ECE {result['ece']:.4f}"
        )
        print(
            f"  adaptation {result['adaptation_ms_per_task']:.2f} ms/task | "
            f"inference {result['inference_ms_per_task']:.2f} ms/task"
        )
        print(
            f"  model size {result['model_size_mb']:.3f} MB | "
            f"parameters {_format_parameters(result['parameters'])} | "
            f"peak RSS {result['peak_rss_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
