"""EdgeBench V0 for nanoMaternalPFN deployment variants."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .benchmark import brier_score, expected_calibration_error
from .baselines import _binary_log_loss
from .model import NanoMaternalPFN
from .synthetic import generate_task


VARIANTS = {
    "cpu-fp32": ("cpu", "fp32"),
    "mps-fp32": ("mps", "fp32"),
    "mps-fp16": ("mps", "fp16"),
    "cpu-int8": ("cpu", "int8"),
}


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _serialized_state_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def _state_tensor_bytes(model: nn.Module) -> int:
    return _tensor_bytes(model.state_dict())


def _dynamic_int8(model: nn.Module) -> nn.Module:
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic

    return quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def prepare_variant(
    checkpoint: str,
    variant: str,
) -> tuple[nn.Module, torch.device]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    device_name, precision = VARIANTS[variant]

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    model = NanoMaternalPFN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    if precision == "int8":
        model = _dynamic_int8(model)
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        if precision == "fp16":
            model = model.half()
        model = model.to(device)

    return model, device


def benchmark_variant(
    *,
    checkpoint: str,
    variant: str,
    n_tasks: int = 500,
    seed_start: int = 100_000,
    ece_bins: int = 10,
    warmup_runs: int = 5,
) -> dict[str, Any]:
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")

    model, device = prepare_variant(checkpoint, variant)

    warmup_task = generate_task(seed=seed_start)
    x_context = torch.from_numpy(warmup_task.X_context).unsqueeze(0).to(device)
    y_context = torch.from_numpy(warmup_task.y_context).unsqueeze(0).to(device)
    x_query = torch.from_numpy(warmup_task.X_query).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            model(x_context, y_context, x_query)
        _synchronize(device)

    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    latency_seconds = 0.0

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
            latency_seconds += time.perf_counter() - start

            probabilities = (
                torch.softmax(logits, dim=-1)[0, :, 1]
                .float()
                .cpu()
                .numpy()
            )
            all_y.append(task.y_query)
            all_p.append(probabilities)

    y = np.concatenate(all_y)
    p1 = np.concatenate(all_p)
    predictions = (p1 >= 0.5).astype(np.int64)

    return {
        "variant": variant,
        "device": device.type,
        "tasks": n_tasks,
        "accuracy": float((predictions == y).mean()),
        "log_loss": _binary_log_loss(y, p1),
        "brier": brier_score(y, p1),
        "ece": expected_calibration_error(y, p1, n_bins=ece_bins),
        "latency_ms_per_task": 1000.0 * latency_seconds / n_tasks,
        "serialized_size_mb": _serialized_state_bytes(model)
        / (1024.0 * 1024.0),
        "state_tensor_mb": _state_tensor_bytes(model)
        / (1024.0 * 1024.0),
        "peak_rss_mb": _peak_rss_mb(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _run_worker(
    *,
    checkpoint: str,
    variant: str,
    tasks: int,
    seed_start: int,
    ece_bins: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "nanomaternalpfn.edgebench",
        "--worker-variant",
        variant,
        "--checkpoint",
        checkpoint,
        "--tasks",
        str(tasks),
        "--seed-start",
        str(seed_start),
        "--ece-bins",
        str(ece_bins),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return json.loads(completed.stdout)


def _available_variants() -> list[str]:
    variants = ["cpu-fp32", "cpu-int8"]
    if torch.backends.mps.is_available():
        variants.extend(["mps-fp32", "mps-fp16"])
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark nanoMaternalPFN edge deployment variants."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--worker-variant",
        choices=tuple(VARIANTS),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.worker_variant:
        result = benchmark_variant(
            checkpoint=args.checkpoint,
            variant=args.worker_variant,
            n_tasks=args.tasks,
            seed_start=args.seed_start,
            ece_bins=args.ece_bins,
        )
        print(json.dumps(result))
        return

    results = [
        _run_worker(
            checkpoint=args.checkpoint,
            variant=variant,
            tasks=args.tasks,
            seed_start=args.seed_start,
            ece_bins=args.ece_bins,
        )
        for variant in _available_variants()
    ]

    reference = next(
        result for result in results if result["variant"] == "cpu-fp32"
    )

    print(f"held-out tasks: {args.tasks}")
    print(f"machine: {reference['machine']}")
    print(f"torch: {reference['torch_version']}")
    print()

    for result in results:
        accuracy_delta = result["accuracy"] - reference["accuracy"]
        ece_delta = result["ece"] - reference["ece"]

        print(result["variant"])
        print(
            f"  accuracy {result['accuracy']:.3f} "
            f"({accuracy_delta:+.3f}) | "
            f"log loss {result['log_loss']:.4f} | "
            f"Brier {result['brier']:.4f} | "
            f"ECE {result['ece']:.4f} ({ece_delta:+.4f})"
        )
        print(
            f"  latency {result['latency_ms_per_task']:.2f} ms/task | "
            f"serialized {result['serialized_size_mb']:.3f} MB | "
            f"tensor state {result['state_tensor_mb']:.3f} MB | "
            f"peak RSS {result['peak_rss_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()
