"""Minimal training loop for nanoMaternalPFN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import NanoMaternalPFN
from .synthetic import generate_task


def choose_device(requested: str = "auto") -> torch.device:
    """Select CPU, CUDA, or Apple Metal (MPS)."""
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(
    *,
    batch_size: int,
    seed_start: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of independent synthetic tasks."""
    tasks = [generate_task(seed=seed_start + i) for i in range(batch_size)]

    x_context = torch.from_numpy(
        np.stack([task.X_context for task in tasks])
    ).to(device)
    y_context = torch.from_numpy(
        np.stack([task.y_context for task in tasks])
    ).to(device)
    x_query = torch.from_numpy(
        np.stack([task.X_query for task in tasks])
    ).to(device)
    y_query = torch.from_numpy(
        np.stack([task.y_query for task in tasks])
    ).to(device)

    return x_context, y_context, x_query, y_query


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    """Run one optimization step and return loss and query accuracy."""
    x_context, y_context, x_query, y_query = batch

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(x_context, y_context, x_query)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y_query.reshape(-1),
    )

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y_query).float().mean()

    return float(loss.detach().cpu()), float(accuracy.detach().cpu())


def save_checkpoint(model: NanoMaternalPFN, path: str | Path) -> None:
    """Save model weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train(
    *,
    steps: int = 100,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 0,
    device_name: str = "auto",
) -> NanoMaternalPFN:
    """Train a small prototype on freshly generated synthetic tasks."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    torch.manual_seed(seed)
    device = choose_device(device_name)

    model = NanoMaternalPFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"device: {device}")
    print(f"steps: {steps}")
    print(f"batch size: {batch_size}")

    for step in range(1, steps + 1):
        seed_start = seed + (step - 1) * batch_size
        batch = make_batch(
            batch_size=batch_size,
            seed_start=seed_start,
            device=device,
        )
        loss, accuracy = train_step(model, optimizer, batch)

        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"step {step:4d} | "
                f"loss {loss:.4f} | "
                f"query accuracy {accuracy:.3f}"
            )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train nanoMaternalPFN on synthetic tasks."
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    args = parser.parse_args()

    model = train(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
    )

    if args.save:
        save_checkpoint(model, args.save)
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
