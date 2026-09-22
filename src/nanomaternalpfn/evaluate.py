"""Held-out evaluation for nanoMaternalPFN."""

from __future__ import annotations

import argparse

import torch

from .model import NanoMaternalPFN
from .train import choose_device, make_batch


@torch.no_grad()
def evaluate_model(
    model: NanoMaternalPFN,
    *,
    n_tasks: int = 100,
    batch_size: int = 10,
    seed_start: int = 100_000,
    device_name: str = "auto",
) -> tuple[float, float]:
    """Evaluate mean cross-entropy and query accuracy on unseen task seeds."""
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    device = choose_device(device_name)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for offset in range(0, n_tasks, batch_size):
        current_batch = min(batch_size, n_tasks - offset)
        batch = make_batch(
            batch_size=current_batch,
            seed_start=seed_start + offset,
            device=device,
        )
        x_context, y_context, x_query, y_query = batch

        logits = model(x_context, y_context, x_query)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y_query.reshape(-1),
            reduction="sum",
        )

        predictions = logits.argmax(dim=-1)
        total_loss += float(loss.cpu())
        total_correct += int((predictions == y_query).sum().cpu())
        total_examples += y_query.numel()

    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nanoMaternalPFN on unseen synthetic tasks."
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    args = parser.parse_args()

    model = NanoMaternalPFN()

    label = "untrained"
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
        label = "trained"

    loss, accuracy = evaluate_model(
        model,
        n_tasks=args.tasks,
        batch_size=args.batch_size,
        seed_start=args.seed_start,
        device_name=args.device,
    )

    print(f"model: {label}")
    print(f"held-out tasks: {args.tasks}")
    print(f"loss: {loss:.4f}")
    print(f"query accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
