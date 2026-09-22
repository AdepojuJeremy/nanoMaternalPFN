import torch

from nanomaternalpfn import NanoMaternalPFN
from nanomaternalpfn.evaluate import evaluate_model
from nanomaternalpfn.train import save_checkpoint


def test_evaluate_model_returns_valid_metrics():
    model = NanoMaternalPFN(
        d_model=32,
        n_heads=4,
        hidden_dim=64,
        n_layers=1,
    )

    loss, accuracy = evaluate_model(
        model,
        n_tasks=2,
        batch_size=1,
        seed_start=10_000,
        device_name="cpu",
    )

    assert loss > 0
    assert 0.0 <= accuracy <= 1.0


def test_checkpoint_round_trip(tmp_path):
    model = NanoMaternalPFN()
    path = tmp_path / "model.pt"

    save_checkpoint(model, path)

    restored = NanoMaternalPFN()
    restored.load_state_dict(torch.load(path, map_location="cpu"))

    for expected, actual in zip(
        model.parameters(),
        restored.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)
