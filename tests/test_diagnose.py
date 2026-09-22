import torch

from nanomaternalpfn import NanoMaternalPFN
from nanomaternalpfn.diagnose import evaluate_by_family


def test_family_diagnostics_return_valid_metrics(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(NanoMaternalPFN().state_dict(), checkpoint)

    results = evaluate_by_family(
        checkpoint=str(checkpoint),
        n_tasks=6,
        seed_start=300_000,
        rf_trees=5,
        device_name="cpu",
    )

    assert results

    for result in results:
        assert result.tasks >= 1
        assert result.loss > 0
        assert 0.0 <= result.accuracy <= 1.0
