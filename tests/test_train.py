import torch

from nanomaternalpfn import NanoMaternalPFN
from nanomaternalpfn.train import make_batch, train_step


def test_one_training_step_updates_parameters():
    device = torch.device("cpu")
    model = NanoMaternalPFN(
        d_model=32,
        n_heads=4,
        hidden_dim=64,
        n_layers=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = make_batch(
        batch_size=2,
        seed_start=0,
        device=device,
    )

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    loss, accuracy = train_step(model, optimizer, batch)

    after = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= accuracy <= 1.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, after, strict=True)
    )
