import pytest
import torch

from nanomaternalpfn import NanoMaternalPFN, generate_task
from nanomaternalpfn.edgebench import _state_tensor_bytes


def test_state_tensor_bytes_matches_fp32_parameter_scale():
    model = NanoMaternalPFN()

    assert _state_tensor_bytes(model) > 1_000_000


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple MPS",
)
def test_fp16_model_forward_on_mps():
    device = torch.device("mps")
    model = NanoMaternalPFN().half().to(device).eval()
    task = generate_task(seed=123)

    x_context = torch.from_numpy(task.X_context).unsqueeze(0).to(device)
    y_context = torch.from_numpy(task.y_context).unsqueeze(0).to(device)
    x_query = torch.from_numpy(task.X_query).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_context, y_context, x_query)

    assert logits.dtype == torch.float16
    assert logits.shape == (1, 50, 2)
    assert torch.isfinite(logits).all()
