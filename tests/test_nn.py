import torch

from alphabetago.features import N_PLANES
from alphabetago.nn import PolicyOwnershipNet


def test_forward_shapes():
    model = PolicyOwnershipNet(board_size=9, channels=16, n_blocks=2)
    x = torch.randn(4, N_PLANES, 9, 9)
    policy_logits, ownership = model(x)
    assert policy_logits.shape == (4, 9 * 9 + 1)
    assert ownership.shape == (4, 9, 9)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(ownership).all()
    # Ownership is bounded by tanh.
    assert (ownership.abs() <= 1.0).all()


def test_parameter_count_reasonable():
    model = PolicyOwnershipNet(board_size=9, channels=32, n_blocks=3)
    n = model.parameter_count()
    assert 10_000 < n < 200_000  # small net


def test_backward_runs():
    model = PolicyOwnershipNet(board_size=9, channels=16, n_blocks=2)
    x = torch.randn(2, N_PLANES, 9, 9)
    policy_logits, ownership = model(x)
    target_policy = torch.tensor([0, 1])
    target_own = torch.zeros(2, 9, 9)
    loss = (
        torch.nn.functional.cross_entropy(policy_logits, target_policy)
        + torch.nn.functional.mse_loss(ownership, target_own)
    )
    loss.backward()
    # Some parameter received a non-zero gradient.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
