import torch

from alphabetago.training import _apply_d4, apply_d4_sample


def test_d4_identity_is_identity():
    t = torch.arange(16).view(4, 4)
    out = _apply_d4(t, sym=0)
    assert torch.equal(out, t)


def test_d4_eight_distinct_transformations():
    # On a 3x3 with all-distinct values, the 8 D4 elements give 8 distinct outputs.
    t = torch.arange(9).view(3, 3)
    seen = set()
    for sym in range(8):
        out = _apply_d4(t, sym)
        seen.add(tuple(out.flatten().tolist()))
    assert len(seen) == 8


def test_apply_d4_sample_consistency():
    """A stone marked at (r, c) in features should land at the same (r', c') in
    the transformed ownership map, AND the policy target one-hot at that move
    should land at the matching index in the transformed policy."""
    n = 5
    pol_dim = n * n + 1
    for sym in range(8):
        features = torch.zeros(2, n, n)
        features[0, 1, 2] = 1.0  # marker plane has stone at (1, 2)

        ownership = torch.zeros(n, n)
        ownership[1, 2] = 1.0  # ownership marker at the same point

        policy = torch.zeros(pol_dim)
        policy[1 * n + 2] = 1.0  # one-hot policy on the same move

        f_out, p_out, o_out = apply_d4_sample(features, policy, ownership, sym, n)

        # Find the new (r, c).
        nz = (f_out[0] == 1.0).nonzero(as_tuple=False)
        assert nz.shape[0] == 1
        r2, c2 = nz[0].tolist()

        # Ownership has the marker in the same new spot.
        assert o_out[r2, c2] == 1.0
        assert o_out.sum() == 1.0

        # Policy: one-hot at flat-index (r2, c2).
        flat = r2 * n + c2
        assert p_out[flat] == 1.0
        assert p_out.sum() == 1.0

        # PASS slot stays at index n*n.
        assert p_out[n * n] == 0.0


def test_apply_d4_preserves_pass():
    n = 5
    pol_dim = n * n + 1
    features = torch.zeros(2, n, n)
    ownership = torch.zeros(n, n)
    policy = torch.zeros(pol_dim)
    policy[n * n] = 1.0  # PASS one-hot

    for sym in range(8):
        _, p_out, _ = apply_d4_sample(features, policy, ownership, sym, n)
        assert p_out[n * n] == 1.0
        assert p_out[: n * n].sum() == 0.0
