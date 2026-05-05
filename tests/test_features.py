import numpy as np

from alphabetago.board import BLACK, EMPTY, PASS, WHITE, Board
from alphabetago.features import (
    LIB_BUCKETS,
    N_PLANES,
    P_LAST_MOVE,
    P_ONES,
    P_OPP_LIBS,
    P_OPP_SIZE,
    P_OPP_STONES,
    P_OWN_STONES,
    SIZE_BUCKETS,
    featurize,
    ownership_target,
    policy_target,
)


def test_planes_count():
    assert N_PLANES == 2 + 2 * LIB_BUCKETS + 2 * SIZE_BUCKETS + 1 + 1


def test_empty_board_features():
    b = Board(size=9)
    f = featurize(b)
    assert f.shape == (N_PLANES, 9, 9)
    assert f.dtype == np.float32
    # No stones, no last move; only the constant ones plane is non-zero.
    assert f[P_OWN_STONES].sum() == 0
    assert f[P_OPP_STONES].sum() == 0
    assert f[P_LAST_MOVE].sum() == 0
    assert (f[P_ONES] == 1.0).all()


def test_stone_planes_oriented_to_player():
    b = Board(size=9)
    p = b.point(4, 4)
    b.play(p)  # B plays
    # Now W is to play. From W's perspective, the stone at (4,4) is opponent.
    f = featurize(b)
    assert f[P_OPP_STONES, 4, 4] == 1.0
    assert f[P_OWN_STONES, 4, 4] == 0.0
    # Black stone has 4 liberties → bucket index 3 (since cap is 4 → bucket 4-1=3).
    assert f[P_OPP_LIBS + 3, 4, 4] == 1.0
    # Chain size = 1 → bucket 0.
    assert f[P_OPP_SIZE + 0, 4, 4] == 1.0
    # Last move plane.
    assert f[P_LAST_MOVE, 4, 4] == 1.0


def test_liberty_buckets_capped():
    b = Board(size=9)
    # A center stone has exactly 4 liberties.
    b.play(b.point(4, 4))
    f = featurize(b)  # W to play; (4,4) is opp.
    # bucket index for libs=4 is min(4, 4)-1 = 3.
    assert f[P_OPP_LIBS + 3, 4, 4] == 1.0
    # No other liberty buckets active.
    for k in range(LIB_BUCKETS - 1):
        assert f[P_OPP_LIBS + k, 4, 4] == 0.0


def test_chain_size_bucket_after_merge():
    b = Board(size=9)
    # Build a 3-stone black chain at (4,4)-(4,5)-(4,6).
    b.play(b.point(4, 4))
    b.play(b.point(0, 0))  # W somewhere
    b.play(b.point(4, 6))
    b.play(b.point(0, 1))
    b.play(b.point(4, 5))  # connects
    # W to play; the 3-stone chain is opponent for W.
    f = featurize(b)
    # Chain size 3 → bucket 2.
    for c in (4, 5, 6):
        assert f[P_OPP_SIZE + 2, 4, c] == 1.0


def test_last_move_zero_after_pass():
    b = Board(size=9)
    b.play(b.point(4, 4))
    b.play(PASS)
    f = featurize(b)  # B to play (pass was W's).
    # After a pass, last_move is None → the last-move plane is all zeros.
    assert f[P_LAST_MOVE].sum() == 0


def test_policy_target_indices():
    b = Board(size=9)
    assert policy_target(PASS, b) == 81
    assert policy_target(b.point(0, 0), b) == 0
    assert policy_target(b.point(0, 8), b) == 8
    assert policy_target(b.point(8, 8), b) == 80


def test_ownership_target_orientation():
    b = Board(size=5)
    # Final ownership: a couple of mock entries (not from real play).
    own = {p: EMPTY for p in b.on_board_points()}
    own[b.point(0, 0)] = BLACK
    own[b.point(0, 1)] = WHITE
    # From black's perspective.
    t = ownership_target(own, BLACK, b)
    assert t[0, 0] == 1.0
    assert t[0, 1] == -1.0
    assert t[2, 2] == 0.0
    # From white's perspective the signs flip.
    t = ownership_target(own, WHITE, b)
    assert t[0, 0] == -1.0
    assert t[0, 1] == 1.0
    assert t[2, 2] == 0.0
