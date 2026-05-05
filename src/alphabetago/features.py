"""Featurize a Board state into NN input planes, oriented to the side to play.

Plane layout (20 channels total):
   0          : own stones
   1          : opponent stones
   2..5       : own chain liberty count, one-hot bucketed at 1, 2, 3, >=4
   6..9       : opponent chain liberty count, one-hot bucketed at 1, 2, 3, >=4
  10..13      : own chain stone count, one-hot bucketed at 1, 2, 3, >=4
  14..17      : opponent chain stone count, one-hot bucketed at 1, 2, 3, >=4
  18          : last-move location (zero plane on a pass)
  19          : all-ones constant

The bucketed liberty / chain-size planes carry the chain stats per stone the
user wants the network to see.
"""

from __future__ import annotations

import numpy as np

from alphabetago.board import EMPTY, PASS, Board, opponent

LIB_BUCKETS = 4
SIZE_BUCKETS = 4
N_PLANES = 2 + 2 * LIB_BUCKETS + 2 * SIZE_BUCKETS + 1 + 1

# Plane index offsets.
P_OWN_STONES = 0
P_OPP_STONES = 1
P_OWN_LIBS = 2
P_OPP_LIBS = P_OWN_LIBS + LIB_BUCKETS
P_OWN_SIZE = P_OPP_LIBS + LIB_BUCKETS
P_OPP_SIZE = P_OWN_SIZE + SIZE_BUCKETS
P_LAST_MOVE = P_OPP_SIZE + SIZE_BUCKETS
P_ONES = P_LAST_MOVE + 1


def featurize(board: Board) -> np.ndarray:
    """Return a [N_PLANES, size, size] float32 array oriented to the side to play."""
    n = board.size
    me = board.to_play
    them = opponent(me)
    out = np.zeros((N_PLANES, n, n), dtype=np.float32)

    for r in range(n):
        for c in range(n):
            point = board.point(r, c)
            v = board.stone_at(point)
            if v == me:
                out[P_OWN_STONES, r, c] = 1.0
                lib_bucket = min(board.chain_liberties_at(point), LIB_BUCKETS) - 1
                out[P_OWN_LIBS + lib_bucket, r, c] = 1.0
                size_bucket = min(board.chain_size_at(point), SIZE_BUCKETS) - 1
                out[P_OWN_SIZE + size_bucket, r, c] = 1.0
            elif v == them:
                out[P_OPP_STONES, r, c] = 1.0
                lib_bucket = min(board.chain_liberties_at(point), LIB_BUCKETS) - 1
                out[P_OPP_LIBS + lib_bucket, r, c] = 1.0
                size_bucket = min(board.chain_size_at(point), SIZE_BUCKETS) - 1
                out[P_OPP_SIZE + size_bucket, r, c] = 1.0

    if board.last_move is not None:
        r, c = board.coord(board.last_move)
        out[P_LAST_MOVE, r, c] = 1.0

    out[P_ONES, :, :] = 1.0
    return out


def policy_target(move: int, board: Board) -> int:
    """Convert a move (mailbox point or PASS) to a flat policy index in [0, n*n].

    The pass move maps to index n*n (last position).
    """
    n = board.size
    if move == PASS:
        return n * n
    r, c = board.coord(move)
    return r * n + c


def ownership_target(
    final_ownership: dict[int, int], to_play: int, board: Board
) -> np.ndarray:
    """Per-point target in {-1, 0, +1} oriented to `to_play`.

    +1 if the side to play owns the point at the end of the game, -1 if the
    opponent does, and 0 if the point is contested (or has no determinable
    owner under Tromp-Taylor).
    """
    n = board.size
    out = np.zeros((n, n), dtype=np.float32)
    for r in range(n):
        for c in range(n):
            point = board.point(r, c)
            v = final_ownership[point]
            if v == to_play:
                out[r, c] = 1.0
            elif v == EMPTY:
                out[r, c] = 0.0
            else:
                out[r, c] = -1.0
    return out
