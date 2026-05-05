"""Turn `GameRecord`s into per-position training tensors.

Replays each game move-by-move, capturing at every step:
    - input features (oriented to the side to play)
    - flat policy target (the move actually played in the random game)
    - per-point ownership target (the eventual game ownership, oriented)

Featurization is parallelized across processes since this is the expensive
step — each game does ~120 featurize calls and we may have tens of thousands
of games.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from alphabetago.board import Board
from alphabetago.features import N_PLANES, featurize, ownership_target, policy_target
from alphabetago.selfplay import GameRecord


def _process_game(rec: GameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = rec.size
    n_pos = len(rec.moves)
    feats = np.zeros((n_pos, N_PLANES, n, n), dtype=np.int8)
    policies = np.zeros(n_pos, dtype=np.int32)
    ownerships = np.zeros((n_pos, n, n), dtype=np.int8)
    board = Board(size=n)
    for i, move in enumerate(rec.moves):
        feats[i] = featurize(board).astype(np.int8)
        policies[i] = policy_target(move, board)
        ownerships[i] = ownership_target(rec.final_ownership, board.to_play, board).astype(np.int8)
        board.play(move)
    return feats, policies, ownerships


def featurize_games(
    games: list[GameRecord], n_workers: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize all positions across all games.

    Returns:
        features: [total_positions, N_PLANES, size, size] int8
        policies: [total_positions] int32
        ownerships: [total_positions, size, size] int8
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, max(1, len(games)))
    chunksize = max(1, len(games) // (n_workers * 8))
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_game, games, chunksize=chunksize)
    feats_list, pol_list, own_list = zip(*results, strict=True)
    return (
        np.concatenate(feats_list),
        np.concatenate(pol_list),
        np.concatenate(own_list),
    )
