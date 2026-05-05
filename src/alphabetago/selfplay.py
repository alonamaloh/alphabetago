"""Random self-play loop.

Plays complete games with both sides choosing uniformly at random from the
legal moves that are not in their own eye. Returns a `GameRecord` containing
the move sequence, final Tromp-Taylor score, and per-point final ownership.
This module exists to validate the rules end-to-end before any neural net is
introduced; it is not a strong player.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

from alphabetago.board import BLACK, EMPTY, PASS, WHITE, Board


@dataclass
class GameRecord:
    size: int
    moves: list[int]
    final_score: int
    final_ownership: dict[int, int]

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    @property
    def n_passes(self) -> int:
        return sum(1 for m in self.moves if m == PASS)

    @property
    def winner(self) -> int:
        if self.final_score > 0:
            return BLACK
        if self.final_score < 0:
            return WHITE
        return EMPTY


def random_move(board: Board, rng: random.Random) -> int:
    """Pick a uniform random legal move, excluding the side-to-play's own eyes.

    Returns `PASS` when no such move exists.
    """
    color = board.to_play
    candidates: list[int] = []
    for p in board.on_board_points():
        if board.is_legal(p) and not board.is_eye_for(color, p):
            candidates.append(p)
    if not candidates:
        return PASS
    return rng.choice(candidates)


def play_random_game(
    size: int = 9,
    seed: int | None = None,
    max_moves: int = 2000,
) -> GameRecord:
    """Play one self-play game with both sides choosing moves at random."""
    rng = random.Random(seed)
    board = Board(size=size)
    moves: list[int] = []
    while not board.is_game_over and len(moves) < max_moves:
        m = random_move(board, rng)
        board.play(m)
        moves.append(m)
    return GameRecord(
        size=size,
        moves=moves,
        final_score=board.tromp_taylor_score(),
        final_ownership=board.ownership(),
    )


def _play_one(args: tuple[int, int, int]) -> GameRecord:
    size, seed, max_moves = args
    return play_random_game(size=size, seed=seed, max_moves=max_moves)


def play_random_games_parallel(
    n_games: int,
    size: int = 9,
    base_seed: int = 0,
    max_moves: int = 2000,
    n_workers: int | None = None,
) -> list[GameRecord]:
    """Generate `n_games` self-play games across a process pool.

    Each game gets a distinct seed (`base_seed + i`) so the run is reproducible
    given the same base seed and worker count.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, n_games)
    args_list = [(size, base_seed + i, max_moves) for i in range(n_games)]
    chunksize = max(1, n_games // (n_workers * 8))
    with mp.Pool(n_workers) as pool:
        return pool.map(_play_one, args_list, chunksize=chunksize)


def save_games(games: list[GameRecord], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_games(path: str | Path) -> list[GameRecord]:
    with open(path, "rb") as f:
        return pickle.load(f)
