"""Random self-play loop.

Plays complete games with both sides choosing uniformly at random from the
legal moves that are not in their own eye. Returns a `GameRecord` containing
the move sequence, final Tromp-Taylor score, and per-point final ownership.
This module exists to validate the rules end-to-end before any neural net is
introduced; it is not a strong player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from alphabetago.board import BLACK, EMPTY, PASS, WHITE, Board


@dataclass
class GameRecord:
    size: int
    komi: float
    moves: list[int]
    final_score: float
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
    komi: float = 7.0,
    seed: int | None = None,
    max_moves: int = 2000,
) -> GameRecord:
    """Play one self-play game with both sides choosing moves at random."""
    rng = random.Random(seed)
    board = Board(size=size, komi=komi)
    moves: list[int] = []
    while not board.is_game_over and len(moves) < max_moves:
        m = random_move(board, rng)
        board.play(m)
        moves.append(m)
    return GameRecord(
        size=size,
        komi=komi,
        moves=moves,
        final_score=board.tromp_taylor_score(),
        final_ownership=board.ownership(),
    )
