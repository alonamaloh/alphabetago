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
    # If the game was generated via search self-play, this is the per-move
    # search visit distribution (one dict per move actually played). For
    # random self-play this stays None; training falls back to a one-hot
    # target on the played move.
    search_policies: list[dict[int, float]] | None = None

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


def play_search_game(
    model,
    device,
    size: int = 9,
    n_iterations: int = 4,
    leaves_per_batch: int = 64,
    seed: int | None = None,
    max_moves: int = 2000,
    temperature_moves: int = 20,
    temperature: float = 1.0,
) -> GameRecord:
    """Self-play a single game where each move is chosen by NN-guided search.

    At every position:
      - Run the search to populate the tree.
      - Compute the alpha-beta best non-eye move at the root; this is recorded
        as a one-hot policy target (the search's recommendation).
      - For move selection, sample from the (eye-filtered) visit distribution
        with `temperature` for the first `temperature_moves` plies, and
        greedily play the alpha-beta best after that.

    Eye-avoidance mirrors what `random_move` does, otherwise search-driven
    games never pass and run forever.
    """
    import numpy as np

    from alphabetago.search import (
        best_non_eye_move,
        filter_visits_no_eyes,
        sample_from_visit_policy,
        search,
        search_visit_policy,
    )

    rng = np.random.default_rng(seed)
    board = Board(size=size)
    moves: list[int] = []
    visit_policies: list[dict[int, float]] = []

    while not board.is_game_over and len(moves) < max_moves:
        root, _, _ = search(
            board, model, device,
            n_iterations=n_iterations,
            leaves_per_batch=leaves_per_batch,
        )
        side = board.to_play
        target_move, _ = best_non_eye_move(root, side)
        if target_move is None:
            target_move = PASS

        if len(moves) < temperature_moves:
            visits = search_visit_policy(root)
            if visits:
                visits = filter_visits_no_eyes(visits, board, side)
                played = sample_from_visit_policy(visits, temperature, rng)
            else:
                played = PASS
        else:
            played = target_move

        visit_policies.append({target_move: 1.0})
        board.play(played)
        moves.append(played)

    return GameRecord(
        size=size,
        moves=moves,
        final_score=board.tromp_taylor_score(),
        final_ownership=board.ownership(),
        search_policies=visit_policies,
    )


def play_match_game(
    model_black,
    model_white,
    device,
    size: int = 9,
    n_iterations: int = 4,
    leaves_per_batch: int = 32,
    seed: int | None = None,
    max_moves: int = 2000,
    temperature_moves: int = 20,
    temperature: float = 1.0,
) -> GameRecord:
    """Play one game between two models. `model_black` plays Black,
    `model_white` plays White. Move selection mirrors `play_search_game`
    (eye-avoidance, sample-from-visits with temperature in the opening,
    greedy on the alpha-beta best move afterwards).
    """
    import numpy as np

    from alphabetago.search import (
        best_non_eye_move,
        filter_visits_no_eyes,
        sample_from_visit_policy,
        search,
        search_visit_policy,
    )

    rng = np.random.default_rng(seed)
    board = Board(size=size)
    moves: list[int] = []

    while not board.is_game_over and len(moves) < max_moves:
        side = board.to_play
        model = model_black if side == BLACK else model_white
        root, _, _ = search(
            board, model, device,
            n_iterations=n_iterations,
            leaves_per_batch=leaves_per_batch,
        )
        target_move, _ = best_non_eye_move(root, side)
        if target_move is None:
            target_move = PASS

        if len(moves) < temperature_moves:
            visits = search_visit_policy(root)
            if visits:
                visits = filter_visits_no_eyes(visits, board, side)
                played = sample_from_visit_policy(visits, temperature, rng)
            else:
                played = PASS
        else:
            played = target_move

        board.play(played)
        moves.append(played)

    return GameRecord(
        size=size,
        moves=moves,
        final_score=board.tromp_taylor_score(),
        final_ownership=board.ownership(),
    )


def play_search_games_serial(
    n_games: int,
    model,
    device,
    size: int = 9,
    n_iterations: int = 4,
    leaves_per_batch: int = 64,
    base_seed: int = 0,
    max_moves: int = 2000,
    temperature_moves: int = 30,
    temperature: float = 1.0,
    progress_every: int = 0,
) -> list[GameRecord]:
    """Generate `n_games` search self-play games sequentially (single GPU)."""
    import time

    games: list[GameRecord] = []
    t0 = time.time()
    for i in range(n_games):
        rec = play_search_game(
            model=model,
            device=device,
            size=size,
            n_iterations=n_iterations,
            leaves_per_batch=leaves_per_batch,
            seed=base_seed + i,
            max_moves=max_moves,
            temperature_moves=temperature_moves,
            temperature=temperature,
        )
        games.append(rec)
        if progress_every and (i + 1) % progress_every == 0:
            dt = time.time() - t0
            avg = dt / (i + 1)
            eta = avg * (n_games - i - 1)
            print(
                f"  [{i + 1}/{n_games}] {avg:.1f}s/game, ETA {eta / 60:.1f} min",
                flush=True,
            )
    return games


def save_games(games: list[GameRecord], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_games(path: str | Path) -> list[GameRecord]:
    with open(path, "rb") as f:
        return pickle.load(f)
