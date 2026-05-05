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


def play_search_games_vectorized(
    n_games: int,
    n_concurrent: int,
    model,
    device,
    size: int = 9,
    n_iterations: int = 4,
    leaves_per_batch: int = 32,
    base_seed: int = 0,
    max_moves: int = 2000,
    temperature_moves: int = 20,
    temperature: float = 1.0,
) -> list[GameRecord]:
    """Run `n_games` search self-play games, executing `n_concurrent` of them
    side-by-side in this process so that all of their NN evaluations get
    folded into a single batched forward pass per search iteration.

    Within each batch of `n_concurrent` games, all alive games take a move
    in lock-step: every "tick" runs (1) one root-expansion forward pass over
    all the games' fresh roots; (2) `n_iterations` rounds of leaf collection
    + a single batched forward pass across leaves from every alive game; (3)
    each game picks its move (eye-aware, temperature schedule, alpha-beta
    best non-eye thereafter) and plays.
    """
    import numpy as np

    from alphabetago.search import (
        Node,
        best_non_eye_move,
        collect_unexpanded_leaves,
        expand_nodes,
        filter_visits_no_eyes,
        sample_from_visit_policy,
        search_visit_policy,
    )

    all_records: list[GameRecord] = []
    seed_cursor = base_seed
    games_left = n_games
    while games_left > 0:
        m = min(n_concurrent, games_left)
        rngs = [np.random.default_rng(seed_cursor + i) for i in range(m)]
        boards = [Board(size=size) for _ in range(m)]
        move_lists: list[list[int]] = [[] for _ in range(m)]
        pol_lists: list[list[dict[int, float]]] = [[] for _ in range(m)]
        alive = [True] * m

        while any(alive):
            # Build (or refresh) a root for each alive game.
            roots: list[Node | None] = [None] * m
            fresh: list[Node] = []
            for i in range(m):
                if not alive[i]:
                    continue
                r = Node.from_board(boards[i])
                roots[i] = r
                if not r.is_terminal:
                    fresh.append(r)
            if fresh:
                expand_nodes(fresh, model, device)

            # n_iterations of batched leaf expansion.
            for _ in range(n_iterations):
                pooled: list[Node] = []
                for i in range(m):
                    if not alive[i]:
                        continue
                    r = roots[i]
                    if r is None or r.is_terminal:
                        continue
                    pooled.extend(collect_unexpanded_leaves(r, leaves_per_batch))
                if not pooled:
                    break
                expand_nodes(pooled, model, device)

            # Each alive game picks and plays a move.
            for i in range(m):
                if not alive[i]:
                    continue
                r = roots[i]
                board = boards[i]
                if r is None or board.is_game_over:
                    alive[i] = False
                    continue
                side = board.to_play
                target_move, _ = best_non_eye_move(r, side)
                if target_move is None:
                    target_move = PASS
                if len(move_lists[i]) < temperature_moves:
                    visits = search_visit_policy(r)
                    if visits:
                        visits = filter_visits_no_eyes(visits, board, side)
                        played = sample_from_visit_policy(visits, temperature, rngs[i])
                    else:
                        played = PASS
                else:
                    played = target_move
                pol_lists[i].append({target_move: 1.0})
                board.play(played)
                move_lists[i].append(played)
                if board.is_game_over or len(move_lists[i]) >= max_moves:
                    alive[i] = False

        for i in range(m):
            all_records.append(
                GameRecord(
                    size=size,
                    moves=move_lists[i],
                    final_score=boards[i].tromp_taylor_score(),
                    final_ownership=boards[i].ownership(),
                    search_policies=pol_lists[i],
                )
            )

        seed_cursor += m
        games_left -= m

    return all_records


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


def _selfplay_worker(packed):
    """Top-level worker for multi-process self-play. Each invocation loads
    its own model from `ckpt_path` (CUDA contexts are per-process)."""
    import torch as _torch

    from alphabetago.training import load_model as _load_model

    (
        ckpt_str, n_local, size, n_iters, batch, seed, max_moves,
        temp_moves, temp,
    ) = packed
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = _load_model(Path(ckpt_str), device)
    return play_search_games_serial(
        n_games=n_local,
        model=model,
        device=device,
        size=size,
        n_iterations=n_iters,
        leaves_per_batch=batch,
        base_seed=seed,
        max_moves=max_moves,
        temperature_moves=temp_moves,
        temperature=temp,
        progress_every=0,
    )


def play_search_games_multiproc(
    n_games: int,
    n_workers: int,
    ckpt_path,
    size: int = 9,
    n_iterations: int = 4,
    leaves_per_batch: int = 32,
    base_seed: int = 0,
    max_moves: int = 2000,
    temperature_moves: int = 20,
    temperature: float = 1.0,
) -> list[GameRecord]:
    """Multi-process self-play. Spawns up to `n_workers` worker processes,
    each loading the model from `ckpt_path` independently and running serial
    self-play on its share of games. Bypasses the GIL (each worker is its
    own Python interpreter), at the cost of (a) per-worker model load and
    CUDA-context init, (b) GPU calls from different processes serializing
    at the kernel scheduler. Empirically the sweet spot on this hardware is
    ~16-20 workers for small per-call batch sizes.

    Each game gets a distinct seed `base_seed + i`.
    """
    n_workers = max(1, min(n_workers, n_games))
    per = n_games // n_workers
    leftover = n_games - per * n_workers
    chunks = [per + (1 if i < leftover else 0) for i in range(n_workers)]

    args_list = []
    seed = base_seed
    for n_local in chunks:
        if n_local == 0:
            continue
        args_list.append((
            str(ckpt_path),
            n_local,
            size,
            n_iterations,
            leaves_per_batch,
            seed,
            max_moves,
            temperature_moves,
            temperature,
        ))
        seed += n_local

    ctx = mp.get_context("spawn")
    n_active = len(args_list)
    with ctx.Pool(processes=n_active) as pool:
        results = pool.map(_selfplay_worker, args_list)

    games: list[GameRecord] = []
    for r in results:
        games.extend(r)
    return games


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
