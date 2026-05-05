"""Generate self-play games with NN-guided alpha-beta search.

Each move during play is chosen by running the search and then sampling
from the resulting visit distribution (with temperature for the first
`--temperature-moves` plies, greedy after that). The visit distribution
is stored on the GameRecord for later use as a policy target.

Usage:
    python scripts/gen_search_selfplay.py \
        --ckpt runs/random9x9_ln/model.pt --n 100 \
        --iterations 4 --batch 32 \
        --out data/games_search_gen1.pkl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from alphabetago.nn import PolicyOwnershipNet
from alphabetago.selfplay import (
    play_search_games_multiproc,
    play_search_games_serial,
    save_games,
)


def load_model(ckpt_path: Path, device: torch.device) -> PolicyOwnershipNet:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = PolicyOwnershipNet(
        board_size=cfg["board_size"],
        in_planes=cfg["in_planes"],
        channels=cfg["channels"],
        n_blocks=cfg["n_blocks"],
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=4,
                        help="Search iterations per move.")
    parser.add_argument("--batch", type=int, default=32,
                        help="Leaves per search iteration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Run self-play across this many processes (>=2 for multi-proc)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Probe model config / size by loading once on the main process.
    probe = load_model(args.ckpt, device)
    size = probe.board_size
    print(
        f"Model: size={size} channels={probe.channels} blocks={probe.n_blocks} "
        f"params={probe.parameter_count():,}"
    )
    del probe
    print(f"Device: {device}")
    print(
        f"Search per move: {args.iterations} iterations x {args.batch} leaves "
        f"= {args.iterations * args.batch} NN evals (plus the root)."
    )
    print(
        f"Temperature: {args.temperature} for first {args.temperature_moves} plies, "
        "then greedy."
    )
    print(f"Workers: {args.workers}")
    print(f"Generating {args.n} games...")

    t0 = time.time()
    if args.workers > 1:
        games = play_search_games_multiproc(
            n_games=args.n,
            n_workers=args.workers,
            ckpt_path=args.ckpt,
            size=size,
            n_iterations=args.iterations,
            leaves_per_batch=args.batch,
            base_seed=args.seed,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
        )
    else:
        model = load_model(args.ckpt, device)
        games = play_search_games_serial(
            n_games=args.n,
            model=model,
            device=device,
            size=size,
            n_iterations=args.iterations,
            leaves_per_batch=args.batch,
            base_seed=args.seed,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
            progress_every=max(1, args.n // 20),
        )
    t1 = time.time()
    print(f"Done in {t1 - t0:.1f}s ({(t1 - t0) / args.n:.1f} s/game).")

    save_games(games, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved {len(games)} games to {args.out} ({size_mb:.1f} MB).")

    n_b = sum(1 for g in games if g.winner == 1)
    n_w = sum(1 for g in games if g.winner == 2)
    n_jigo = len(games) - n_b - n_w
    avg_moves = sum(g.n_moves for g in games) / len(games)
    avg_score = sum(g.final_score for g in games) / len(games)
    print(
        f"Stats: avg_moves={avg_moves:.1f}, avg_score(B)={avg_score:+.2f}, "
        f"B/W/jigo wins = {n_b}/{n_w}/{n_jigo}"
    )


if __name__ == "__main__":
    main()
