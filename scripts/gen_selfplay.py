"""Generate random-self-play games and pickle them to disk.

Usage:
    python scripts/gen_selfplay.py --n 10000 --size 9 \
        --workers 24 --out data/games_random_9x9.pkl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from alphabetago.selfplay import play_random_games_parallel, save_games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000, help="Number of games to play.")
    parser.add_argument("--size", type=int, default=9, help="Board size.")
    parser.add_argument("--workers", type=int, default=None, help="Worker process count.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--out", type=Path, required=True, help="Output pickle path.")
    args = parser.parse_args()

    print(
        f"Generating {args.n} games (size={args.size}) with "
        f"{args.workers or 'auto'} workers, base seed {args.seed}..."
    )
    t0 = time.time()
    games = play_random_games_parallel(
        n_games=args.n,
        size=args.size,
        base_seed=args.seed,
        n_workers=args.workers,
    )
    t1 = time.time()
    print(
        f"Generated {len(games)} games in {t1 - t0:.1f}s "
        f"({(t1 - t0) / len(games) * 1000:.1f} ms/game)."
    )

    save_games(games, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved to {args.out} ({size_mb:.1f} MB).")

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
