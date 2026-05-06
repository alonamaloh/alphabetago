"""Convert a directory of KataGo training-game SGFs into one GameRecord pickle.

Walks `--sgf-dir` (recursively), parses every .sgf with our minimal SGF
parser, replays each game through our Tromp-Taylor engine (including any
SGF setup stones via Board.setup), and pickles the resulting list of
GameRecords. Games that fail to replay (illegal moves under our rules)
are dropped and counted.

Usage:
    python scripts/import_katago_sgfs.py \
        --sgf-dir data/katago_sgfs/9x9 --size 9 \
        --workers 24 --out data/katago_sgfs/games_9x9.pkl
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from alphabetago.selfplay import save_games
from alphabetago.sgf import load_sgf_file, sgf_to_game_record


def _process(args):
    path, size = args
    try:
        text = load_sgf_file(path)
        rec = sgf_to_game_record(text, max_size=size)
        return rec
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    paths = list(args.sgf_dir.rglob("*.sgf"))
    print(f"Found {len(paths)} SGF files under {args.sgf_dir}.")
    if not paths:
        return

    n_workers = args.workers or mp.cpu_count()
    chunksize = max(1, len(paths) // (n_workers * 8))
    work = [(p, args.size) for p in paths]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process, work, chunksize=chunksize)
    dt = time.time() - t0

    games = [r for r in results if r is not None]
    n_drops = len(results) - len(games)
    print(
        f"Parsed in {dt:.1f}s: kept {len(games)} games, dropped {n_drops} "
        f"({n_drops / max(1, len(results)) * 100:.1f}%)"
    )

    save_games(games, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved {len(games)} games to {args.out} ({size_mb:.1f} MB).")

    if games:
        avg_moves = sum(len(g.moves) for g in games) / len(games)
        avg_score = sum(g.final_score for g in games) / len(games)
        n_b = sum(1 for g in games if g.final_score > 0)
        n_w = sum(1 for g in games if g.final_score < 0)
        n_jigo = sum(1 for g in games if g.final_score == 0)
        print(
            f"Stats: avg_moves={avg_moves:.1f}, avg_score(B)={avg_score:+.2f}, "
            f"B/W/jigo (by recomputed area score) = {n_b}/{n_w}/{n_jigo}"
        )


if __name__ == "__main__":
    main()
