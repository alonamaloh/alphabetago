"""Re-run the closed-loop *training* schedule with a K-window replay buffer
over an existing pre-recorded sequence of self-play game pickles. No new
self-play and no per-gen match eval — just a training-only ablation that
isolates the effect of the replay buffer.

For each generation N from 1 to --n-gens:
  - Load games from gen max(1, N-K+1) through gen N (inclusive).
  - Fine-tune from gen N-1's checkpoint (or --init-ckpt for N=1) on the
    combined buffer for --epochs epochs.
  - Save the resulting model as gen N.

Pair this with `scripts/match_eval.py` afterwards to compare the resulting
checkpoint against the K=1 baseline at the same generation count.

Usage:
    python scripts/replay_loop.py --init-ckpt runs/keep/gen0_random.pt \
        --data-dir data/loop --out-dir runs/replay_k10 \
        --n-gens 42 --k 10 --epochs 4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from alphabetago.selfplay import load_games
from alphabetago.training import fine_tune, load_model, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-ckpt", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing genNNN.pkl files."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-gens", type=int, default=42)
    parser.add_argument("--k", type=int, default=10, help="Replay buffer window.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.out_dir / "log.csv"

    fieldnames = [
        "gen", "buffer_first", "buffer_last", "n_positions", "wall_sec",
        "tr_pol", "tr_own", "tr_top1",
        "val_pol", "val_own", "val_top1",
    ]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Init: {args.init_ckpt}")
    print(f"K (replay window): {args.k}, n_gens: {args.n_gens}, epochs: {args.epochs}")
    print(f"Device: {device}", flush=True)

    prev_ckpt = args.init_ckpt
    for gen in range(1, args.n_gens + 1):
        gen_start = time.time()
        gen_dir = args.out_dir / f"gen{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = gen_dir / "model.pt"
        if ckpt_path.exists():
            print(f"[gen {gen}] checkpoint exists, skipping (resume).", flush=True)
            prev_ckpt = ckpt_path
            continue

        first = max(1, gen - args.k + 1)
        last = gen
        games: list = []
        for n in range(first, last + 1):
            pkl = args.data_dir / f"gen{n:03d}.pkl"
            if not pkl.exists():
                raise FileNotFoundError(f"Missing data file: {pkl}")
            games.extend(load_games(pkl))
        n_positions = sum(len(g.moves) for g in games)

        model = load_model(prev_ckpt, device)
        tr, va = fine_tune(
            model, games, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            workers=args.workers,
            seed=gen,
        )
        save_model(model, ckpt_path)
        del model
        torch.cuda.empty_cache()

        wall = time.time() - gen_start
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "gen": gen,
                "buffer_first": first,
                "buffer_last": last,
                "n_positions": n_positions,
                "wall_sec": f"{wall:.1f}",
                "tr_pol": f"{tr['policy_loss']:.4f}",
                "tr_own": f"{tr['own_mse']:.4f}",
                "tr_top1": f"{tr['top1']:.4f}",
                "val_pol": f"{va['policy_loss']:.4f}",
                "val_own": f"{va['own_mse']:.4f}",
                "val_top1": f"{va['top1']:.4f}",
            })

        print(
            f"[gen {gen}] buffer=gen{first:03d}-gen{last:03d}  "
            f"{n_positions:>6d} pos  {wall:5.1f}s  "
            f"val_pol={va['policy_loss']:.3f} val_top1={va['top1']:.3f} "
            f"val_own={va['own_mse']:.3f}",
            flush=True,
        )
        prev_ckpt = ckpt_path


if __name__ == "__main__":
    main()
