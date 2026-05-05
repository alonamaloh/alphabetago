"""Multi-generation search-play / fine-tune / eval autopilot loop.

For each generation N from 1 to --n-gens:
  1. Generate `games_per_gen` search self-play games with the gen N-1 model.
  2. Fine-tune from gen N-1 weights on those games for `epochs` epochs.
  3. Match-eval gen N vs gen N-1 over `eval_games` games (alternate sides).
  4. Append all metrics to a CSV log.

Idempotent: if `runs/.../genN/model.pt` already exists, that generation is
skipped, so the loop can be resumed after an interruption.

Usage:
    python scripts/run_loop.py --init-ckpt runs/random9x9_ln/model.pt \
        --n-gens 50 --out-dir runs/loop --data-dir data/loop
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import torch

from alphabetago.board import BLACK, WHITE
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.selfplay import (
    load_games,
    play_match_game,
    play_match_games_multiproc,
    play_search_games_multiproc,
    play_search_games_serial,
    save_games,
)
from alphabetago.training import fine_tune, load_model, save_model


def run_match(
    model_a: PolicyOwnershipNet,
    model_b: PolicyOwnershipNet,
    n_games: int,
    device: torch.device,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, float]:
    a_b = a_w = b_b = b_w = jigos = 0
    a_score_total = 0.0
    for i in range(n_games):
        a_is_black = (i % 2 == 0)
        if a_is_black:
            black, white = model_a, model_b
        else:
            black, white = model_b, model_a
        rec = play_match_game(
            model_black=black,
            model_white=white,
            device=device,
            size=model_a.board_size,
            n_iterations=args.search_iters,
            leaves_per_batch=args.search_batch,
            seed=seed_offset + i,
            temperature_moves=20,
            temperature=1.0,
        )
        a_score_total += rec.final_score if a_is_black else -rec.final_score
        if rec.winner == BLACK:
            if a_is_black:
                a_b += 1
            else:
                b_b += 1
        elif rec.winner == WHITE:
            if a_is_black:
                b_w += 1
            else:
                a_w += 1
        else:
            jigos += 1
    return {
        "a_wins": a_b + a_w,
        "b_wins": b_b + b_w,
        "jigos": jigos,
        "a_score_avg": a_score_total / n_games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-ckpt", type=Path, required=True)
    parser.add_argument("--n-gens", type=int, default=50)
    parser.add_argument("--games-per-gen", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/loop"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/loop"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--search-iters", type=int, default=4)
    parser.add_argument("--search-batch", type=int, default=32)
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--k", type=int, default=1,
        help="Replay-buffer window: train on the union of the last K gens' games."
    )
    parser.add_argument(
        "--sp-workers", type=int, default=1,
        help="Run self-play across this many processes (>=2 for multi-proc)."
    )
    parser.add_argument(
        "--eval-workers", type=int, default=1,
        help="Run match-play across this many processes (>=2 for multi-proc)."
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.out_dir / "log.csv"

    fieldnames = [
        "gen", "wall_sec",
        "buffer_first", "buffer_last", "buffer_positions",
        "sp_avg_moves", "sp_b_wins", "sp_w_wins", "sp_jigos",
        "tr_pol", "tr_own", "tr_top1",
        "val_pol", "val_own", "val_top1",
        "eval_new_wins", "eval_prev_wins", "eval_jigos",
        "eval_new_winrate", "eval_score_margin",
    ]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(f"Init: {args.init_ckpt}")
    print(
        f"Generations: {args.n_gens}, games/gen: {args.games_per_gen}, "
        f"eval games: {args.eval_games}"
    )
    print(f"Replay buffer K: {args.k}")
    print(f"Search per move: {args.search_iters} iters x {args.search_batch} leaves")
    print(f"Device: {device}", flush=True)

    games_buffer: deque[list] = deque(maxlen=args.k)
    prev_ckpt = args.init_ckpt
    for gen in range(1, args.n_gens + 1):
        gen_start = time.time()
        gen_dir = args.out_dir / f"gen{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = gen_dir / "model.pt"
        games_path = args.data_dir / f"gen{gen:03d}.pkl"

        if ckpt_path.exists():
            print(f"[gen {gen}] checkpoint exists, skipping (resume).", flush=True)
            if games_path.exists():
                games_buffer.append(load_games(games_path))
            prev_ckpt = ckpt_path
            continue

        print(f"\n[gen {gen}] self-play with {prev_ckpt}", flush=True)
        # Probe the checkpoint once for board size; cheap.
        probe = load_model(prev_ckpt, device)
        board_size = probe.board_size
        t0 = time.time()
        if args.sp_workers > 1:
            del probe
            torch.cuda.empty_cache()
            games = play_search_games_multiproc(
                n_games=args.games_per_gen,
                n_workers=args.sp_workers,
                ckpt_path=prev_ckpt,
                size=board_size,
                n_iterations=args.search_iters,
                leaves_per_batch=args.search_batch,
                base_seed=gen * 10000,
                temperature_moves=args.temperature_moves,
                temperature=args.temperature,
            )
        else:
            probe.eval()
            games = play_search_games_serial(
                n_games=args.games_per_gen,
                model=probe,
                device=device,
                size=board_size,
                n_iterations=args.search_iters,
                leaves_per_batch=args.search_batch,
                base_seed=gen * 10000,
                temperature_moves=args.temperature_moves,
                temperature=args.temperature,
                progress_every=0,
            )
            del probe
            torch.cuda.empty_cache()
        sp_t = time.time() - t0
        save_games(games, games_path)
        sp_avg_moves = sum(g.n_moves for g in games) / len(games)
        sp_b_wins = sum(1 for g in games if g.winner == BLACK)
        sp_w_wins = sum(1 for g in games if g.winner == WHITE)
        sp_jigos = len(games) - sp_b_wins - sp_w_wins
        print(
            f"  self-play {sp_t:.0f}s  avg_moves={sp_avg_moves:.1f}  "
            f"B/W/J={sp_b_wins}/{sp_w_wins}/{sp_jigos}",
            flush=True,
        )

        games_buffer.append(games)
        buffer_games: list = []
        for batch in games_buffer:
            buffer_games.extend(batch)
        buffer_first = max(1, gen - len(games_buffer) + 1)
        buffer_last = gen
        buffer_positions = sum(len(g.moves) for g in buffer_games)

        new_model = load_model(prev_ckpt, device)
        t0 = time.time()
        tr, va = fine_tune(
            new_model, buffer_games, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            workers=args.workers,
            seed=gen,
        )
        train_t = time.time() - t0
        save_model(new_model, ckpt_path)
        print(
            f"  train {train_t:.1f}s  buffer=gen{buffer_first:03d}-gen{buffer_last:03d} "
            f"({buffer_positions} pos)  tr_pol={tr['policy_loss']:.3f} "
            f"val_pol={va['policy_loss']:.3f} val_top1={va['top1']:.3f} "
            f"val_own={va['own_mse']:.3f}",
            flush=True,
        )

        t0 = time.time()
        if args.eval_workers > 1:
            del new_model
            torch.cuda.empty_cache()
            match = play_match_games_multiproc(
                n_games=args.eval_games,
                n_workers=args.eval_workers,
                ckpt_a=ckpt_path,
                ckpt_b=prev_ckpt,
                size=board_size,
                n_iterations=args.search_iters,
                leaves_per_batch=args.search_batch,
                base_seed=gen * 1_000_000,
                temperature_moves=args.temperature_moves,
                temperature=args.temperature,
            )
        else:
            prev_model = load_model(prev_ckpt, device)
            prev_model.eval()
            new_model.eval()
            match = run_match(
                new_model, prev_model, args.eval_games, device, args,
                seed_offset=gen * 1_000_000,
            )
        eval_t = time.time() - t0
        decisive = match["a_wins"] + match["b_wins"]
        winrate = match["a_wins"] / decisive if decisive else 0.5
        print(
            f"  eval {eval_t:.0f}s  new vs prev = {match['a_wins']}-{match['b_wins']} "
            f"(jigo {match['jigos']})  winrate={winrate:.3f}  "
            f"score_margin={match['a_score_avg']:+.2f}",
            flush=True,
        )

        wall = time.time() - gen_start
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "gen": gen,
                "wall_sec": f"{wall:.1f}",
                "buffer_first": buffer_first,
                "buffer_last": buffer_last,
                "buffer_positions": buffer_positions,
                "sp_avg_moves": f"{sp_avg_moves:.1f}",
                "sp_b_wins": sp_b_wins,
                "sp_w_wins": sp_w_wins,
                "sp_jigos": sp_jigos,
                "tr_pol": f"{tr['policy_loss']:.4f}",
                "tr_own": f"{tr['own_mse']:.4f}",
                "tr_top1": f"{tr['top1']:.4f}",
                "val_pol": f"{va['policy_loss']:.4f}",
                "val_own": f"{va['own_mse']:.4f}",
                "val_top1": f"{va['top1']:.4f}",
                "eval_new_wins": match["a_wins"],
                "eval_prev_wins": match["b_wins"],
                "eval_jigos": match["jigos"],
                "eval_new_winrate": f"{winrate:.4f}",
                "eval_score_margin": f"{match['a_score_avg']:.2f}",
            })

        del prev_model, new_model
        torch.cuda.empty_cache()
        prev_ckpt = ckpt_path
        print(f"[gen {gen}] DONE in {wall:.0f}s ({wall / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
