"""Run a head-to-head match between two model checkpoints.

Each game has alternate sides for the two models so first-move advantage
is split evenly. Move selection follows the same rules as search self-play
(eye-avoidance, opening temperature, greedy alpha-beta best afterwards).

Usage:
    python scripts/match_eval.py --a runs/random9x9_ln/model.pt \
        --b runs/gen1/model.pt --n 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from alphabetago.board import BLACK, WHITE
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.selfplay import play_match_game, play_match_games_multiproc


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
    parser.add_argument("--a", type=Path, required=True, help="Model A checkpoint.")
    parser.add_argument("--b", type=Path, required=True, help="Model B checkpoint.")
    parser.add_argument("--n", type=int, default=50, help="Number of games.")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Match: A={args.a} vs B={args.b}")
    print(f"Device: {device}, {args.n} games, workers={args.workers}")

    if args.workers > 1:
        # Probe A for board size, then hand off to multiproc.
        probe = load_model(args.a, device)
        size = probe.board_size
        del probe
        torch.cuda.empty_cache()
        t0 = time.time()
        match = play_match_games_multiproc(
            n_games=args.n,
            n_workers=args.workers,
            ckpt_a=args.a,
            ckpt_b=args.b,
            size=size,
            n_iterations=args.iterations,
            leaves_per_batch=args.batch,
            base_seed=args.seed,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
        )
        dt = time.time() - t0
        a_total = match["a_wins"]
        b_total = match["b_wins"]
        jigos = match["jigos"]
        decisive = a_total + b_total
        print(f"\nFinal ({dt:.0f}s):")
        print(f"  A wins: {a_total}")
        print(f"  B wins: {b_total}")
        print(f"  Jigos:  {jigos}")
        if decisive > 0:
            a_winrate = a_total / decisive
            n = decisive
            z = 1.96
            denom = 1 + z * z / n
            center = (a_winrate + z * z / (2 * n)) / denom
            margin = z * (a_winrate * (1 - a_winrate) / n + z * z / (4 * n * n)) ** 0.5 / denom
            lo, hi = max(0.0, center - margin), min(1.0, center + margin)
            print(
                f"  A win rate (decisive only): {a_winrate * 100:.1f}%  "
                f"[95% CI {lo * 100:.1f}–{hi * 100:.1f}%]"
            )
        print(f"  Mean score (A perspective): {match['a_score_avg']:+.2f}")
        return

    model_a = load_model(args.a, device)
    model_b = load_model(args.b, device)
    print(
        f"Search per move: {args.iterations} iters x {args.batch} leaves "
        f"({args.iterations * args.batch} NN evals)"
    )

    a_b_wins = a_w_wins = b_b_wins = b_w_wins = jigos = 0
    a_score_total = 0.0  # signed score from A's perspective each game
    t0 = time.time()
    for i in range(args.n):
        a_is_black = (i % 2 == 0)
        if a_is_black:
            black_model, white_model = model_a, model_b
        else:
            black_model, white_model = model_b, model_a
        rec = play_match_game(
            model_black=black_model,
            model_white=white_model,
            device=device,
            size=model_a.board_size,
            n_iterations=args.iterations,
            leaves_per_batch=args.batch,
            seed=args.seed + i,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
        )
        a_score_total += rec.final_score if a_is_black else -rec.final_score

        if rec.winner == BLACK:
            if a_is_black:
                a_b_wins += 1
            else:
                b_b_wins += 1
        elif rec.winner == WHITE:
            if a_is_black:
                b_w_wins += 1
            else:
                a_w_wins += 1
        else:
            jigos += 1

        if (i + 1) % max(1, args.n // 10) == 0:
            dt = time.time() - t0
            a_total = a_b_wins + a_w_wins
            b_total = b_b_wins + b_w_wins
            played = a_total + b_total + jigos
            print(
                f"  [{i + 1}/{args.n}] A:{a_total} B:{b_total} jigo:{jigos}  "
                f"({dt:.0f}s, {dt / played:.1f}s/game)"
            )

    a_total = a_b_wins + a_w_wins
    b_total = b_b_wins + b_w_wins
    decisive = a_total + b_total
    print()
    print("Final:")
    print(f"  A wins: {a_total}  ({a_b_wins} as B, {a_w_wins} as W)")
    print(f"  B wins: {b_total}  ({b_b_wins} as B, {b_w_wins} as W)")
    print(f"  Jigos:  {jigos}")
    if decisive > 0:
        a_winrate = a_total / decisive
        # Wilson 95% CI for the binomial proportion.
        n = decisive
        z = 1.96
        denom = 1 + z * z / n
        center = (a_winrate + z * z / (2 * n)) / denom
        margin = z * (a_winrate * (1 - a_winrate) / n + z * z / (4 * n * n)) ** 0.5 / denom
        lo, hi = max(0.0, center - margin), min(1.0, center + margin)
        print(
            f"  A win rate (decisive only): {a_winrate * 100:.1f}%  "
            f"[95% CI {lo * 100:.1f}–{hi * 100:.1f}%]"
        )
    print(f"  Mean score (A perspective, all games): {a_score_total / args.n:+.2f}")


if __name__ == "__main__":
    main()
