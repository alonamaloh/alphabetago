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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from alphabetago.board import BLACK, WHITE
from alphabetago.dataset import featurize_games
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.selfplay import (
    play_match_game,
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
    return model


def save_model(model: PolicyOwnershipNet, ckpt_path: Path) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "board_size": model.board_size,
                "in_planes": model.in_planes,
                "channels": model.channels,
                "n_blocks": model.n_blocks,
            },
        },
        ckpt_path,
    )


def fine_tune(
    model: PolicyOwnershipNet,
    games: list,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    feats, policies, ownerships = featurize_games(games, n_workers=args.workers)
    n = feats.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    feats = feats[perm]
    policies = policies[perm]
    ownerships = ownerships[perm]
    n_val = max(1, int(n * 0.1))

    feats_t = torch.from_numpy(feats)
    pol_t = torch.from_numpy(policies)
    own_t = torch.from_numpy(ownerships)
    train_ds = TensorDataset(feats_t[n_val:], pol_t[n_val:], own_t[n_val:])
    val_ds = TensorDataset(feats_t[:n_val], pol_t[:n_val], own_t[:n_val])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    final_train: dict[str, float] = {}
    final_val: dict[str, float] = {}
    for _ in range(args.epochs):
        for split, loader, is_train in (
            ("train", train_loader, True),
            ("val", val_loader, False),
        ):
            model.train(is_train)
            tot_pol = tot_own = 0.0
            tot_acc = 0
            n_seen = 0
            for f, p, o in loader:
                f = f.to(device, non_blocking=True).float()
                p = p.to(device, non_blocking=True).float()
                o = o.to(device, non_blocking=True).float()
                with torch.set_grad_enabled(is_train):
                    pl, op = model(f)
                    log_p = F.log_softmax(pl, dim=1)
                    pol_loss = -(p * log_p).sum(dim=1).mean()
                    own_loss = F.mse_loss(op, o)
                    loss = pol_loss + own_loss
                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                bs = f.size(0)
                tot_pol += pol_loss.item() * bs
                tot_own += own_loss.item() * bs
                tot_acc += int((pl.argmax(1) == p.argmax(1)).sum().item())
                n_seen += bs
            metrics = {
                "policy_loss": tot_pol / n_seen,
                "own_mse": tot_own / n_seen,
                "top1": tot_acc / n_seen,
            }
            if split == "train":
                final_train = metrics
            else:
                final_val = metrics
    return final_train, final_val


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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = args.out_dir / "log.csv"

    fieldnames = [
        "gen", "wall_sec",
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
    print(f"Search per move: {args.search_iters} iters x {args.search_batch} leaves")
    print(f"Device: {device}", flush=True)

    prev_ckpt = args.init_ckpt
    for gen in range(1, args.n_gens + 1):
        gen_start = time.time()
        gen_dir = args.out_dir / f"gen{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = gen_dir / "model.pt"
        games_path = args.data_dir / f"gen{gen:03d}.pkl"

        if ckpt_path.exists():
            print(f"[gen {gen}] checkpoint exists, skipping (resume).", flush=True)
            prev_ckpt = ckpt_path
            continue

        print(f"\n[gen {gen}] self-play with {prev_ckpt}", flush=True)
        prev_model = load_model(prev_ckpt, device)
        prev_model.eval()
        t0 = time.time()
        games = play_search_games_serial(
            n_games=args.games_per_gen,
            model=prev_model,
            device=device,
            size=prev_model.board_size,
            n_iterations=args.search_iters,
            leaves_per_batch=args.search_batch,
            base_seed=gen * 10000,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
            progress_every=0,
        )
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

        new_model = load_model(prev_ckpt, device)
        t0 = time.time()
        tr, va = fine_tune(new_model, games, args, device, seed=gen)
        train_t = time.time() - t0
        save_model(new_model, ckpt_path)
        print(
            f"  train {train_t:.1f}s  tr_pol={tr['policy_loss']:.3f} "
            f"val_pol={va['policy_loss']:.3f} val_top1={va['top1']:.3f} "
            f"val_own={va['own_mse']:.3f}",
            flush=True,
        )

        del prev_model
        torch.cuda.empty_cache()

        prev_model = load_model(prev_ckpt, device)
        prev_model.eval()
        new_model.eval()
        t0 = time.time()
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
