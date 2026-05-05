"""Quick throughput benchmark for self-play. Times generating N games under
different parallelization strategies; reports games/second and avg moves/game.

Modes:
  serial      — current code path (one game at a time).
  vectorized  — M concurrent games per process, batched NN calls.
  multiproc   — N processes each running serial play independently.

Usage:
    python scripts/bench_selfplay.py --ckpt runs/keep/gen0_random.pt \
        --mode serial --n 20 --iters 4 --batch 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from alphabetago.nn import PolicyOwnershipNet


def _mp_worker(rank, n_local, ckpt, iters, batch, seed_offset, temp_moves, temp, q):
    import torch as _torch

    from alphabetago.selfplay import play_search_games_serial

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = load_model(Path(ckpt), device)
    games = play_search_games_serial(
        n_games=n_local,
        model=model,
        device=device,
        size=model.board_size,
        n_iterations=iters,
        leaves_per_batch=batch,
        base_seed=seed_offset,
        temperature_moves=temp_moves,
        temperature=temp,
        progress_every=0,
    )
    q.put((rank, len(games), sum(len(g.moves) for g in games)))


def load_model(ckpt_path: Path, device: torch.device) -> PolicyOwnershipNet:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    m = PolicyOwnershipNet(
        board_size=cfg["board_size"],
        in_planes=cfg["in_planes"],
        channels=cfg["channels"],
        n_blocks=cfg["n_blocks"],
    ).to(device)
    m.load_state_dict(state["model_state"])
    m.eval()
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["serial", "vectorized", "multiproc"], default="serial"
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--concurrent", type=int, default=8,
                        help="Games run concurrently in vectorized mode.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes in multiproc mode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature-moves", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode}  Device: {device}")
    print(
        f"Search: {args.iters} iters x {args.batch} leaves "
        f"= {args.iters * args.batch} NN evals/move"
    )

    if args.mode == "serial":
        from alphabetago.selfplay import play_search_games_serial

        model = load_model(args.ckpt, device)
        t0 = time.time()
        games = play_search_games_serial(
            n_games=args.n,
            model=model,
            device=device,
            size=model.board_size,
            n_iterations=args.iters,
            leaves_per_batch=args.batch,
            base_seed=args.seed,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
            progress_every=0,
        )
        dt = time.time() - t0
    elif args.mode == "vectorized":
        from alphabetago.selfplay import play_search_games_vectorized

        model = load_model(args.ckpt, device)
        print(f"Concurrent games per process: {args.concurrent}  "
              f"-> batches up to {args.concurrent * args.batch} leaves")
        t0 = time.time()
        games = play_search_games_vectorized(
            n_games=args.n,
            n_concurrent=args.concurrent,
            model=model,
            device=device,
            size=model.board_size,
            n_iterations=args.iters,
            leaves_per_batch=args.batch,
            base_seed=args.seed,
            temperature_moves=args.temperature_moves,
            temperature=args.temperature,
        )
        dt = time.time() - t0
    elif args.mode == "multiproc":
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        per_worker = args.n // args.workers
        leftover = args.n - per_worker * args.workers
        chunks = [per_worker + (1 if i < leftover else 0) for i in range(args.workers)]
        print(f"Workers: {args.workers}  Per-worker games: {chunks}")

        q = ctx.Queue()
        procs = []
        seed = args.seed
        t0 = time.time()
        for rank, n_local in enumerate(chunks):
            if n_local == 0:
                continue
            p = ctx.Process(
                target=_mp_worker,
                args=(rank, n_local, str(args.ckpt), args.iters, args.batch,
                      seed, args.temperature_moves, args.temperature, q),
            )
            p.start()
            procs.append(p)
            seed += n_local
        results = [q.get() for _ in procs]
        for p in procs:
            p.join()
        dt = time.time() - t0
        total = sum(r[1] for r in results)
        avg_moves = sum(r[2] for r in results) / total if total else 0
        print(f"\nResults: {total} games in {dt:.1f}s ({dt / total * 1000:.0f} ms/game)")
        print(f"  throughput: {total / dt:.2f} games/s")
        print(f"  avg moves/game: {avg_moves:.1f}")
        return

    avg_moves = sum(len(g.moves) for g in games) / len(games)
    print(f"\nResults: {len(games)} games in {dt:.1f}s ({dt / len(games) * 1000:.0f} ms/game)")
    print(f"  throughput: {len(games) / dt:.2f} games/s")
    print(f"  avg moves/game: {avg_moves:.1f}")


if __name__ == "__main__":
    main()
