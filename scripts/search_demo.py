"""Run the alpha-beta-with-info-bit search on a few hand-crafted positions
using a trained checkpoint, and report what the search does relative to the
raw policy/value predictions.

Usage:
    python scripts/search_demo.py --ckpt runs/random9x9_ln/model.pt \
        --iterations 8 --batch 128
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from alphabetago.board import BLACK, PASS, Board
from alphabetago.features import featurize
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.search import count_nodes, expand_nodes, search


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


def fmt_move(board: Board, move: int | None) -> str:
    if move is None:
        return "—"
    if move == PASS:
        return "PASS"
    r, c = board.coord(move)
    return f"({r},{c})"


def raw_predictions(
    board: Board, model: PolicyOwnershipNet, device: torch.device
) -> tuple[dict[int, float], float, np.ndarray]:
    """Single-position NN evaluation (no search). Returns the same fields the
    search would store on a node."""
    feats = torch.from_numpy(featurize(board)).unsqueeze(0).to(device).float()
    with torch.no_grad():
        policy_logits, ownership = model(feats)
    n = board.size
    legal = board.legal_moves(include_pass=True)
    legal_indices = [n * n if m == PASS else (board.coord(m)[0] * n + board.coord(m)[1])
                     for m in legal]
    logits_np = policy_logits[0].cpu().numpy()
    legal_logits = np.array([logits_np[i] for i in legal_indices])
    e = np.exp(legal_logits - legal_logits.max())
    probs = e / e.sum()
    pol = {m: float(p) for m, p in zip(legal, probs, strict=True)}
    own = ownership[0].cpu().numpy()
    return pol, float(own.sum()), own


def report_position(
    label: str,
    board: Board,
    model: PolicyOwnershipNet,
    device: torch.device,
    n_iterations: int,
    leaves_per_batch: int,
) -> None:
    print(f"\n=== {label} ===")
    print(board.text())
    side = "B" if board.to_play == BLACK else "W"
    print(f"to_play: {side}   move_number: {board.move_number}")

    pol, value_raw, _ = raw_predictions(board, model, device)
    top5 = sorted(pol.items(), key=lambda kv: -kv[1])[:5]
    print("\nRaw NN (no search):")
    print(f"  value (sum of ownerships, {side}-perspective): {value_raw:+.2f}")
    print("  top-5 policy moves:")
    for m, p in top5:
        cost = -math.log2(p) if p > 0 else float("inf")
        print(f"    {fmt_move(board, m):>8}  p={p:.3f}  cost={cost:.2f} bits")

    t0 = time.time()
    root, best_m, value = search(
        board, model, device,
        n_iterations=n_iterations,
        leaves_per_batch=leaves_per_batch,
    )
    t1 = time.time()
    total, expanded = count_nodes(root)
    print(f"\nSearch ({n_iterations} iters x {leaves_per_batch} leaves):")
    print(f"  wall time: {t1 - t0:.2f}s")
    print(f"  tree: {total} nodes total, {expanded} expanded")
    print(f"  best move: {fmt_move(board, best_m)}    "
          f"alpha-beta value ({side}-perspective): {value:+.2f}")
    if best_m in root.policy:
        prior = root.policy[best_m]
        rank = sum(1 for _, p in pol.items() if p > prior) + 1
        print(f"  best move's policy prior: {prior:.3f} (rank {rank})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    size = model.board_size
    print(f"Loaded model: size={size}, channels={model.channels}, "
          f"blocks={model.n_blocks}, params={model.parameter_count():,}")
    print(f"Device: {device}")
    # Warm up CUDA so the first timed search isn't dominated by allocator setup.
    expand_nodes([], model, device)
    _ = expand_nodes  # silence linters

    # Position 1: empty board.
    b = Board(size=size)
    report_position("empty board", b, model, device, args.iterations, args.batch)

    # Position 2: one black stone in center.
    b = Board(size=size)
    b.play(b.point(size // 2, size // 2))
    report_position("after B (center)", b, model, device, args.iterations, args.batch)

    # Position 3: black corner stone, white opposite corner.
    b = Board(size=size)
    b.play(b.point(2, 2))
    b.play(b.point(size - 3, size - 3))
    report_position(
        f"after B(2,2) W({size - 3},{size - 3})",
        b, model, device, args.iterations, args.batch,
    )


if __name__ == "__main__":
    main()
