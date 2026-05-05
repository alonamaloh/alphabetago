"""Train a small policy/ownership net on random-self-play games.

Loads pickled games, featurizes every position once into in-RAM int8 arrays,
then runs a standard PyTorch training loop. After training, dumps a model
checkpoint and runs a few sanity-check predictions on hand-crafted positions.

Usage:
    python scripts/train.py --games data/games_random_9x9.pkl \
        --out runs/random9x9 --epochs 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from alphabetago.board import BLACK, Board
from alphabetago.dataset import featurize_games
from alphabetago.features import N_PLANES, featurize
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.selfplay import load_games
from alphabetago.training import D4Dataset


def make_dataloaders(
    feats: np.ndarray,
    policies: np.ndarray,
    ownerships: np.ndarray,
    val_frac: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    n = feats.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    feats = feats[perm]
    policies = policies[perm]
    ownerships = ownerships[perm]
    n_val = int(n * val_frac)
    feats_t = torch.from_numpy(feats)
    pol_t = torch.from_numpy(policies)
    own_t = torch.from_numpy(ownerships)
    board_size = feats.shape[-1]
    train_base = TensorDataset(feats_t[n_val:], pol_t[n_val:], own_t[n_val:])
    train_ds = D4Dataset(train_base, board_size)
    val_ds = TensorDataset(feats_t[:n_val], pol_t[:n_val], own_t[:n_val])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_loader, val_loader


def run_epoch(
    model: PolicyOwnershipNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    own_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_pol = 0.0
    total_own = 0.0
    total_acc = 0.0
    n_seen = 0
    for feats, policies, ownerships in loader:
        feats = feats.to(device, non_blocking=True).float()
        policies = policies.to(device, non_blocking=True).float()
        ownerships = ownerships.to(device, non_blocking=True).float()
        with torch.set_grad_enabled(is_train):
            policy_logits, own_pred = model(feats)
            log_probs = F.log_softmax(policy_logits, dim=1)
            pol_loss = -(policies * log_probs).sum(dim=1).mean()
            own_loss = F.mse_loss(own_pred, ownerships)
            loss = pol_loss + own_weight * own_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = feats.size(0)
        total_pol += pol_loss.item() * bs
        total_own += own_loss.item() * bs
        total_acc += (policy_logits.argmax(dim=1) == policies.argmax(dim=1)).sum().item()
        n_seen += bs
    return {
        "policy_loss": total_pol / n_seen,
        "ownership_mse": total_own / n_seen,
        "policy_top1_acc": total_acc / n_seen,
    }


def sanity_check(model: PolicyOwnershipNet, device: torch.device, size: int) -> None:
    """Print ownership predictions for a few handcrafted positions."""
    model.eval()
    print("\n=== Sanity-check predictions ===")
    cases: list[tuple[str, Board]] = []

    cases.append(("empty board", Board(size=size)))

    b = Board(size=size)
    b.play(b.point(size // 2, size // 2))
    cases.append(("single B in center", b))

    b = Board(size=size)
    b.play(b.point(1, 1))   # B
    b.play(b.point(size - 2, size - 2))  # W
    cases.append(("B near (1,1), W near (n-2,n-2)", b))

    for label, board in cases:
        feats = torch.from_numpy(featurize(board)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            policy_logits, own_pred = model(feats)
        own = own_pred[0].cpu().numpy()
        # Rotate ownership to absolute black perspective for readability.
        if board.to_play != BLACK:
            own = -own
        print(f"\n[{label}]  to_play={'B' if board.to_play == BLACK else 'W'}")
        print(board.text())
        print("predicted ownership (black perspective; range -1..+1):")
        for r in range(size):
            row = " ".join(f"{own[r, c]:+0.2f}" for c in range(size))
            print(f"  {row}")
        own_sum = own.sum()
        print(f"  sum(ownership) = {own_sum:+.2f}  (predicted score, B perspective)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path, required=True, nargs="+",
        help="One or more pickles of GameRecords (concatenated)."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output dir for checkpoint and log."
    )
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--own-weight", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU workers used for one-shot featurization.")
    parser.add_argument("--init", type=Path, default=None,
                        help="Optional checkpoint to initialize weights from.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading games from {len(args.games)} file(s)...")
    games = []
    for path in args.games:
        games.extend(load_games(path))
    print(f"  {len(games)} games loaded total.")

    print("Featurizing positions (one-time, parallel)...")
    t0 = time.time()
    feats, policies, ownerships = featurize_games(games, n_workers=args.workers)
    t1 = time.time()
    n_pos = feats.shape[0]
    bytes_ = feats.nbytes + policies.nbytes + ownerships.nbytes
    print(
        f"  {n_pos} positions in {t1 - t0:.1f}s "
        f"({bytes_ / 1e9:.2f} GB int8/int32)."
    )

    train_loader, val_loader = make_dataloaders(
        feats, policies, ownerships, args.val_frac, args.batch_size, args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    model = PolicyOwnershipNet(
        board_size=args.size, channels=args.channels, n_blocks=args.blocks
    ).to(device)
    if args.init is not None:
        state = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Initialized weights from {args.init}")
    print(f"Model parameters: {model.parameter_count():,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    log_path = args.out / "train.log"
    with open(log_path, "w") as log:
        log.write("epoch,split,policy_loss,ownership_mse,policy_top1_acc\n")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_stats = run_epoch(model, train_loader, device, optimizer, args.own_weight)
            val_stats = run_epoch(model, val_loader, device, None, args.own_weight)
            t1 = time.time()
            tr = train_stats
            va = val_stats
            print(
                f"epoch {epoch:>2}: "
                f"train pol={tr['policy_loss']:.4f} own={tr['ownership_mse']:.4f} "
                f"acc={tr['policy_top1_acc']:.3f}  |  "
                f"val pol={va['policy_loss']:.4f} own={va['ownership_mse']:.4f} "
                f"acc={va['policy_top1_acc']:.3f}  ({t1 - t0:.1f}s)"
            )
            for split, stats in [("train", train_stats), ("val", val_stats)]:
                log.write(
                    f"{epoch},{split},"
                    f"{stats['policy_loss']:.6f},"
                    f"{stats['ownership_mse']:.6f},"
                    f"{stats['policy_top1_acc']:.6f}\n"
                )
            log.flush()

    ckpt_path = args.out / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "board_size": args.size,
                "in_planes": N_PLANES,
                "channels": args.channels,
                "n_blocks": args.blocks,
            },
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}.")

    sanity_check(model, device, args.size)


if __name__ == "__main__":
    main()
