"""Shared training helpers used by run_loop.py and replay_loop.py.

A single fine-tuning step over a list of GameRecords: featurizes them all,
splits into train/val, runs N epochs of soft-cross-entropy + ownership-MSE,
returns the final-epoch metrics.

Training samples are augmented with a random D4 board symmetry (one of 8
rotations/reflections) per `__getitem__` call to prevent the network from
memorizing position-specific spatial patterns within a game and overfitting.
The transformation is applied consistently to features, policy target, and
ownership target. Validation samples are NOT augmented so that val metrics
remain a deterministic measure of generalization.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from alphabetago.dataset import featurize_games
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.selfplay import GameRecord


def _apply_d4(t: torch.Tensor, sym: int) -> torch.Tensor:
    """Apply one of 8 D4 dihedral transformations to a tensor whose last two
    dims are (H, W). `sym` ∈ [0, 7]: bit 0 = horizontal flip, bits 1-2 =
    number of 90° CCW rotations."""
    if sym & 1:
        t = torch.flip(t, dims=[-1])
    k = (sym >> 1) & 3
    if k:
        t = torch.rot90(t, k, dims=[-2, -1])
    return t


def apply_d4_sample(
    features: torch.Tensor,
    policy: torch.Tensor,
    ownership: torch.Tensor,
    sym: int,
    board_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same D4 symmetry to a (features, policy, ownership) triple.

    `features`: [C, H, W]. `policy`: [H*W + 1] (board moves + PASS).
    `ownership`: [H, W]. PASS stays at index H*W (it's not a board point).
    """
    n = board_size
    features_out = _apply_d4(features, sym)
    policy_board = policy[: n * n].view(n, n)
    policy_board = _apply_d4(policy_board, sym).contiguous().view(-1)
    policy_out = torch.cat([policy_board, policy[n * n :]])
    ownership_out = _apply_d4(ownership, sym)
    return features_out, policy_out, ownership_out


class D4Dataset(Dataset):
    """Wrap a TensorDataset of (features, policy, ownership) triples and
    apply a random D4 symmetry per item access. Length is preserved; each
    epoch sees fresh random transformations."""

    def __init__(self, base: TensorDataset, board_size: int):
        self._base = base
        self._size = board_size

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        f, p, o = self._base[idx]
        sym = random.randrange(8)
        return apply_d4_sample(f, p, o, sym, self._size)


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
    games: list[GameRecord],
    device: torch.device,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    workers: int | None = None,
    seed: int = 0,
    val_frac: float = 0.1,
) -> tuple[dict[str, float], dict[str, float]]:
    """Train `model` on `games` for `epochs` epochs. Returns the final-epoch
    train and val metrics."""
    feats, policies, ownerships = featurize_games(games, n_workers=workers)
    n = feats.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    feats = feats[perm]
    policies = policies[perm]
    ownerships = ownerships[perm]
    n_val = max(1, int(n * val_frac))

    feats_t = torch.from_numpy(feats)
    pol_t = torch.from_numpy(policies)
    own_t = torch.from_numpy(ownerships)
    board_size = feats.shape[-1]
    train_base = TensorDataset(feats_t[n_val:], pol_t[n_val:], own_t[n_val:])
    val_ds = TensorDataset(feats_t[:n_val], pol_t[:n_val], own_t[:n_val])
    train_ds = D4Dataset(train_base, board_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    final_train: dict[str, float] = {}
    final_val: dict[str, float] = {}
    for _ in range(epochs):
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
