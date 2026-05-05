"""Plot the per-generation training/match curves from a `run_loop.py` log.

Reads `<out-dir>/log.csv` and writes a `curves.png` (and prints a summary).

Usage:
    python scripts/plot_loop.py --log runs/loop/log.csv \
        --out runs/loop/curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with open(args.log) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("No rows in log.")
        return

    gens = [int(r["gen"]) for r in rows]
    winrates = [float(r["eval_new_winrate"]) for r in rows]
    margins = [float(r["eval_score_margin"]) for r in rows]
    val_top1 = [float(r["val_top1"]) for r in rows]
    val_own = [float(r["val_own"]) for r in rows]

    print(f"{len(rows)} generations logged.")
    print(
        f"Win rate of gen N vs gen N-1: "
        f"min={min(winrates):.2f} max={max(winrates):.2f} "
        f"last5_mean={sum(winrates[-5:]) / min(5, len(winrates)):.2f}"
    )
    print(
        f"Score margin (gen N vs gen N-1): "
        f"min={min(margins):+.2f} max={max(margins):+.2f} "
        f"last5_mean={sum(margins[-5:]) / min(5, len(margins)):+.2f}"
    )

    out_path = args.out or args.log.with_name("curves.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(gens, winrates, "-o", markersize=3)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("win rate of gen N vs gen N-1")
    ax.set_ylim(0, 1)
    ax.set_title("Self-improvement (decisive winrate)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(gens, margins, "-o", markersize=3, color="C1")
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("mean score margin (points)")
    ax.set_title("Mean score margin (gen N vs gen N-1)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gens, val_top1, "-o", markersize=3, color="C2")
    ax.set_xlabel("generation")
    ax.set_ylabel("validation top-1 accuracy")
    ax.set_title("Policy top-1 (val) over generations")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(gens, val_own, "-o", markersize=3, color="C3")
    ax.set_xlabel("generation")
    ax.set_ylabel("validation ownership MSE")
    ax.set_title("Ownership MSE (val) over generations")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
