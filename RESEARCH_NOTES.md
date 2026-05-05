# Research notes

Working notes on what we've built, what we've learned so far, and what's
worth trying next. Updated as we go; the goal is a useful handover doc, not
a finished writeup.

## Project goal

Build a Go-playing engine on **9×9, no komi** that uses **alpha-beta search**
(not MCTS), guided by a neural network. The unusual ingredient is
**information-bit deepening**: the cost of a path from the root is the sum
of `-log2(policy_probability)` over its moves, so high-prior lines reach
deeper before low-prior ones get any consideration. Two questions of
interest:

1. Is alpha-beta-with-info-bit-deepening viable in Go (huge branching factor)?
2. Can we get a learning signal "from nothing" — random init, no pretraining?

## Pipeline as it stands

- **Rules.** `Board` keeps a 1-D mailbox with off-board sentinels, a union-find
  over chains with per-root liberty *sets*, plus a Zobrist hash maintained
  incrementally. `is_legal` enforces **positional superko** via tentative-hash
  lookup against the full position history. No komi. Tromp-Taylor area
  scoring; ownership is per-point classified BLACK / WHITE / EMPTY (contested).
- **Featurizer.** 20 input planes, oriented to the side to play: own/opp
  stones (2), one-hot bucketed liberty count for own and opp (8), one-hot
  bucketed chain stone count for own and opp (8), legality mask (1),
  all-ones (1). The legality plane folds in ko + superko prohibitions
  automatically.
- **Network.** Small ResNet, **3 blocks × 32 channels**, ~75k parameters.
  Channel-wise LayerNorm (ConvNeXt-style) so train and inference behave
  identically. Two heads: policy logits over `size² + 1` (PASS at the end);
  ownership in (-1, 1) via tanh. **Predicted score is the sum of ownership
  predictions** — there is no separate value head. Ownership target ∈ {-1, 0, +1};
  loss is MSE.
- **Search.** Each iteration:
  1. Walk the in-memory tree best-first by accumulated `-log2(prior)`,
     materializing children lazily; stop when we have K unexpanded leaves.
  2. Featurize all K boards, run them through the NN as one batch, populate
     each leaf's policy (legal-only softmax) / value / ownership.

  After all iterations, a single negamax alpha-beta sweep over the populated
  tree (moves visited in descending policy order, for tighter cutoffs)
  returns the root value and the best move. Currently a true tree, no
  transposition / DAG.
- **Training.** Soft cross-entropy on the policy + MSE on per-point
  ownership. Adam, lr 5e-4 for fine-tuning (1e-3 for the from-scratch
  random-data run), weight decay 1e-4.
- **Self-play.** Search-driven move selection with a heuristic
  **eye-avoidance** crutch (filter own-eye moves out of candidates;
  PASS as fallback). Otherwise games never terminate. Move policy target
  is **one-hot on the alpha-beta best non-eye move** (NOT a soft target —
  see "key insights" below). Move played is sampled from the
  eye-filtered visit distribution with temperature for the first 20 plies,
  greedy thereafter.

## Generation 0 (random-data baseline)

10k random self-play games (eye-avoiding random move selection), 1.2M
positions, 5 epochs of from-scratch training. Learns sensible ownership
patterns (a single black stone in the center predicts ownership +0.45 at
the stone with a positive halo) but the policy is essentially uniform —
random play teaches no policy structure beyond eye-avoidance. Saved as
`runs/keep/gen0_random.pt`.

## The 50-generation closed-loop experiment

**Recipe.** Per generation N: (1) generate **100 search self-play games**
with the gen N−1 model (4 search iterations × 32 leaves = 128 NN evals per
move, ~3.5 s/game on a 3080 Ti); (2) **fine-tune from gen N−1 weights** for
8 epochs (lr 5e-4, batch 128); (3) **head-to-head 50-game match** between
gen N and gen N−1, alternating sides.

**Driver:** `scripts/run_loop.py` (idempotent — re-uses an existing
`runs/.../genNNN/model.pt` if present).

**Run summary.** Stopped at the end of generation 41 (gen 42's checkpoint
is on disk but not match-evaluated). Highlights from `runs/loop/log.csv`:

| gen | winrate vs prev | score margin | val top-1 | val own MSE |
|---:|---:|---:|---:|---:|
| 1   | **0.84** | +10.0 | 0.128 | 0.549 |
| 2   | 0.64 | +6.9  | 0.191 | 0.446 |
| 5   | 0.60 | +14.3 | 0.128 | 0.561 |
| 10  | 0.45 | +0.7  | 0.119 | 0.449 |
| 20  | 0.57 | −0.2  | 0.151 | 0.554 |
| 31  | 0.35 | −8.5  | 0.117 | 0.592 |
| 41  | 0.54 | +3.4  | 0.146 | 0.545 |

`last_5_mean(winrate) = 0.53`, `last_5_mean(margin) = +0.05`. Across all 41
gens: winrate range **0.16 – 0.84**, margin range **−26 to +25**. Plot:
`notes/loop_50gen_curves.png`.

(The prior, manually-trained `gen1` and `gen2` checkpoints in `runs/gen1`
and `runs/gen2` were 92%-vs-gen-0 and 75%-vs-gen-1 respectively. They are
NOT the same as the loop's gen001 / gen002 — different seeds.)

## What this run says

**The loop produces a real signal in the first cycle.** Gen 1 dominates
gen 0 (0.84 winrate, +10 score margin), and gen 2 dominates gen 1 (0.64,
+7). That establishes that search-driven self-play data, fine-tuned into
the previous generation, contains learnable structure beyond the prior.

**The signal does not compound under this recipe.** From roughly **gen 5
onward**, gen-N-vs-gen-(N−1) winrate collapses into noise around 0.5.
Validation top-1 accuracy and ownership MSE are flat across all 41 gens —
no monotone improvement, just oscillation. The curve looks plateaued.

The most likely culprits, roughly in order of suspected importance:

1. **No replay buffer.** Each gen fine-tunes on its own ~10k positions for
   8 epochs and partly forgets prior gens. Standard AlphaZero practice is
   to train on the union of recent N gens; we don't.
2. **Tiny dataset per gen.** 100 games × ~100 moves ≈ 10k positions is
   small. The per-cycle gradient signal is dominated by sampling noise.
3. **Weak per-move target.** The search uses 128 NN evaluations on a
   near-uniform policy. The alpha-beta best move at this budget is only
   slightly better than the prior, so the per-position teaching signal is
   shallow.
4. **Tiny network.** 75k parameters, 3 blocks. May be capacity-limited;
   the board still has many "modes" the net can't simultaneously fit.

## Key insights worth not forgetting

- **Soft targets from alpha-beta are a bad idea.** Alpha-beta values for
  non-best moves are upper bounds tightened only to the cutoff threshold —
  often `≤ best_v + ε` even for atrocious moves. They contain no quality
  ranking among non-best moves. The natural search-derived policy target is
  **one-hot on the best move**.
- **MCTS-style "visit count" targets don't translate cleanly to alpha-beta.**
  In MCTS each visit walks the tree under UCB, so visit counts encode value
  estimates. Our leaf collection is pure best-first by prior, so visit counts
  here mostly *re-encode the prior* rather than adding search information.
  We tried this; it didn't add useful signal beyond the one-hot best move.
- **Eye-avoidance is a crutch.** The simple GnuGo-style eye check
  (`Board.is_eye_for`) is shape-only and ignores life/death. Without it,
  search-driven games never pass and run to the move-cap (we hit 2000 moves
  on every game). With it, games terminate at ~100 moves. The plan is to
  drop it once the value head clearly punishes own-eye fills (and the
  policy avoids them) — currently neither is true.
- **Positional superko via Zobrist is cheap.** ~7 ms/game self-play
  overhead. Worth doing properly even at this scale.
- **Game length is a soft signal.** Random play games average 119 moves;
  gen-0 search-play 84; gen-1 search-play 100; loop gens 95–100. As the
  player gets more deliberate, games lengthen until the eye-avoidance
  ceiling stabilizes them.

## Saved checkpoints (for comparing future recipes)

All in `runs/keep/` (gitignored — these are local artifacts, not source):

| File | What it is |
|---|---|
| `gen0_random.pt` | Random-self-play-trained baseline (gen 0). |
| `loop_gen001.pt` | After 1 cycle of search-loop fine-tuning from gen 0. |
| `loop_gen005.pt` | Around when the per-cycle improvement saturated. |
| `loop_gen020.pt` | Mid-plateau. |
| `loop_gen041.pt` | Last logged generation; representative of the plateau. |

`runs/random9x9_ln/model.pt` is the same model as `gen0_random.pt`.

`runs/loop/genNNN/model.pt` for N = 1..42 are still present; the `runs/keep/`
copies are pinned references. Likewise, the **per-generation game data is
preserved** in `data/loop/gen001.pkl … data/loop/gen042.pkl` so the next
recipe can warm-start from those games (replay buffer, etc.) without
regenerating them.

## What to try next, in roughly descending leverage

1. **Replay buffer.** Train each new gen on the union of the last K gens'
   games (e.g., K=10 → ~100k positions). Closes the most likely cause of
   the plateau and is cheap to implement.
2. **Bigger search budget per move.** 128 NN evals on a near-uniform policy
   is shallow; 512 or 1024 sharpens the best-move target.
3. **More games per generation** (500–1000) — higher SNR per cycle, more
   decisive head-to-head outcomes.
4. **Bigger / deeper net.** 75k params is small. 4–6 blocks × 64 channels
   would still be modest by KataGo standards but ~10× the capacity.
5. **KataGo SGF supervised pretraining** — the fallback plan from the
   start. Replay KataGo's published 9×9 self-play games through our engine
   to produce *our*-format training data, supervised-pretrain the net, then
   resume the closed loop from that warm start. Would also let us tune
   network architecture and search hyperparameters against a real signal
   instead of in the noise-floor regime.
6. **Drop eye-avoidance.** Probably premature until the value head can
   clearly tell that own-eye fills are bad.
7. **DAG transposition.** Currently a true tree; sharing nodes across
   transposed positions would compound search depth in tactical phases.

A sensible smallest experiment is **(1) + (2) + a fresh 30-gen run** off
the `gen0_random.pt` start, comparing the new run's gen-N curve to the
50-gen baseline curve we just produced. If that still plateaus, escalate
to (3) and/or (5).

## Open questions

- Is the plateau fundamental to alpha-beta + best-first leaf collection at
  this net capacity, or just a function of the small per-cycle dataset?
- Does our information-bit deepening actually do better than uniform
  iterative deepening once the policy has real structure? Right now the
  policy is too flat to tell.
- At what point does eye-avoidance stop being load-bearing? Specifically,
  does the value head ever start predicting strongly negative ownership for
  own-eye-fill moves under our current training?
- Does the score-from-ownership shortcut hold up empirically (does
  `sum(predicted_ownership)` track the eventual game score on held-out
  positions)?
