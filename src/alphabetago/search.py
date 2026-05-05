"""Alpha-beta search with information-bit deepening and batched leaf evaluation.

Tree structure:
- Each node holds a Board snapshot, the NN's policy/value/ownership predictions
  (populated on expansion), and a dict of explored children.
- The cost of a path from the root is the sum of `-log2(policy_prob)` over the
  moves on the path, so high-probability lines reach deeper into the tree
  before low-probability ones get any consideration.

Each search iteration:
1. Walk the tree best-first by accumulated bit cost and collect up to K
   distinct unexpanded leaves.
2. Featurize all K boards and run them through the NN as a single batch.
3. Populate each leaf's policy / value / ownership.

After the iteration budget is exhausted, run a standard negamax alpha-beta
over the in-memory tree (moves visited in descending policy order, for
better cutoffs) to recover the root value and the best move.

The tree is currently a true tree (one parent per node). Transposition / DAG
storage is a planned optimization, not implemented here.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from alphabetago.board import BLACK, PASS, Board
from alphabetago.features import featurize
from alphabetago.nn import PolicyOwnershipNet


@dataclass
class Node:
    board: Board
    is_terminal: bool
    terminal_score: float | None = None
    expanded: bool = False
    policy: dict[int, float] | None = None
    value: float | None = None
    ownership: np.ndarray | None = None
    children: dict[int, Node] = field(default_factory=dict)

    @classmethod
    def from_board(cls, board: Board) -> Node:
        is_terminal = board.is_game_over
        terminal_score: float | None = None
        if is_terminal:
            tt = board.tromp_taylor_score()
            terminal_score = float(tt) if board.to_play == BLACK else float(-tt)
        return cls(
            board=board.copy(),
            is_terminal=is_terminal,
            terminal_score=terminal_score,
        )


def _ensure_child(parent: Node, move: int) -> Node:
    child = parent.children.get(move)
    if child is None:
        new_board = parent.board.copy()
        new_board.play(move)
        child = Node.from_board(new_board)
        parent.children[move] = child
    return child


def collect_unexpanded_leaves(root: Node, max_leaves: int) -> list[Node]:
    """Walk the tree best-first by accumulated `-log2(prob)`, collecting at
    most `max_leaves` distinct unexpanded non-terminal nodes.

    The heap holds *edges* (parent, move) rather than pre-built child nodes,
    so a child Board is materialized only when its edge is actually popped.
    """
    if root.is_terminal:
        return []
    if not root.expanded:
        return [root]

    # Heap entry: (accumulated_cost, tiebreaker, parent, move).
    heap: list[tuple[float, int, Node, int]] = []
    counter = 0
    for move, prob in root.policy.items():
        if prob > 0.0:
            heapq.heappush(heap, (-math.log2(prob), counter, root, move))
            counter += 1

    leaves: list[Node] = []
    while heap and len(leaves) < max_leaves:
        cost, _, parent, move = heapq.heappop(heap)
        child = _ensure_child(parent, move)
        if child.is_terminal:
            continue
        if not child.expanded:
            leaves.append(child)
            continue
        for m, p in child.policy.items():
            if p > 0.0:
                heapq.heappush(heap, (cost - math.log2(p), counter, child, m))
                counter += 1
    return leaves


def _build_policy_dict(board: Board, logits: np.ndarray) -> dict[int, float]:
    """Softmax `logits` over the side-to-play's legal moves; return move -> prob."""
    n = board.size
    legal = board.legal_moves(include_pass=True)
    if not legal:
        return {}
    legal_indices: list[int] = []
    for m in legal:
        if m == PASS:
            legal_indices.append(n * n)
        else:
            r, c = board.coord(m)
            legal_indices.append(r * n + c)
    legal_logits = np.array([logits[i] for i in legal_indices])
    m_max = legal_logits.max()
    e = np.exp(legal_logits - m_max)
    probs = e / e.sum()
    return {m: float(p) for m, p in zip(legal, probs, strict=True)}


def expand_nodes(
    nodes: list[Node], model: PolicyOwnershipNet, device: torch.device
) -> None:
    """Batch-evaluate the given unexpanded nodes through the NN."""
    targets = [n for n in nodes if not n.expanded and not n.is_terminal]
    if not targets:
        return
    feats = np.stack([featurize(n.board) for n in targets])
    feats_t = torch.from_numpy(feats).float().to(device)
    model.eval()
    with torch.no_grad():
        policy_logits, ownership = model(feats_t)
    policy_logits_np = policy_logits.cpu().numpy()
    ownership_np = ownership.cpu().numpy()
    for i, node in enumerate(targets):
        node.policy = _build_policy_dict(node.board, policy_logits_np[i])
        node.ownership = ownership_np[i]
        node.value = float(ownership_np[i].sum())
        node.expanded = True


def alpha_beta(node: Node, alpha: float, beta: float) -> tuple[float, int | None]:
    """Negamax alpha-beta over the in-memory tree. Returns (value, best_move)
    from `node`'s side-to-play perspective. Moves visited in descending policy
    probability for tighter cutoffs."""
    if node.is_terminal:
        return node.terminal_score, None
    if not node.expanded:
        return 0.0, None
    if not node.children:
        return node.value, None

    best_v = -math.inf
    best_m: int | None = None
    moves_sorted = sorted(node.policy.items(), key=lambda kv: -kv[1])
    for move, _prob in moves_sorted:
        child = node.children.get(move)
        if child is None:
            continue
        v_child, _ = alpha_beta(child, -beta, -alpha)
        v = -v_child
        if v > best_v:
            best_v = v
            best_m = move
        if best_v > alpha:
            alpha = best_v
        if alpha >= beta:
            break
    if best_m is None:
        return node.value, None
    return best_v, best_m


def search(
    board: Board,
    model: PolicyOwnershipNet,
    device: torch.device,
    n_iterations: int = 8,
    leaves_per_batch: int = 128,
) -> tuple[Node, int | None, float]:
    """Run the iterative search and return (root, best_move, root_value).

    `root_value` is from the side-to-play's perspective at the root.
    """
    root = Node.from_board(board)
    if root.is_terminal:
        return root, None, root.terminal_score
    expand_nodes([root], model, device)
    for _ in range(n_iterations):
        leaves = collect_unexpanded_leaves(root, leaves_per_batch)
        if not leaves:
            break
        expand_nodes(leaves, model, device)
    value, move = alpha_beta(root, -math.inf, math.inf)
    return root, move, value


def count_nodes(root: Node) -> tuple[int, int]:
    """Return (total, expanded) node counts in the tree rooted at `root`."""
    total = 0
    expanded = 0
    stack = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        total += 1
        if node.expanded:
            expanded += 1
        stack.extend(node.children.values())
    return total, expanded
