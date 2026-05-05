import math

import numpy as np
import torch

from alphabetago.board import PASS, WHITE, Board
from alphabetago.nn import PolicyOwnershipNet
from alphabetago.search import (
    Node,
    alpha_beta,
    collect_unexpanded_leaves,
    count_nodes,
    expand_nodes,
    search,
)


def _untrained_model(size: int = 9) -> PolicyOwnershipNet:
    torch.manual_seed(0)
    return PolicyOwnershipNet(board_size=size, channels=8, n_blocks=1)


def test_node_terminal_score_orientation():
    b = Board(size=5)
    b.play(b.point(2, 2))  # Black places one stone in the center.
    b.play(PASS)           # White passes.
    b.play(PASS)           # Black passes — game ends.
    # Tromp-Taylor score is +25 from black's perspective. After B-PASS-PASS,
    # the side to play is white, so terminal_score (W's view) is -25.
    node = Node.from_board(b)
    assert node.is_terminal
    assert b.to_play == WHITE
    assert node.terminal_score == -25.0


def test_root_unexpanded_collection():
    b = Board(size=5)
    root = Node.from_board(b)
    leaves = collect_unexpanded_leaves(root, max_leaves=8)
    assert leaves == [root]


def test_expand_populates_fields():
    b = Board(size=5)
    root = Node.from_board(b)
    model = _untrained_model(size=5)
    expand_nodes([root], model, torch.device("cpu"))
    assert root.expanded
    assert root.policy is not None
    # Legal-move-only normalization: probabilities sum to ~1.
    assert math.isclose(sum(root.policy.values()), 1.0, abs_tol=1e-5)
    # Value is sum of ownership predictions.
    assert root.value is not None
    assert isinstance(root.ownership, np.ndarray)
    assert root.ownership.shape == (5, 5)


def test_alpha_beta_terminal():
    b = Board(size=5)
    b.play(b.point(2, 2))
    b.play(PASS)
    b.play(PASS)
    node = Node.from_board(b)
    v, m = alpha_beta(node, -math.inf, math.inf)
    assert v == node.terminal_score
    assert m is None


def test_alpha_beta_unexpanded_returns_zero():
    b = Board(size=5)
    root = Node.from_board(b)
    v, m = alpha_beta(root, -math.inf, math.inf)
    assert v == 0.0
    assert m is None


def test_collect_after_root_expanded_returns_children():
    b = Board(size=5)
    root = Node.from_board(b)
    model = _untrained_model(size=5)
    expand_nodes([root], model, torch.device("cpu"))
    leaves = collect_unexpanded_leaves(root, max_leaves=4)
    assert len(leaves) == 4
    # All collected leaves are children of root (one move from root).
    for leaf in leaves:
        assert leaf is not root
        assert not leaf.expanded


def test_search_runs_end_to_end_on_5x5():
    b = Board(size=5)
    model = _untrained_model(size=5)
    root, move, value = search(
        b, model, torch.device("cpu"), n_iterations=3, leaves_per_batch=8
    )
    assert root.expanded
    # Some interior tree got built.
    total, expanded = count_nodes(root)
    assert total > 1
    assert expanded > 1
    # Best move is one of root's children's moves (or PASS).
    assert move in root.policy
    # Value is finite.
    assert math.isfinite(value)


def test_search_on_terminal_board():
    b = Board(size=5)
    b.play(PASS)
    b.play(PASS)
    model = _untrained_model(size=5)
    root, move, value = search(b, model, torch.device("cpu"))
    assert root.is_terminal
    assert move is None
    assert value == root.terminal_score
