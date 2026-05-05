import pytest

from alphabetago.board import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    Board,
    IllegalMoveError,
    opponent,
)


def _play_seq(board, moves):
    """Apply a sequence of moves: each entry is a (row, col) tuple, an int point, or PASS."""
    for m in moves:
        if isinstance(m, tuple):
            board.play(board.point(*m))
        else:
            board.play(m)


def test_initial_state():
    b = Board(size=9)
    assert b.size == 9
    assert b.to_play == BLACK
    assert b.move_number == 0
    assert not b.is_game_over
    for p in b.on_board_points():
        assert b.stone_at(p) == EMPTY
    assert len(b.legal_moves()) == 9 * 9 + 1
    assert b.position_hash == 0


def test_coord_roundtrip():
    b = Board(size=9)
    for r in range(9):
        for c in range(9):
            assert b.coord(b.point(r, c)) == (r, c)


def test_alternation():
    b = Board(size=9)
    assert b.to_play == BLACK
    b.play(b.point(0, 0))
    assert b.to_play == WHITE
    b.play(b.point(0, 1))
    assert b.to_play == BLACK
    b.play(PASS)
    assert b.to_play == WHITE


def test_simple_play():
    b = Board(size=9)
    p = b.point(4, 4)
    b.play(p)
    assert b.stone_at(p) == BLACK
    assert b.chain_size_at(p) == 1
    assert b.chain_liberties_at(p) == 4
    assert b.move_number == 1
    assert b.last_move == p
    assert b.last_captured_count == 0


def test_play_existing_point_illegal():
    b = Board(size=9)
    p = b.point(4, 4)
    b.play(p)
    assert not b.is_legal(p)
    with pytest.raises(IllegalMoveError):
        b.play(p)


def test_off_board_point_rejected():
    b = Board(size=9)
    with pytest.raises(ValueError):
        b.point(-1, 0)
    with pytest.raises(ValueError):
        b.point(0, 9)


def test_corner_and_edge_liberties():
    b = Board(size=9)
    b.play(b.point(0, 0))  # corner: 2 libs
    assert b.chain_liberties_at(b.point(0, 0)) == 2
    b.play(b.point(0, 4))  # edge: 3 libs
    assert b.chain_liberties_at(b.point(0, 4)) == 3


def test_chain_merge():
    b = Board(size=9)
    _play_seq(b, [(4, 4), (0, 0), (4, 6), (0, 1)])
    assert b.chain_size_at(b.point(4, 4)) == 1
    assert b.chain_size_at(b.point(4, 6)) == 1
    b.play(b.point(4, 5))
    assert b.chain_size_at(b.point(4, 4)) == 3
    assert b.chain_size_at(b.point(4, 5)) == 3
    assert b.chain_size_at(b.point(4, 6)) == 3
    # Combined liberties of B at (4,4)-(4,5)-(4,6): 8 unique empty neighbors.
    assert b.chain_liberties_at(b.point(4, 5)) == 8


def test_capture_single_stone():
    b = Board(size=9)
    _play_seq(b, [(0, 1), (0, 0), (1, 0)])
    assert b.stone_at(b.point(0, 0)) == EMPTY
    assert b.last_captured_count == 1


def test_capture_multi_stone_chain():
    b = Board(size=9)
    _play_seq(b, [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2), (0, 3)])
    for c in range(3):
        assert b.stone_at(b.point(0, c)) == EMPTY
    assert b.last_captured_count == 3


def test_capture_releases_liberty_to_neighbor():
    """The capturing stone gains liberties from the freed cells."""
    b = Board(size=9)
    _play_seq(b, [(0, 1), (0, 0), (1, 0)])
    p = b.point(1, 0)
    assert b.stone_at(p) == BLACK
    # Empty neighbors of (1, 0) after capture: (0, 0) [freed], (1, 1), (2, 0).
    assert b.chain_liberties_at(p) == 3


def test_suicide_illegal():
    b = Board(size=9)
    _play_seq(b, [PASS, (0, 1), PASS, (1, 0)])
    # Black to play; (0, 0) has only opponent neighbors with > 1 liberty each.
    assert not b.is_legal(b.point(0, 0))
    assert b.is_legal(PASS)


def test_double_pass_ends_game():
    b = Board(size=9)
    assert not b.is_game_over
    b.play(PASS)
    assert not b.is_game_over
    b.play(PASS)
    assert b.is_game_over
    assert not b.is_legal(b.point(4, 4))
    assert not b.is_legal(PASS)


def test_pass_resets_after_real_move():
    b = Board(size=9)
    b.play(PASS)
    b.play(b.point(4, 4))
    b.play(PASS)
    assert not b.is_game_over


def test_ko():
    """Standard ko: black captures, white cannot immediately recapture."""
    b = Board(size=9)
    # Build:   . X O . .
    #          X O . O .
    #          . X O . .
    # Black plays (1, 2) and captures the white stone at (1, 1).
    _play_seq(
        b,
        [
            (0, 1),  # B
            (0, 2),  # W
            (1, 0),  # B
            (1, 1),  # W
            (2, 1),  # B
            (1, 3),  # W
            PASS,    # B (one extra so we have an even number of B/W moves)
            (2, 2),  # W
            (1, 2),  # B captures W (1, 1)
        ],
    )
    assert b.to_play == WHITE
    # Positional superko forbids the immediate recapture.
    assert not b.is_legal(b.point(1, 1))
    # After both sides play elsewhere, the position differs from the prior
    # one and the ko square is legal again.
    b.play(b.point(7, 7))
    b.play(b.point(7, 8))
    assert b.is_legal(b.point(1, 1))


def test_tromp_taylor_empty_board():
    b = Board(size=9)
    b.play(PASS)
    b.play(PASS)
    # No stones; the empty region touches no color → score = 0.
    assert b.tromp_taylor_score() == 0


def test_tromp_taylor_single_stone():
    b = Board(size=5)
    b.play(b.point(2, 2))
    b.play(PASS)
    b.play(PASS)
    assert b.is_game_over
    # 1 stone + 24 empty cells (all touching only B) = 25 black area.
    assert b.tromp_taylor_score() == 25


def test_tromp_taylor_two_colors_contested():
    b = Board(size=5)
    b.play(b.point(1, 2))  # B
    b.play(b.point(3, 2))  # W
    b.play(PASS)
    b.play(PASS)
    # Empty region touches both colors → contested, no territory. 1-1 = 0.
    assert b.tromp_taylor_score() == 0


def test_ownership_simple():
    b = Board(size=5)
    b.play(b.point(2, 2))
    b.play(PASS)
    b.play(PASS)
    own = b.ownership()
    for p in b.on_board_points():
        assert own[p] == BLACK


def test_copy_independence():
    b = Board(size=9)
    b.play(b.point(4, 4))
    b2 = b.copy()
    b2.play(b2.point(4, 5))
    assert b.stone_at(b.point(4, 5)) == EMPTY
    assert b2.stone_at(b2.point(4, 5)) == WHITE
    assert b.move_number == 1
    assert b2.move_number == 2
    b.play(b.point(0, 0))
    assert b2.stone_at(b2.point(0, 0)) == EMPTY


def test_opponent_helper():
    assert opponent(BLACK) == WHITE
    assert opponent(WHITE) == BLACK
