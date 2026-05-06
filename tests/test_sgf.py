from alphabetago.board import BLACK, PASS, WHITE, Board
from alphabetago.sgf import parse_sgf, sgf_to_game_record


def test_parse_basic():
    sgf = "(;FF[4]GM[1]SZ[9]KM[7]RU[Tromp-Taylor]RE[B+5];B[ec];W[fd];B[];W[])"
    p = parse_sgf(sgf)
    assert p["size"] == 9
    assert p["setup_black"] == []
    assert p["setup_white"] == []
    assert p["result"] == "B+5"
    # ec = col 4, row 2 → flat index 2*9+4 = 22.
    # fd = col 5, row 3 → 3*9+5 = 32.
    assert p["moves"] == [("B", 22), ("W", 32), ("B", PASS), ("W", PASS)]


def test_parse_setup():
    sgf = "(;FF[4]GM[1]SZ[9]AB[dd][fd][ef]AW[df][dg][eg];B[ff])"
    p = parse_sgf(sgf)
    # dd = (3, 3) → 3*9+3 = 30; fd = (3, 5) = 32; ef = (5, 4) = 49.
    assert sorted(p["setup_black"]) == sorted([30, 32, 49])
    # df = (5, 3) = 48; dg = (6, 3) = 57; eg = (6, 4) = 58.
    assert sorted(p["setup_white"]) == sorted([48, 57, 58])
    assert p["moves"] == [("B", 50)]  # ff = (5,5) → 5*9+5 = 50.


def test_replay_simple_game():
    sgf = "(;FF[4]GM[1]SZ[9]KM[7]RE[B+81];B[ee];W[];B[])"
    # B(4,4) center, W passes, B passes. Tromp-Taylor area: 1 stone + 80
    # all-empty cells reaching only black = 81 black area, 0 white area.
    rec = sgf_to_game_record(sgf)
    assert rec is not None
    assert rec.size == 9
    assert rec.final_score == 81
    assert rec.moves[-2:] == [PASS, PASS]


def test_replay_with_setup():
    """A setup-prefixed game replays through the engine."""
    sgf = "(;FF[4]GM[1]SZ[9]AB[ee]AW[ff];B[];W[])"
    rec = sgf_to_game_record(sgf)
    assert rec is not None
    # No actual moves besides the two passes; setup placed two stones.
    # Tromp-Taylor area: B has 1 stone touching 2 territory, W has 1 stone
    # touching 2 territory, the bulk of empty cells are contested.
    # We don't pin the exact score; just verify the replay didn't fail.
    assert rec.final_score is not None


def test_replay_returns_none_on_illegal():
    """Suicide is illegal in our engine — a sui1 game with such a move drops."""
    # Setup white at (0,1) and (1,0). With both corner-neighbors of (0,0)
    # owned by white (each with > 1 liberty), B[aa] would be a suicide.
    sgf = "(;FF[4]GM[1]SZ[9]AW[ba][ab];B[aa])"
    rec = sgf_to_game_record(sgf)
    assert rec is None


def test_size_mismatch():
    sgf = "(;FF[4]GM[1]SZ[19];B[dd])"
    rec = sgf_to_game_record(sgf, max_size=9)
    assert rec is None


def test_board_setup_method():
    """Direct test of Board.setup."""
    b = Board(size=9)
    b.setup(BLACK, b.point(0, 0))
    b.setup(WHITE, b.point(8, 8))
    assert b.stone_at(b.point(0, 0)) == BLACK
    assert b.stone_at(b.point(8, 8)) == WHITE
    # to_play unchanged: still BLACK after setup.
    assert b.to_play == BLACK
    # Hash differs from empty board.
    assert b.position_hash != 0
