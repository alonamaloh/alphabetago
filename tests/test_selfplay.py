import random

from alphabetago.board import BLACK, EMPTY, PASS, WHITE, Board
from alphabetago.selfplay import (
    load_games,
    play_random_game,
    play_random_games_parallel,
    random_move,
    save_games,
)


def test_eye_detection_corner():
    b = Board(size=9)
    # Build a 1-point eye for B at (0, 0): B at (0,1), (1,0), (1,1).
    sequence = [(0, 1), (5, 5), (1, 0), (5, 6), (1, 1)]
    for m in sequence:
        b.play(b.point(*m))
    assert b.is_eye_for(BLACK, b.point(0, 0))
    assert not b.is_eye_for(WHITE, b.point(0, 0))


def test_eye_not_eye_when_diag_is_opponent_corner():
    b = Board(size=9)
    # Same shape but with a white stone on the only on-board diagonal — fails
    # the corner threshold (0 non-color diagonals allowed).
    b.play(b.point(0, 1))  # B
    b.play(b.point(1, 1))  # W
    b.play(b.point(1, 0))  # B
    assert not b.is_eye_for(BLACK, b.point(0, 0))


def test_random_move_avoids_own_eye():
    b = Board(size=9)
    sequence = [(0, 1), (5, 5), (1, 0), (5, 6), (1, 1)]
    for m in sequence:
        b.play(b.point(*m))
    b.play(PASS)  # let it be black's turn
    assert b.to_play == BLACK
    rng = random.Random(0)
    forbidden = b.point(0, 0)
    for _ in range(50):
        assert random_move(b, rng) != forbidden


def test_random_game_completes():
    rec = play_random_game(size=9, seed=42)
    # Random play with eye-avoidance always ends naturally with two passes.
    assert rec.moves[-2:] == [PASS, PASS]
    assert rec.size == 9


def test_score_consistent_with_ownership():
    rec = play_random_game(size=9, seed=42)
    n_b = sum(1 for v in rec.final_ownership.values() if v == BLACK)
    n_w = sum(1 for v in rec.final_ownership.values() if v == WHITE)
    assert rec.final_score == n_b - n_w


def test_random_game_5x5():
    rec = play_random_game(size=5, seed=7)
    assert rec.size == 5
    assert rec.moves[-2:] == [PASS, PASS]
    n_b = sum(1 for v in rec.final_ownership.values() if v == BLACK)
    n_w = sum(1 for v in rec.final_ownership.values() if v == WHITE)
    assert rec.final_score == n_b - n_w


def test_different_seeds_differ():
    rec1 = play_random_game(size=9, seed=1)
    rec2 = play_random_game(size=9, seed=2)
    assert rec1.moves != rec2.moves


def test_same_seed_reproducible():
    rec1 = play_random_game(size=9, seed=42)
    rec2 = play_random_game(size=9, seed=42)
    assert rec1.moves == rec2.moves
    assert rec1.final_score == rec2.final_score
    assert rec1.final_ownership == rec2.final_ownership


def test_game_record_winner():
    rec = play_random_game(size=9, seed=42)
    if rec.final_score > 0:
        assert rec.winner == BLACK
    elif rec.final_score < 0:
        assert rec.winner == WHITE
    else:
        assert rec.winner == EMPTY


def test_parallel_matches_serial():
    n = 8
    base = 100
    serial = [play_random_game(size=9, seed=base + i) for i in range(n)]
    parallel = play_random_games_parallel(
        n_games=n, size=9, base_seed=base, n_workers=4
    )
    for s, p in zip(serial, parallel, strict=True):
        assert s.moves == p.moves
        assert s.final_score == p.final_score


def test_save_and_load_games(tmp_path):
    games = [play_random_game(size=5, seed=i) for i in range(3)]
    path = tmp_path / "games.pkl"
    save_games(games, path)
    loaded = load_games(path)
    assert len(loaded) == 3
    for original, restored in zip(games, loaded, strict=True):
        assert original.moves == restored.moves
        assert original.final_score == restored.final_score
        assert original.final_ownership == restored.final_ownership
