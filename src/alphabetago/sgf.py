"""Minimal SGF parser for KataGo training-game SGFs.

We only care about: SZ (board size), AB/AW (setup stones), the move sequence
(;B[..]/;W[..] including pass = empty brackets), and RE (final result).
Comment fields, engine-eval annotations, etc. are ignored. The parser is
deliberately regex-based and lenient — KataGo SGFs are uniform enough that
a real SGF tree walker is overkill.

`sgf_to_game_record` plays the parsed game through a fresh `Board`,
including any setup stones, and returns a `GameRecord`. If a move turns out
to be illegal under our rules (e.g., a suicide play in a sui1 game, or a
positional-superko violation that the source ruleset allowed), the function
returns `None` — caller should drop that game.
"""

from __future__ import annotations

import re
from pathlib import Path

from alphabetago.board import BLACK, PASS, WHITE, Board, IllegalMoveError
from alphabetago.selfplay import GameRecord

_SZ_RE = re.compile(r"SZ\[(\d+)\]")
# AB[xx][yy]... or AW[xx][yy]... (setup stones; KataGo uses these for
# random-position-style game starts as well as proper handicap).
_SETUP_RE = re.compile(r"A([BW])((?:\[[a-z]*\])+)")
_BRACKETS_RE = re.compile(r"\[([a-z]*)\]")
# Moves: ;B[..] or ;W[..]. Note the leading semicolon distinguishes this
# from the AB/AW setup form.
_MOVE_RE = re.compile(r";([BW])\[([a-z]*)\]")
_RESULT_RE = re.compile(r"RE\[([^\]]*)\]")


def _sgf_coord(coord: str, size: int) -> int:
    """Convert an SGF coord like 'fc' to a (row, col) flat-board index in
    [0, size*size), or `PASS` for empty / off-board (e.g. legacy 'tt')."""
    if not coord:
        return PASS
    if len(coord) != 2:
        return PASS
    c = ord(coord[0]) - ord("a")
    r = ord(coord[1]) - ord("a")
    if not (0 <= r < size and 0 <= c < size):
        return PASS
    return r * size + c


def parse_sgf(text: str) -> dict:
    """Return {'size', 'setup_black', 'setup_white', 'moves', 'result'}.

    `setup_black`/`setup_white` are lists of (row, col) board points.
    `moves` is a list of ('B'|'W', point_or_PASS) pairs. `result` is the
    raw RE[] string (e.g. 'B+1.5', 'W+R', 'Draw', 'Void')."""
    size_m = _SZ_RE.search(text)
    size = int(size_m.group(1)) if size_m else 19

    setup_black: list[int] = []
    setup_white: list[int] = []
    for m in _SETUP_RE.finditer(text):
        target = setup_black if m.group(1) == "B" else setup_white
        for coord in _BRACKETS_RE.findall(m.group(2)):
            p = _sgf_coord(coord, size)
            if p != PASS:
                target.append(p)

    moves: list[tuple[str, int]] = []
    for m in _MOVE_RE.finditer(text):
        moves.append((m.group(1), _sgf_coord(m.group(2), size)))

    result_m = _RESULT_RE.search(text)
    result = result_m.group(1) if result_m else ""

    return {
        "size": size,
        "setup_black": setup_black,
        "setup_white": setup_white,
        "moves": moves,
        "result": result,
    }


def sgf_to_game_record(text: str, max_size: int | None = None) -> GameRecord | None:
    """Parse and replay an SGF; return a GameRecord or None if the game can't
    be replayed cleanly under our rules.

    `max_size`: if set and the SGF's board size differs, returns None.
    """
    parsed = parse_sgf(text)
    n = parsed["size"]
    if max_size is not None and n != max_size:
        return None

    board = Board(size=n)

    # SGF point convention here: we stored (row * size + col). Convert to
    # the Board's internal mailbox index via Board.point.
    def to_mailbox(flat: int) -> int:
        if flat == PASS:
            return PASS
        return board.point(flat // n, flat % n)

    try:
        for p in parsed["setup_black"]:
            board.setup(BLACK, to_mailbox(p))
        for p in parsed["setup_white"]:
            board.setup(WHITE, to_mailbox(p))
    except (IllegalMoveError, ValueError):
        return None

    moves: list[int] = []
    # Determine the side to move at the start of the move sequence. SGF
    # convention is that a node with B[] sets it to black having played;
    # we honor whatever color SGF asserts move-by-move, and skip the move
    # if it doesn't match whose turn it is according to our engine.
    for color_letter, point in parsed["moves"]:
        sgf_color = BLACK if color_letter == "B" else WHITE
        if sgf_color != board.to_play:
            # Some KataGo SGFs replay color out of order (rare, mostly in
            # setup-position games where black moves first regardless of
            # how setup left the count). Force our engine to honor the
            # SGF's assertion by playing a synthetic PASS to flip turn.
            if board.is_legal(PASS):
                board.play(PASS)
                moves.append(PASS)
            else:
                return None
        mailbox = to_mailbox(point)
        if not board.is_legal(mailbox):
            return None
        board.play(mailbox)
        moves.append(mailbox)
        if board.is_game_over:
            break

    return GameRecord(
        size=n,
        moves=moves,
        final_score=board.tromp_taylor_score(),
        final_ownership=board.ownership(),
    )


def load_sgf_file(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")
