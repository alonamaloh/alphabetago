"""Go board with incremental chain and liberty tracking.

Internal representation: a 1-D byte array of length (size + 2)^2 with a one-cell
sentinel border of OFFBOARD values. On-board points are addressed by 1-D index;
the constant `PASS` represents a pass move.

Chains are tracked with a union-find structure. Per-root metadata stores the
chain's stone count and the set of its liberties. Both are updated incrementally
on every move.
"""

from __future__ import annotations

import random

EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2
OFFBOARD: int = 3

PASS: int = -1


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class IllegalMoveError(ValueError):
    pass


_ZOBRIST_CACHE: dict[int, tuple[list[int], list[int]]] = {}


def _get_zobrist(stride: int) -> tuple[list[int], list[int]]:
    """Return (black_table, white_table); each is a list of 64-bit hash values
    indexed by 1-D mailbox point. Cached per stride so all boards of a given
    size share the same tables (and therefore comparable hashes)."""
    table = _ZOBRIST_CACHE.get(stride)
    if table is None:
        rng = random.Random(f"alphabetago-zobrist-{stride}")
        nn = stride * stride
        table = (
            [rng.getrandbits(64) for _ in range(nn)],
            [rng.getrandbits(64) for _ in range(nn)],
        )
        _ZOBRIST_CACHE[stride] = table
    return table


class Board:
    EMPTY = EMPTY
    BLACK = BLACK
    WHITE = WHITE
    OFFBOARD = OFFBOARD
    PASS = PASS

    def __init__(self, size: int = 9):
        if size < 2:
            raise ValueError(f"size must be >= 2 (got {size})")
        self._n = size
        self._stride = size + 2
        nn = self._stride * self._stride
        self._cells = bytearray([OFFBOARD] * nn)
        for r in range(size):
            for c in range(size):
                self._cells[self._point(r, c)] = EMPTY
        self._parent: list[int] = list(range(nn))
        self._chain_stones: list[int] = [0] * nn
        self._chain_liberties: list[set[int]] = [set() for _ in range(nn)]
        self._zobrist_black, self._zobrist_white = _get_zobrist(self._stride)
        self._hash: int = 0
        self._history: set[int] = {0}
        self._to_play: int = BLACK
        self._consecutive_passes: int = 0
        self._move_number: int = 0
        self._last_move: int | None = None
        self._last_captured_count: int = 0

    # ------------------------------------------------------------------ basic

    @property
    def size(self) -> int:
        return self._n

    @property
    def to_play(self) -> int:
        return self._to_play

    @property
    def move_number(self) -> int:
        return self._move_number

    @property
    def is_game_over(self) -> bool:
        return self._consecutive_passes >= 2

    @property
    def position_hash(self) -> int:
        return self._hash

    @property
    def last_move(self) -> int | None:
        return self._last_move

    @property
    def last_captured_count(self) -> int:
        return self._last_captured_count

    # ------------------------------------------------------------ coordinates

    def _point(self, row: int, col: int) -> int:
        return (row + 1) * self._stride + (col + 1)

    def point(self, row: int, col: int) -> int:
        if not (0 <= row < self._n and 0 <= col < self._n):
            raise ValueError(f"({row}, {col}) is off the board (size {self._n})")
        return self._point(row, col)

    def coord(self, point: int) -> tuple[int, int]:
        return point // self._stride - 1, point % self._stride - 1

    def _neighbors(self, point: int) -> tuple[int, int, int, int]:
        s = self._stride
        return (point - s, point - 1, point + 1, point + s)

    def on_board_points(self) -> list[int]:
        return [self._point(r, c) for r in range(self._n) for c in range(self._n)]

    # ------------------------------------------------------------------ stone

    def stone_at(self, point: int) -> int:
        v = self._cells[point]
        return v if v != OFFBOARD else EMPTY

    def chain_size_at(self, point: int) -> int:
        if self._cells[point] not in (BLACK, WHITE):
            return 0
        return self._chain_stones[self._find(point)]

    def chain_liberties_at(self, point: int) -> int:
        if self._cells[point] not in (BLACK, WHITE):
            return 0
        return len(self._chain_liberties[self._find(point)])

    # ------------------------------------------------------------- union-find

    def _find(self, point: int) -> int:
        root = point
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[point] != root:
            self._parent[point], point = root, self._parent[point]
        return root

    def _union(self, a: int, b: int) -> None:
        ra = self._find(a)
        rb = self._find(b)
        if ra == rb:
            return
        if self._chain_stones[ra] < self._chain_stones[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._chain_stones[ra] += self._chain_stones[rb]
        self._chain_liberties[ra] |= self._chain_liberties[rb]
        self._chain_liberties[rb] = set()
        self._chain_stones[rb] = 0

    # --------------------------------------------------------------- legality

    def _zobrist_for(self, color: int) -> list[int]:
        return self._zobrist_black if color == BLACK else self._zobrist_white

    def _stones_in_chain(self, seed: int) -> list[int]:
        """Flood-fill the same-colored stones connected to `seed`."""
        color = self._cells[seed]
        visited: set[int] = set()
        stack = [seed]
        out: list[int] = []
        while stack:
            p = stack.pop()
            if p in visited or self._cells[p] != color:
                continue
            visited.add(p)
            out.append(p)
            for n in self._neighbors(p):
                if n not in visited:
                    stack.append(n)
        return out

    def _tentative_hash_after_move(self, point: int) -> int:
        """Hypothetical position hash if `to_play` plays at `point`. Includes
        the effect of any captures the move would cause. Does not mutate state.

        Assumes the move is otherwise legal at the suicide-rule level (the
        caller has already verified it).
        """
        me = self._to_play
        opp = opponent(me)
        z_me = self._zobrist_for(me)
        z_opp = self._zobrist_for(opp)
        h = self._hash ^ z_me[point]
        captured_roots: set[int] = set()
        for n in self._neighbors(point):
            if self._cells[n] != opp:
                continue
            root = self._find(n)
            if root in captured_roots:
                continue
            if self._chain_liberties[root] == {point}:
                captured_roots.add(root)
                for s in self._stones_in_chain(n):
                    h ^= z_opp[s]
        return h

    def is_legal(self, point: int) -> bool:
        if self.is_game_over:
            return False
        if point == PASS:
            return True
        if not (0 <= point < len(self._cells)):
            return False
        if self._cells[point] != EMPTY:
            return False
        # Suicide check: at least one of (empty neighbor, friendly chain with
        # spare liberty, opponent chain reduced to zero liberties) must hold.
        my_color = self._to_play
        opp = opponent(my_color)
        not_suicide = False
        for n in self._neighbors(point):
            v = self._cells[n]
            if v == EMPTY:
                not_suicide = True
                break
            if v == my_color and len(self._chain_liberties[self._find(n)]) > 1:
                not_suicide = True
                break
            if v == opp and len(self._chain_liberties[self._find(n)]) == 1:
                not_suicide = True
                break
        if not not_suicide:
            return False
        # Positional superko: the resulting board state must not match any
        # previously seen one.
        return self._tentative_hash_after_move(point) not in self._history

    def legal_moves(self, include_pass: bool = True) -> list[int]:
        moves = [p for p in self.on_board_points() if self.is_legal(p)]
        if include_pass and not self.is_game_over:
            moves.append(PASS)
        return moves

    # ---------------------------------------------------------------- play

    def play(self, point: int) -> None:
        if not self.is_legal(point):
            raise IllegalMoveError(f"illegal move at point {point}")
        self._move_number += 1
        if point == PASS:
            self._consecutive_passes += 1
            self._last_move = None
            self._last_captured_count = 0
            self._to_play = opponent(self._to_play)
            return

        self._consecutive_passes = 0
        my_color = self._to_play
        opp = opponent(my_color)
        z_me = self._zobrist_for(my_color)
        z_opp = self._zobrist_for(opp)

        # Step 1: remove `point` as a liberty from neighboring chains; capture
        # any opponent chain reduced to zero liberties.
        captured_count = 0
        for n in self._neighbors(point):
            v = self._cells[n]
            if v == opp:
                root = self._find(n)
                self._chain_liberties[root].discard(point)
                if not self._chain_liberties[root]:
                    captured_count += self._capture_chain(n, z_opp)
            elif v == my_color:
                self._chain_liberties[self._find(n)].discard(point)

        # Step 2: place our stone, initialize singleton chain.
        self._cells[point] = my_color
        self._parent[point] = point
        self._chain_stones[point] = 1
        self._chain_liberties[point] = {
            m for m in self._neighbors(point) if self._cells[m] == EMPTY
        }
        self._hash ^= z_me[point]

        # Step 3: merge with adjacent friendly chains.
        for n in self._neighbors(point):
            if self._cells[n] == my_color:
                self._union(point, n)

        self._last_move = point
        self._last_captured_count = captured_count
        self._to_play = opp
        self._history.add(self._hash)

    def _capture_chain(self, seed: int, zobrist_table: list[int]) -> int:
        root = self._find(seed)
        stones = self._stones_in_chain(seed)
        for s in stones:
            self._cells[s] = EMPTY
            self._hash ^= zobrist_table[s]
        for s in stones:
            for m in self._neighbors(s):
                if self._cells[m] in (BLACK, WHITE):
                    self._chain_liberties[self._find(m)].add(s)
        self._chain_stones[root] = 0
        self._chain_liberties[root] = set()
        return len(stones)

    # --------------------------------------------------------------- scoring

    def tromp_taylor_score(self) -> int:
        """Area score from black's perspective (positive = black ahead). No komi."""
        black, white = self._area_counts()
        return black - white

    def ownership(self) -> dict[int, int]:
        """Map each on-board point to BLACK, WHITE, or EMPTY (contested)."""
        owners: dict[int, int] = {}
        visited: set[int] = set()
        for p in self.on_board_points():
            v = self._cells[p]
            if v in (BLACK, WHITE):
                owners[p] = v
            elif p not in visited:
                region, region_owner = self._classify_region(p)
                visited |= region
                for r in region:
                    owners[r] = region_owner
        return owners

    def _area_counts(self) -> tuple[int, int]:
        black = 0
        white = 0
        visited: set[int] = set()
        for p in self.on_board_points():
            v = self._cells[p]
            if v == BLACK:
                black += 1
            elif v == WHITE:
                white += 1
            elif p not in visited:
                region, region_owner = self._classify_region(p)
                visited |= region
                if region_owner == BLACK:
                    black += len(region)
                elif region_owner == WHITE:
                    white += len(region)
        return black, white

    def _classify_region(self, start: int) -> tuple[set[int], int]:
        region: set[int] = set()
        seen: set[int] = set()
        stack = [start]
        touches_black = False
        touches_white = False
        while stack:
            q = stack.pop()
            if q in seen:
                continue
            seen.add(q)
            v = self._cells[q]
            if v == BLACK:
                touches_black = True
            elif v == WHITE:
                touches_white = True
            elif v == EMPTY:
                region.add(q)
                for m in self._neighbors(q):
                    if m not in seen:
                        stack.append(m)
        if touches_black and not touches_white:
            return region, BLACK
        if touches_white and not touches_black:
            return region, WHITE
        return region, EMPTY

    # -------------------------------------------------------------- eye check

    def is_eye_for(self, color: int, point: int) -> bool:
        """Heuristic eye-like check used by simple agents to avoid filling their
        own eyes. A point is considered an eye for `color` when all orthogonal
        neighbors are `color` (or off-board) and at most one diagonal neighbor
        is not `color` (zero on edges and corners).

        This is a heuristic — not a strict eye definition — but it is sufficient
        to keep random self-play terminating at sensible game lengths.
        """
        if self._cells[point] != EMPTY:
            return False
        is_edge = False
        for n in self._neighbors(point):
            v = self._cells[n]
            if v == OFFBOARD:
                is_edge = True
            elif v != color:
                return False
        s = self._stride
        diagonals = (point - s - 1, point - s + 1, point + s - 1, point + s + 1)
        bad = 0
        for d in diagonals:
            v = self._cells[d]
            if v == OFFBOARD:
                continue
            if v != color:
                bad += 1
        threshold = 0 if is_edge else 1
        return bad <= threshold

    # ----------------------------------------------------------------- copy

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b._n = self._n
        b._stride = self._stride
        b._cells = bytearray(self._cells)
        b._parent = self._parent[:]
        b._chain_stones = self._chain_stones[:]
        b._chain_liberties = [s.copy() for s in self._chain_liberties]
        b._zobrist_black = self._zobrist_black
        b._zobrist_white = self._zobrist_white
        b._hash = self._hash
        b._history = self._history.copy()
        b._to_play = self._to_play
        b._consecutive_passes = self._consecutive_passes
        b._move_number = self._move_number
        b._last_move = self._last_move
        b._last_captured_count = self._last_captured_count
        return b

    # ----------------------------------------------------------------- text

    def text(self) -> str:
        rows = []
        for r in range(self._n):
            chars = []
            for c in range(self._n):
                v = self._cells[self._point(r, c)]
                if v == EMPTY:
                    chars.append(".")
                elif v == BLACK:
                    chars.append("X")
                elif v == WHITE:
                    chars.append("O")
            rows.append("".join(chars))
        return "\n".join(rows)

    def __repr__(self) -> str:
        side = "B" if self._to_play == BLACK else "W"
        return (
            f"Board(size={self._n}, to_play={side}, move={self._move_number})\n"
            f"{self.text()}"
        )
