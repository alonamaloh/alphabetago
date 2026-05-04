"""Go board with incremental chain and liberty tracking.

Internal representation: a 1-D byte array of length (size + 2)^2 with a one-cell
sentinel border of OFFBOARD values. On-board points are addressed by 1-D index;
the constant `PASS` represents a pass move.

Chains are tracked with a union-find structure. Per-root metadata stores the
chain's stone count and the set of its liberties. Both are updated incrementally
on every move.
"""

from __future__ import annotations

EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2
OFFBOARD: int = 3

PASS: int = -1


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class IllegalMoveError(ValueError):
    pass


class Board:
    EMPTY = EMPTY
    BLACK = BLACK
    WHITE = WHITE
    OFFBOARD = OFFBOARD
    PASS = PASS

    def __init__(self, size: int = 9, komi: float = 7.0):
        if size < 2:
            raise ValueError(f"size must be >= 2 (got {size})")
        self._n = size
        self._stride = size + 2
        self._komi = komi
        nn = self._stride * self._stride
        self._cells = bytearray([OFFBOARD] * nn)
        for r in range(size):
            for c in range(size):
                self._cells[self._point(r, c)] = EMPTY
        self._parent: list[int] = list(range(nn))
        self._chain_stones: list[int] = [0] * nn
        self._chain_liberties: list[set[int]] = [set() for _ in range(nn)]
        self._ko_point: int | None = None
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
    def komi(self) -> float:
        return self._komi

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
    def ko_point(self) -> int | None:
        return self._ko_point

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

    def is_legal(self, point: int) -> bool:
        if self.is_game_over:
            return False
        if point == PASS:
            return True
        if not (0 <= point < len(self._cells)):
            return False
        if self._cells[point] != EMPTY:
            return False
        if self._ko_point is not None and point == self._ko_point:
            return False
        my_color = self._to_play
        opp = opponent(my_color)
        for n in self._neighbors(point):
            v = self._cells[n]
            if v == EMPTY:
                return True
            if v == my_color and len(self._chain_liberties[self._find(n)]) > 1:
                return True
            if v == opp and len(self._chain_liberties[self._find(n)]) == 1:
                return True
        return False

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
            self._ko_point = None
            self._last_move = None
            self._last_captured_count = 0
            self._to_play = opponent(self._to_play)
            return

        self._consecutive_passes = 0
        my_color = self._to_play
        opp = opponent(my_color)

        # Step 1: remove `point` as a liberty from neighboring chains; capture
        # any opponent chain reduced to zero liberties.
        captured_count = 0
        for n in self._neighbors(point):
            v = self._cells[n]
            if v == opp:
                root = self._find(n)
                self._chain_liberties[root].discard(point)
                if not self._chain_liberties[root]:
                    captured_count += self._capture_chain(n)
            elif v == my_color:
                self._chain_liberties[self._find(n)].discard(point)

        # Step 2: place our stone, initialize singleton chain.
        self._cells[point] = my_color
        self._parent[point] = point
        self._chain_stones[point] = 1
        self._chain_liberties[point] = {
            m for m in self._neighbors(point) if self._cells[m] == EMPTY
        }

        # Step 3: merge with adjacent friendly chains.
        for n in self._neighbors(point):
            if self._cells[n] == my_color:
                self._union(point, n)

        # Step 4: ko bookkeeping. A simple-ko situation arises when our move
        # captured exactly one stone and our resulting chain has exactly one
        # stone with exactly one liberty — that liberty is the ko point.
        new_root = self._find(point)
        if (
            captured_count == 1
            and self._chain_stones[new_root] == 1
            and len(self._chain_liberties[new_root]) == 1
        ):
            (self._ko_point,) = self._chain_liberties[new_root]
        else:
            self._ko_point = None

        self._last_move = point
        self._last_captured_count = captured_count
        self._to_play = opp

    def _capture_chain(self, seed: int) -> int:
        color = self._cells[seed]
        root = self._find(seed)
        stones: list[int] = []
        visited: set[int] = set()
        stack = [seed]
        while stack:
            p = stack.pop()
            if p in visited or self._cells[p] != color:
                continue
            visited.add(p)
            stones.append(p)
            for n in self._neighbors(p):
                if n not in visited:
                    stack.append(n)
        for s in stones:
            self._cells[s] = EMPTY
        for s in stones:
            for m in self._neighbors(s):
                if self._cells[m] in (BLACK, WHITE):
                    self._chain_liberties[self._find(m)].add(s)
        self._chain_stones[root] = 0
        self._chain_liberties[root] = set()
        return len(stones)

    # --------------------------------------------------------------- scoring

    def tromp_taylor_score(self) -> float:
        """Area score from black's perspective (positive = black ahead)."""
        black, white = self._area_counts()
        return black - white - self._komi

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

    # ----------------------------------------------------------------- copy

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b._n = self._n
        b._stride = self._stride
        b._komi = self._komi
        b._cells = bytearray(self._cells)
        b._parent = self._parent[:]
        b._chain_stones = self._chain_stones[:]
        b._chain_liberties = [s.copy() for s in self._chain_liberties]
        b._ko_point = self._ko_point
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
