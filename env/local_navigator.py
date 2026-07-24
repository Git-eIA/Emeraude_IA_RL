"""local_navigator: intra-map movement from observed collisions.

The Explorer learns which cell-to-cell moves are blocked by bumping into walls
(pressed a direction, position did not change -> wall). Those observations fill a
WallMap, and A* plans a path over the known-walkable grid, replanning whenever a
new wall is discovered. No tile-behavior probe, no emulator, no ROM. Emerald
(BPEF) only.
"""
from __future__ import annotations

import heapq

from env.world_reader import WorldSnapshot

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# Grid convention: x grows right, y grows down. up decreases y.
DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def resolve_move(before: WorldSnapshot, after: WorldSnapshot) -> str:
    """Classify one attempted step: 'moved' | 'blocked' | 'transition'."""
    if before.map_id != after.map_id:
        return "transition"
    if before.pos != after.pos:
        return "moved"
    return "blocked"


class WallMap:
    """Per-map set of blocked directed edges, learned from observed collisions.

    An edge never observed is assumed walkable (optimistic); A* replans when a
    real wall is later discovered. Blocking is bidirectional: a wall between two
    cells blocks the move both ways.
    """

    def __init__(self) -> None:
        # map_id -> set of (cell, direction) that are known blocked.
        self._blocked: dict[
            tuple[int, int], set[tuple[tuple[int, int], str]]
        ] = {}

    def block(
        self, map_id: tuple[int, int], cell: tuple[int, int], direction: str
    ) -> None:
        """Record a wall in `direction` from `cell`, and its mirror from the neighbour."""
        edges = self._blocked.setdefault(map_id, set())
        edges.add((cell, direction))
        dx, dy = DELTAS[direction]
        neighbour = (cell[0] + dx, cell[1] + dy)
        edges.add((neighbour, OPPOSITE[direction]))

    def is_blocked(
        self, map_id: tuple[int, int], cell: tuple[int, int], direction: str
    ) -> bool:
        """True only if this edge was observed blocked; unknown edges are open."""
        return (cell, direction) in self._blocked.get(map_id, ())


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_path(
    wallmap: WallMap,
    map_id: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[str] | None:
    """A* over the learned grid; list of directions from start to goal, or None.

    Unknown edges are treated as walkable (optimistic). The infinite grid is
    explored lazily; the Manhattan heuristic keeps A* focused toward the goal.
    """
    if start == goal:
        return []

    # Priority queue of (f_score, cell); came_from records (prev_cell, direction).
    open_heap: list[tuple[int, tuple[int, int]]] = [(_manhattan(start, goal), start)]
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        for direction in DIRECTIONS:
            if wallmap.is_blocked(map_id, current, direction):
                continue
            dx, dy = DELTAS[direction]
            neighbour = (current[0] + dx, current[1] + dy)
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbour, 1 << 30):
                g_score[neighbour] = tentative
                came_from[neighbour] = (current, direction)
                f_score = tentative + _manhattan(neighbour, goal)
                heapq.heappush(open_heap, (f_score, neighbour))
    return None


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]],
    goal: tuple[int, int],
) -> list[str]:
    directions: list[str] = []
    cell = goal
    while cell in came_from:
        prev, direction = came_from[cell]
        directions.append(direction)
        cell = prev
    directions.reverse()
    return directions
