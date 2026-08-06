"""grid_navigator: ledge-aware planning + live navigation over a RAM grid.

plan_path_grid is a pure A* over a GridSnapshot with a faithful 2-tile jump
model: a one-way ledge is traversable only in its arrow direction, so ledges are
strictly one-way by construction (no false walls, no bump-learning). navigate_grid
drives the emulator using that plan, keeping a per-run transient-block set so a
live NPC on a static-grid FREE tile degrades to a detour, not a hang.

This module also owns the live-nav primitives that outlive the deleted bump-nav
(snapshot_settled, handle_battle_interruption, probe_step, resolve_move, timing/
key constants); grid_explorer reuses them. Emerald (BPEF) only.
"""
from __future__ import annotations

import heapq

from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# Grid convention: x grows right, y grows down. up decreases y.
DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# The ledge tile that a given direction may descend through.
_LEDGE_FOR: dict[str, TileKind] = {
    "up": TileKind.LEDGE_UP,
    "down": TileKind.LEDGE_DOWN,
    "left": TileKind.LEDGE_LEFT,
    "right": TileKind.LEDGE_RIGHT,
}

_STANDABLE: frozenset[TileKind] = frozenset({TileKind.FREE, TileKind.GRASS})


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_path_grid(
    grid: GridSnapshot,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[tuple[int, int], str]] | None = None,
) -> list[str] | None:
    """A* over a GridSnapshot; list of directions start->goal, or None.

    Nodes are only standable tiles (FREE/GRASS); LEDGE_*/WALL are never nodes.
    From node C in direction d (delta D): adjacent FREE/GRASS -> normal edge cost
    1; adjacent LEDGE_d with a FREE/GRASS landing at C+2D -> directed jump edge
    C->C+2D cost 1; otherwise blocked. `blocked` is an optional set of directed
    edges (cell, direction) to skip (the live navigator's transient NPC-avoidance
    set). Bounded by the finite grid (node set is width*height).
    """
    if start == goal:
        return []
    skip = blocked if blocked is not None else set()

    open_heap: list[tuple[int, tuple[int, int]]] = [(_manhattan(start, goal), start)]
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        for direction in DIRECTIONS:
            if (current, direction) in skip:
                continue
            neighbour = _edge_target(grid, current, direction)
            if neighbour is None:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbour, 1 << 30):
                g_score[neighbour] = tentative
                came_from[neighbour] = (current, direction)
                f_score = tentative + _manhattan(neighbour, goal)
                heapq.heappush(open_heap, (f_score, neighbour))
    return None


def _edge_target(
    grid: GridSnapshot, cell: tuple[int, int], direction: str
) -> tuple[int, int] | None:
    """The standable cell reached from `cell` going `direction`, or None.

    Normal step onto FREE/GRASS, or a one-tile ledge jump landing on FREE/GRASS.
    """
    dx, dy = DELTAS[direction]
    adj = (cell[0] + dx, cell[1] + dy)
    adj_kind = grid.classify_at(*adj)
    if adj_kind in _STANDABLE:
        return adj
    if adj_kind is _LEDGE_FOR[direction]:
        landing = (cell[0] + 2 * dx, cell[1] + 2 * dy)
        if grid.classify_at(*landing) in _STANDABLE:
            return landing
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
