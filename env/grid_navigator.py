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
from typing import Any

from emulator import buttons
from env.battle_player import play_battle
from env.encounter_detector import EncounterWatcher
from env.grid_snapshot import GridSnapshot
from env.heal_detector import HealWatcher
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory, WorldEvent

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


_DIRECTION_KEYS: dict[str, int] = {
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
}

STEP_FRAMES = 24      # hold a d-pad key ~0.4 s: one walking tile
RELEASE_FRAMES = 8    # idle after each press so the GBA doesn't fuse presses
TURN_RETRIES = 2      # a first press may only turn; retry to tell turn from wall
SETTLE_TRIES = 4      # re-read snapshot to skip SaveBlock None frames
BATTLE_TRANSITION_SETTLE = 8   # bounded idle steps to wait out a battle intro/fade


def snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap


def resolve_move(before: Any, after: Any) -> str:
    """Classify one attempted step: 'moved' | 'blocked' | 'transition'."""
    if before.map_id != after.map_id:
        return "transition"
    if before.pos != after.pos:
        return "moved"
    return "blocked"


def handle_battle_interruption(
    emulator: Any, reader: Any, move_type_fn: Any, predict: Any
) -> str | None:
    """Play any live wild battle and return only in overworld control.

    Gating on battle_starting() (flags set, no terminal outcome) catches the intro
    window where in_battle() is still False because the opponent has not populated.
    We idle until in_battle() confirms, play the battle, then idle out the end-fade
    so the caller never presses into a frozen game.

    None when there is no battle (or it was won) so the caller resumes; a terminal
    outcome otherwise: "battle_interrupted" (no Fighter), "battle_lost",
    "battle_timeout".
    """
    if not reader.battle_starting():
        return None
    for _ in range(BATTLE_TRANSITION_SETTLE):
        if reader.in_battle():
            break
        emulator.step(0, RELEASE_FRAMES)
    if not reader.in_battle():
        return None   # flags set but never became a real battle: not ours
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result != "won":
        return "battle_lost" if result == "lost" else "battle_timeout"
    for _ in range(BATTLE_TRANSITION_SETTLE):
        if not reader.battle_starting() and not reader.in_battle():
            break
        emulator.step(0, RELEASE_FRAMES)
    return None


def probe_step(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    """Press `direction`, retrying so a first-press turn isn't read as a wall."""
    outcome = "blocked"
    for _ in range(TURN_RETRIES):
        emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
        emulator.step(0, RELEASE_FRAMES)
        after = snapshot_settled(reader)
        if after is None:
            return "blocked"
        outcome = resolve_move(before, after)
        if outcome != "blocked":
            return outcome
    return outcome


def _adjacent_direction(
    pos: tuple[int, int], target: tuple[int, int]
) -> str | None:
    """Return the direction from pos to target if they are exactly one step apart."""
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    for direction, (ddx, ddy) in DELTAS.items():
        if dx == ddx and dy == ddy:
            return direction
    return None


def navigate_grid(
    emulator: Any,
    reader: Any,
    target: tuple[int, int],
    max_steps: int = 200,
    memory: MapMemory | None = None,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Walk the player to `target` on its current map, planning over the RAM grid.

    Zero false walls, one-way ledges respected by construction. A per-run
    transient-block set catches a planned press that fails to move (a live NPC on
    a static-grid FREE tile) and reroutes; it is per-call only, never persisted.
    When `memory` is given, the live grid is remembered, a border crossing is
    recorded as a portal, a heal tags a healing spot, and a battle tags has_grass.

    Returns 'arrived' | 'unreachable' | 'left_map' | 'timeout' | 'battle_lost' |
    'battle_timeout' | 'battle_interrupted'.
    """
    heal_watcher = HealWatcher()
    enc_watcher = EncounterWatcher()
    blocked: set[tuple[tuple[int, int], str]] = set()

    for _ in range(max_steps):
        before = snapshot_settled(reader)
        if before is None:
            emulator.step(0, RELEASE_FRAMES)   # relocating; idle and retry
            continue
        if memory is not None:
            if heal_watcher.observe(reader.party_hp()):
                memory.observe(before, WorldEvent(healed=True))
            if enc_watcher.observe(reader.in_battle()):
                memory.observe(before, WorldEvent(encounter_started=True))
        interruption = handle_battle_interruption(
            emulator, reader, move_type_fn, predict
        )
        if interruption is not None:
            return interruption
        if before.pos == target:
            return "arrived"

        snap = GridSnapshot.from_reader(reader.grid_reader, before.map_id)
        if snap is None:
            emulator.step(0, RELEASE_FRAMES)   # map not ready; idle and retry
            continue
        if memory is not None:
            memory.remember_grid(snap)

        path = plan_path_grid(snap, before.pos, target, blocked=blocked)
        if path is None:
            # Target may be an off-map neighbour (OOB): if the player is exactly
            # one step away and target is outside the grid bounds, attempt the
            # press — a border crossing returns "left_map" (travel_to portal leg).
            tx, ty = target
            target_oob = not (0 <= tx < snap.width and 0 <= ty < snap.height)
            adj_dir = _adjacent_direction(before.pos, target) if target_oob else None
            # If the OOB press already failed once (NPC on the border tile, wrong
            # facing), it is in `blocked`: a second attempt would only repeat the
            # dead press until max_steps. Treat a blocked OOB edge as unreachable.
            if adj_dir is None or (before.pos, adj_dir) in blocked:
                return "unreachable"
            direction = adj_dir
        else:
            direction = path[0]
        outcome = probe_step(emulator, reader, before, direction)
        if outcome == "transition":
            if memory is not None:
                landed = snapshot_settled(reader)
                if landed is not None:
                    memory.record_portal(
                        before.map_id, before.pos, direction, landed.map_id,
                        False, landed.pos,
                    )
            return "left_map"
        if outcome == "blocked":
            # A press can fail because a wild battle just started on this step
            # (grass), not because of a wall. Consume the battle and re-plan
            # instead of poisoning the tile as unreachable.
            if reader.battle_starting() or reader.in_battle():
                battle = handle_battle_interruption(
                    emulator, reader, move_type_fn, predict
                )
                if battle is not None:
                    return battle
                continue
            blocked.add((before.pos, direction))   # genuine wall / NPC
    return "timeout"
