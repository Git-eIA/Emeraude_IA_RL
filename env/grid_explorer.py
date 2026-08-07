"""grid_explorer: probe a map's border tiles for portals over the RAM grid.

explore_grid replaces map_map. With the RAM grid the terrain (walls, grass,
ledges) is known the instant the map loads, so nothing about geometry needs
probing. The grid holds no warp destinations, so the only thing left to discover
is portals: explore_grid reads + remembers the grid once, routes to each reachable
border FREE/GRASS cell, and steps outward off the map edge. A transition records
a portal (with a step-back reversibility check); a blocked step means that edge is
not a portal. complete = every reachable border candidate tested. The RAM grid
kills the map_map thrash: geometry is never re-probed and blocked edges are never
re-proposed. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from env.grid_navigator import (
    DELTAS,
    handle_battle_interruption,
    navigate_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory

_STANDABLE = (TileKind.FREE, TileKind.GRASS)


def explore_grid(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    target_map: tuple[int, int],
    max_steps: int = 2000,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Probe `target_map`'s reachable border tiles for portals.

    Returns:
      "complete"            — every reachable border candidate tested
      "budget_exhausted"    — hit max_steps before candidates emptied
      "left_map"            — crossed a non-reversible border and could not return
      "battle_interrupted" / "battle_lost" / "battle_timeout"
    """
    here = snapshot_settled(reader)
    if here is None or here.map_id != target_map:
        return "left_map"
    snap = GridSnapshot.from_reader(reader.grid_reader, target_map)
    if snap is None:
        return "budget_exhausted"
    memory.remember_grid(snap)

    candidates = _border_candidates(snap)
    tested: set[tuple[tuple[int, int], str]] = set()
    steps = 0
    for cell, direction in candidates:
        if steps >= max_steps:
            return "budget_exhausted"
        steps += 1
        if (cell, direction) in tested:
            continue
        tested.add((cell, direction))

        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle

        arrived = navigate_grid(
            emulator, reader, cell, memory=memory,
            move_type_fn=move_type_fn, predict=predict,
        )
        if arrived in ("battle_lost", "battle_timeout", "battle_interrupted"):
            return arrived
        if arrived == "left_map":
            return "left_map"
        if arrived != "arrived":
            continue   # unreachable/timeout candidate: skip, no re-probe

        before = snapshot_settled(reader)
        if before is None or before.pos != cell:
            continue
        outcome = probe_step(emulator, reader, before, direction)
        if outcome != "transition":
            continue   # not a portal (blocked / no-op)
        landed = snapshot_settled(reader)
        if landed is None:
            return "left_map"
        probe_step(emulator, reader, landed, _opposite(direction))
        returned = snapshot_settled(reader)
        reversible = returned is not None and returned.map_id == target_map
        memory.record_portal(
            target_map, cell, direction, landed.map_id, reversible, landed.pos
        )
        if not reversible:
            return "left_map"
    return "complete"


_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _opposite(direction: str) -> str:
    return _OPPOSITE[direction]


def _border_candidates(
    snap: GridSnapshot,
) -> list[tuple[tuple[int, int], str]]:
    """Every standable border cell paired with the outward direction off its edge."""
    out: list[tuple[tuple[int, int], str]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            for direction, (dx, dy) in DELTAS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < snap.width and 0 <= ny < snap.height):
                    out.append(((x, y), direction))
    return out
