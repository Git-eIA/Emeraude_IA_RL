"""map_explorer: survey one Emerald map to exhaustion by frontier search.

map_map stands on a cell (now `reached`), looks at its four directions, and
treats any direction that is neither already `tried` nor a known wall as a
`frontier`. It repositions to the nearest frontier cell over KNOWN-walkable
cells only (BFS over `reached`, never plan_path — plan_path's optimistic grid
could route through an unknown door), then probes the one unknown edge:
moved -> new walkable cell, blocked -> a wall (recorded), transition -> a door
(recorded as a portal, then stepped back through the reversible border).
Reuses P2 (WallMap, DELTAS, OPPOSITE, DIRECTIONS) and P3 step-1 primitives
(probe_step, snapshot_settled, timing constants). Emerald (BPEF) only.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from env.live_navigator import RELEASE_FRAMES, probe_step, snapshot_settled
from env.local_navigator import DELTAS, DIRECTIONS, OPPOSITE, WallMap
from env.encounter_detector import EncounterWatcher
from env.map_memory import MapMemory, WorldEvent


def map_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    target_map: tuple[int, int],
    max_steps: int = 2000,
) -> str:
    """Survey `target_map` by frontier search: learn every wall and record
    every door as a portal. Known-walkable repositioning only.

    Returns:
      "complete"          — frontier exhausted; the map is fully known
      "budget_exhausted"  — hit max_steps before the frontier emptied
      "left_map"          — crossed a non-reversible door and could not return
    """
    reached: set[tuple[int, int]] = set()
    tried: set[tuple[tuple[int, int], str]] = set()
    enc_watcher = EncounterWatcher()

    for _ in range(max_steps):
        here = snapshot_settled(reader)
        if here is None:
            emulator.step(0, RELEASE_FRAMES)  # relocating; idle a beat and retry
            continue
        if here.map_id != target_map:
            return "left_map"
        reached.add(here.pos)
        if enc_watcher.observe(reader.in_battle()):
            memory.observe(here, WorldEvent(encounter_started=True))

        plan = _nearest_frontier(reached, tried, wallmap, target_map, here.pos)
        if plan is None:
            return "complete"
        route, cell, direction = plan

        if not _follow_route(emulator, reader, route):
            continue  # world surprised us; next snapshot re-grounds the survey

        before = snapshot_settled(reader)
        if before is None or before.pos != cell:
            continue  # repositioning landed off the frontier cell; re-loop

        outcome = probe_step(emulator, reader, before, direction)
        tried.add((cell, direction))
        if outcome == "moved":
            dx, dy = DELTAS[direction]
            neighbour = (cell[0] + dx, cell[1] + dy)
            tried.add((neighbour, OPPOSITE[direction]))
        elif outcome == "blocked":
            wallmap.block(target_map, cell, direction)
        elif outcome == "transition":
            landed = snapshot_settled(reader)
            if landed is None:
                return "left_map"  # relocating: cannot record or verify — bail safe
            # Step back through the door; a reversible border returns us to
            # target_map, a one-way warp does not. Record the portal with the
            # reversibility we just proved and the observed landing cell.
            probe_step(emulator, reader, landed, OPPOSITE[direction])
            returned = snapshot_settled(reader)
            reversible = returned is not None and returned.map_id == target_map
            memory.record_portal(
                target_map, cell, direction, landed.map_id, reversible, landed.pos
            )
            if not reversible:
                return "left_map"

    return "budget_exhausted"


def _follow_route(emulator: Any, reader: Any, route: list[str]) -> bool:
    """Press each known-walkable direction; True if every press moved us."""
    for direction in route:
        before = snapshot_settled(reader)
        if before is None:
            return False
        if probe_step(emulator, reader, before, direction) != "moved":
            return False
    return True


def _nearest_frontier(
    reached: set[tuple[int, int]],
    tried: set[tuple[tuple[int, int], str]],
    wallmap: WallMap,
    target_map: tuple[int, int],
    start: tuple[int, int],
) -> tuple[list[str], tuple[int, int], str] | None:
    """BFS over reached cells from `start`; return (route, cell, direction) for
    the nearest reached cell that still has an unexplored, non-walled direction,
    or None if the frontier is empty. Ties break by DIRECTIONS order.

    Terminates because `reached` is finite and `seen` prevents revisits.
    """
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen: set[tuple[int, int]] = {start}
    while queue:
        cell, route = queue.popleft()
        for direction in DIRECTIONS:
            if (cell, direction) in tried:
                continue
            if wallmap.is_blocked(target_map, cell, direction):
                continue
            return route, cell, direction
        for direction in DIRECTIONS:
            if wallmap.is_blocked(target_map, cell, direction):
                continue
            dx, dy = DELTAS[direction]
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in reached and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, route + [direction]))
    return None
