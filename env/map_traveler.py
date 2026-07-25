"""map_traveler: walk the Explorer across maps, door to door.

travel_to chains P2 route planning (plan_route) with P3 step-1 intra-map
navigation (navigate_to): walk to each known portal cell, cross it, repeat until
the goal cell on the goal map is reached. Known territory only — an unknown door
on the route returns "unknown_route" rather than exploring (mapping mode is a
later step). No training, no reward. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from env.live_navigator import navigate_to
from env.local_navigator import DELTAS, WallMap
from env.map_memory import MapMemory
from env.route_planner import plan_route

SETTLE_TRIES = 4   # skip SaveBlock None frames when reading where we landed


def travel_to(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    goal_map: tuple[int, int],
    goal_cell: tuple[int, int],
    max_hops: int = 20,
) -> str:
    """Walk map-by-map to goal_cell on goal_map over known territory.

    Returns 'arrived' | 'unknown_route' | 'unreachable' | 'lost' | 'timeout'.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return navigate_to(emulator, reader, wallmap, goal_cell)

        route = plan_route(memory, here.map_id, goal_map)
        if route is None or len(route) < 2:
            return "unknown_route"
        next_map = route[1]
        crossing = memory.portal(here.map_id, next_map)
        if crossing is None:
            return "unknown_route"   # door not yet discovered (mapping is deferred)

        reached = navigate_to(emulator, reader, wallmap, crossing.from_cell)
        if reached in ("unreachable", "timeout"):
            return reached
        if reached == "left_map":
            continue   # already crossed a border on the way; re-plan from new map

        # On the door cell: press the crossing direction by targeting the off-map
        # neighbour, which transitions on the first press (and records the portal).
        dx, dy = DELTAS[crossing.direction]
        neighbour = (crossing.from_cell[0] + dx, crossing.from_cell[1] + dy)
        navigate_to(emulator, reader, wallmap, neighbour, memory=memory)

        landed = _snapshot_settled(reader)
        if landed is None or landed.map_id != next_map:
            return "lost"
    return "timeout"


def _snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap
