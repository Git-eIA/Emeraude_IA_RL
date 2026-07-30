"""world_surveyor: chart the reachable overworld map-by-map.

survey_world starts wherever the player stands and repeatedly travels to a
not-yet-surveyed map and surveys it (travel_to + map_map), discovering new maps
through the reversible border portals map_map records. Overworld only: building
warps (non-reversible) are never followed. Log-and-continue: a failed leg is
recorded in the SurveyReport and the sweep goes on. No training, no reward,
no Strategist, no fighting. Emerald (BPEF) only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from env.live_navigator import snapshot_settled
from env.local_navigator import WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory, Portal
from env.map_traveler import travel_to


@dataclass(frozen=True)
class SurveyReport:
    surveyed: tuple[tuple[int, int], ...]              # maps charted, in visit order
    failed: tuple[tuple[tuple[int, int], str], ...]    # (map_id, reason)


def survey_world(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    max_maps: int = 50,
) -> SurveyReport:
    """Sweep the reachable overworld with an iterative, bounded BFS.

    Returns a SurveyReport listing every surveyed map and every failed leg with
    a reason ("travel:<outcome>" or "map:<result>"). Bounded by max_maps
    (code-safety rule #2); BFS is iterative, no recursion.
    """
    start = _current_map(reader)
    if start is None:
        return SurveyReport((), (("unknown", "no_start"),))

    pending: deque[tuple[int, int]] = deque([start])
    queued: set[tuple[int, int]] = {start}
    surveyed: list[tuple[int, int]] = []
    failed: list[tuple[tuple[int, int], str]] = []

    for _ in range(max_maps):
        if not pending:
            break
        target = pending.popleft()

        here = _current_map(reader)
        if here != target:
            outcome = travel_to(
                emulator, reader, memory, wallmap,
                target, _entry_cell(memory, target),
            )
            if outcome != "arrived":
                failed.append((target, f"travel:{outcome}"))
                continue

        result = map_map(emulator, reader, memory, wallmap, target)
        if result in ("left_map", "budget_exhausted"):
            failed.append((target, f"map:{result}"))
        surveyed.append(target)

        for portal in _overworld_portals(memory, target):
            nxt = portal.to_map
            if nxt not in queued and nxt not in surveyed:
                queued.add(nxt)
                pending.append(nxt)

    return SurveyReport(tuple(surveyed), tuple(failed))


def _current_map(reader: Any) -> tuple[int, int] | None:
    snap = snapshot_settled(reader)
    return None if snap is None else snap.map_id


def _overworld_portals(memory: MapMemory, map_id: tuple[int, int]) -> list[Portal]:
    """Outgoing portals of map_id that are reversible overworld borders."""
    return [p for p in memory.outgoing_portals(map_id) if p.reversible]


def _entry_cell(memory: MapMemory, target: tuple[int, int]) -> tuple[int, int]:
    """The cell we land on when entering target, from any recorded portal to it.

    Safe to index [0]: target was enqueued only because a portal to it exists.
    """
    return memory.incoming_portals(target)[0].to_cell
