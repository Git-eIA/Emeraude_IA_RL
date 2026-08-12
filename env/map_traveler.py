"""map_traveler: walk the Explorer across maps, door to door.

travel_to chains P2 route planning (plan_route) with ledge-aware intra-map
navigation (navigate_grid): walk to each known portal cell, cross it, repeat
until the goal cell on the goal map is reached. Known territory only — an
unknown door on the route returns "unknown_route" rather than exploring (mapping
mode is a later step). No training, no reward. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from env.grid_navigator import (
    DELTAS,
    DIRECTION_KEYS,
    handle_battle_interruption,
    navigate_grid,
    plan_path_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory
from env.route_planner import plan_route

SETTLE_TRIES = 4   # skip SaveBlock None frames when reading where we landed
BATTLE_OUTCOMES = ("battle_lost", "battle_timeout", "battle_interrupted")


def travel_to(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    goal_map: tuple[int, int],
    goal_cell: tuple[int, int],
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Walk map-by-map to goal_cell on goal_map over known territory.

    Returns 'arrived' | 'unknown_route' | 'unreachable' | 'lost' | 'timeout'
    | 'battle_lost' | 'battle_timeout' | 'battle_interrupted'.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return navigate_grid(
                emulator, reader, goal_cell,
                move_type_fn=move_type_fn, predict=predict,
            )

        route = plan_route(memory, here.map_id, goal_map)
        if route is None or len(route) < 2:
            return "unknown_route"
        next_map = route[1]
        crossing = memory.portal(here.map_id, next_map)
        if crossing is None:
            return "unknown_route"   # door not yet discovered (mapping is deferred)

        reached = navigate_grid(
            emulator, reader, crossing.from_cell,
            move_type_fn=move_type_fn, predict=predict,
        )
        if reached in BATTLE_OUTCOMES:
            return reached
        if reached in ("unreachable", "timeout"):
            return reached
        if reached == "left_map":
            continue   # already crossed a border on the way; re-plan from new map

        # On the door cell: press the crossing direction by targeting the off-map
        # neighbour, which transitions on the first press (and records the portal).
        dx, dy = DELTAS[crossing.direction]
        neighbour = (crossing.from_cell[0] + dx, crossing.from_cell[1] + dy)
        crossed = navigate_grid(
            emulator, reader, neighbour, memory=memory,
            move_type_fn=move_type_fn, predict=predict,
        )
        if crossed in BATTLE_OUTCOMES:
            return crossed
        if crossed in ("unreachable", "timeout"):
            return crossed   # the crossing never fired; not a route divergence

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


# ---------------------------------------------------------------------------
# Crossing helpers (Task 2)
# ---------------------------------------------------------------------------

_WARP_SETTLE_FRAMES = 64  # let the warp engine complete after stepping onto the tile
_UP_HOLD_FRAMES = 24
_PRECISION_STEP_FRAMES = 4
_PRECISION_RELEASE_FRAMES = 32
_PRECISION_MAX_STEPS = 600
_SCAN_MAX_CANDIDATES = 12   # cap border cells tried per crossing (matches _column_scan)
_SCAN_HOLD_PRESSES = 20     # direction presses per candidate before giving up on it
_STANDABLE = frozenset({TileKind.FREE, TileKind.GRASS})
# Try the most promising border exits first: southmost for DOWN, northmost for UP,
# westmost for LEFT, eastmost for RIGHT.
_BORDER_SORT = {
    "down": lambda c: -c[1], "up": lambda c: c[1],
    "left": lambda c: c[0], "right": lambda c: -c[0],
}


def _precision_step(emulator: Any, key: int) -> None:
    """Press a direction for 4 frames (1-tile precision) then release."""
    emulator.step(key, _PRECISION_STEP_FRAMES)
    emulator.step(0, _PRECISION_RELEASE_FRAMES)


def _precision_walk_to(
    emulator: Any, reader: Any, snap: Any, target: tuple[int, int], from_map: tuple[int, int]
) -> bool:
    """Re-plan each step and take 4-frame precision steps toward target on from_map.

    navigate_grid's 24-frame steps overshoot on Littleroot's dense topology; precision
    steps land exactly. Returns True on arrival, False if it leaves the map or stalls.
    """
    for _ in range(_PRECISION_MAX_STEPS):
        here = snapshot_settled(reader)
        if here is None or here.map_id != from_map:
            return False
        if here.pos == target:
            return True
        path = plan_path_grid(snap, here.pos, target)
        if not path:
            return False
        _precision_step(emulator, DIRECTION_KEYS[path[0]])
    return False


def _doorstep_cells(snap: Any) -> list[tuple[int, int]]:
    """Standable cells with a WALL directly above (building doorsteps), southmost-first.

    The Littleroot->lab door is a map warp_event, NOT an MB_WARP (0x60) tile — a
    behaviour scan finds only the player's-house door. Doors render a WALL row above
    the doorstep, so scan geometry instead: any standable cell whose north neighbour
    is WALL is a candidate. Pressing UP from there triggers the warp event."""
    out: list[tuple[int, int]] = []
    for y in range(snap.height - 1, -1, -1):   # southmost first
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            above = snap.classify_at(x, y - 1) if y > 0 else None
            if above is TileKind.WALL:
                out.append((x, y))
    return out


def _cross_in_direction(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int], direction: str,
    move_type_fn: Any = None, predict: Any = None,
) -> str:
    """Cross from_map's border going `direction`. Two kinds, chosen by direction.

    Returns 'crossed' | 'no_crossing' | a battle outcome.
    """
    if direction == "up":
        return _cross_up_warp(emulator, reader, memory, from_map, move_type_fn, predict)
    return _cross_border(emulator, reader, memory, from_map, direction, move_type_fn, predict)


def _border_cells(snap: Any, here_pos: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """Standable cells whose `direction`-neighbour is off-map and that are reachable
    from here_pos, sorted edge-first (southmost for DOWN, etc.)."""
    dx, dy = DELTAS[direction]
    out: list[tuple[int, int]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < snap.width and 0 <= ny < snap.height:
                continue   # neighbour is on-map: not a border cell for this direction
            if (x, y) == here_pos or plan_path_grid(snap, here_pos, (x, y)) is not None:
                out.append((x, y))
    out.sort(key=_BORDER_SORT[direction])
    return out[:_SCAN_MAX_CANDIDATES]


def _probe_border_cell(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int],
    direction: str, move_type_fn: Any, predict: Any,
) -> str | None:
    """Press `direction` up to _SCAN_HOLD_PRESSES times from the current cell.

    On the first map flip, record the reversible portal and return 'crossed'.
    On a battle outcome, return that outcome string. Return None if the loop
    exhausts without crossing (this cell did not cross).
    """
    for _ in range(_SCAN_HOLD_PRESSES):
        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle
        before = _snapshot_settled(reader)
        if before is None or before.map_id != from_map:
            break
        probe_step(emulator, reader, before, direction)
        after = _snapshot_settled(reader)
        if after is not None and after.map_id != from_map:
            memory.record_portal(from_map, before.pos, direction, after.map_id, True, after.pos)
            return "crossed"
    return None


def _cross_border(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int], direction: str,
    move_type_fn: Any, predict: Any,
) -> str:
    """Directional border crossing (ported from probe_return_portals._column_scan).

    Navigate to each edge-first border cell and press `direction`; the first cell that
    flips the map wins and records the reversible portal. A DIRECTED descent, unlike the
    greedy explore_grid sweep which leaves via the first non-reversible border in ANY
    direction (route_101's north exit to Oldale instead of its south exit to Littleroot).
    Returns 'crossed' | 'no_crossing' | a battle outcome.
    """
    here = _snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return "no_crossing"
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return "no_crossing"
    for cell in _border_cells(snap, here.pos, direction):
        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle
        arrived = navigate_grid(
            emulator, reader, cell, memory=memory, move_type_fn=move_type_fn, predict=predict
        )
        if arrived in BATTLE_OUTCOMES:
            return arrived
        if arrived != "arrived":
            continue
        outcome = _probe_border_cell(emulator, reader, memory, from_map, direction, move_type_fn, predict)
        if outcome in BATTLE_OUTCOMES:
            return outcome
        if outcome == "crossed":
            return "crossed"
    return "no_crossing"


def _cross_up_warp(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int],
    move_type_fn: Any, predict: Any,
) -> str:
    """UP interior door: walk onto an MB_WARP tile (which triggers the warp), settle,
    record the portal. explore_grid only tests map-edge cells, so it misses interior doors."""
    battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
    if battle is not None:
        return battle
    here = _snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return "no_crossing"
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return "no_crossing"
    for cell in _doorstep_cells(snap):
        cur = _snapshot_settled(reader)
        if cur is None or cur.map_id != from_map:
            return "no_crossing"
        _precision_walk_to(emulator, reader, snap, cell, from_map)
        landed = _snapshot_settled(reader)
        if landed is not None and landed.map_id != from_map:
            memory.record_portal(from_map, cell, "up", landed.map_id, True, landed.pos)
            return "crossed"
        emulator.step(DIRECTION_KEYS["up"], _UP_HOLD_FRAMES)
        emulator.step(0, _WARP_SETTLE_FRAMES)
        settled = _snapshot_settled(reader)
        if settled is not None and settled.map_id != from_map:
            memory.record_portal(from_map, cell, "up", settled.map_id, True, settled.pos)
            return "crossed"
    return "no_crossing"


def reach_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    goal_map: tuple[int, int],
    direction_by_map: dict[tuple[int, int], str],
    *,
    move_type_fn: Any = None,
    predict: Any = None,
    max_hops: int = 12,
) -> str:
    """Greedy hop loop: follow direction_by_map until goal_map is reached.

    Returns 'arrived' | 'stall' | 'timeout' | 'battle_lost' | 'battle_timeout' |
    'battle_interrupted'. 'stall' = current map not in direction_by_map, or no crossing
    fired. 'timeout' = hop budget exhausted.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating between maps; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return "arrived"
        direction = direction_by_map.get(here.map_id)
        if direction is None:
            return "stall"
        crossed = _cross_in_direction(
            emulator, reader, memory, here.map_id, direction,
            move_type_fn=move_type_fn, predict=predict,
        )
        if crossed in BATTLE_OUTCOMES:
            return crossed
        if crossed != "crossed":
            return "stall"
    return "timeout"
