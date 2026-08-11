"""Throwaway probe: settle the southbound return crossing primitive and record the real chain.

Loads states/post_rival.state (player on route_103 after beating the rival) and
drives a per-hop descent: route_103 -> Oldale -> route_101 -> Littleroot -> lab.

Strategy:

  route_103 -> Oldale: explore_grid sweep (entry north) + column-scan fallback.

  Oldale -> route_101: STATE-RELOAD approach.
    The Oldale south exit (y=19) is only active when the player ENTERED from route_101
    (south entry). After coming from route_103 (north entry), the south connection strip
    is not activated and probe_step DOWN from y=19 returns "blocked".
    Fix: reload post_starter.state (player on route_101), navigate route_101 north to
    (11,1), probe UP to enter Oldale at (11,19), then probe DOWN to return to route_101
    at (11,1). This records both portals and leaves the player on route_101.

  route_101 -> Littleroot: column-scan down (or explore_grid).

  Littleroot -> lab: nav+probe with UP direction (door warp).

Run (from the main repo so the ROM/venv/states are available):
  cd /Users/_eloi/Projets/Emu
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \\
      .venv/bin/python tools/probe_return_portals.py states/post_rival.state
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator import buttons as _buttons
from emulator.gba import GbaEmulator
from env.grid_explorer import explore_grid
from env.grid_navigator import (
    DELTAS,
    handle_battle_interruption,
    navigate_grid,
    plan_path_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory, Portal
from env.world_reader import WorldReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUTE_103 = (0, 18)
OLDALE = (0, 10)
ROUTE_101 = (0, 16)
LITTLEROOT = (0, 9)
LAB = (1, 4)

# Ordered descent: (from_map, to_map, primary_direction)
# Direction = what the player must press to cross the border at each hop.
# NOTE: Oldale→route_101 is DOWN (south).  The last run showed that after explore_grid
# repositions the player to the south-west half (~(1,10)), the south border (y=19) has
# reachable cells: border (4,19) down: reachable=True, (5,19), (7,19) … (15,19).
# The previous "left" hypothesis was wrong — west leads to route_102, not route_101.
RETURN_CHAIN = [
    (ROUTE_103, OLDALE, "down"),
    (OLDALE, ROUTE_101, "down"),
    (ROUTE_101, LITTLEROOT, "down"),
    (LITTLEROOT, LAB, "up"),     # lab door is north of the player in Littleroot
]
CHAIN_ORDER = [ROUTE_103, OLDALE, ROUTE_101, LITTLEROOT, LAB]

# Max presses for the straight-shot hold per hop.
_HOLD_MAX_PRESSES = 40
# Consecutive non-advancing presses before declaring the column wrong (triggers column-scan).
_STALL_PATIENCE = 5
# Max border-cell candidates to try during the column-scan fallback.
_SCAN_MAX_CANDIDATES = 12
# Max hold presses per candidate during the column scan.
_SCAN_HOLD_PRESSES = 20

_STANDABLE = {TileKind.FREE, TileKind.GRASS}
_GLYPH = {
    TileKind.FREE: ".", TileKind.WALL: "#", TileKind.GRASS: '"',
    TileKind.LEDGE_UP: "^", TileKind.LEDGE_DOWN: "v",
    TileKind.LEDGE_LEFT: "<", TileKind.LEDGE_RIGHT: ">",
}

# Party RAM constants (unencrypted; confirmed heatz123/pokeagent + pret decomp).
_PARTY_COUNT_ADDR = 0x020244E9
_PARTY_ADDR = 0x020244EC
_PARTY_STRIDE = 100
_PARTY_CURHP_OFF = 0x56   # u16 current HP
_PARTY_MAXHP_OFF = 0x58   # u16 max HP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _heal_party(emu: GbaEmulator) -> None:
    """Write curHP <- maxHP for every party member to prevent route-grass whiteout.

    Probe-only: accesses emu._core directly since GbaEmulator has no public write_bytes.
    """
    count = emu.read_bytes(_PARTY_COUNT_ADDR, 1)[0]
    for i in range(count):
        base = _PARTY_ADDR + i * _PARTY_STRIDE
        max_hp = int.from_bytes(emu.read_bytes(base + _PARTY_MAXHP_OFF, 2), "little")
        if max_hp == 0:
            continue
        # raw_write(address, byte_value): write curHP lo/hi bytes.
        emu._core.memory.u8.raw_write(base + _PARTY_CURHP_OFF, max_hp & 0xFF)
        emu._core.memory.u8.raw_write(base + _PARTY_CURHP_OFF + 1, (max_hp >> 8) & 0xFF)


def _dump(reader: WorldReader, pos: tuple[int, int]) -> None:
    """Print ASCII grid + reachable border cells from pos."""
    here = reader.snapshot()
    if here is None:
        print("    (snapshot unavailable)")
        return
    snap = GridSnapshot.from_reader(reader.grid_reader, here.map_id)
    if snap is None:
        print("    (grid unavailable)")
        return
    print(f"    grid {snap.width}x{snap.height} player {pos}:")
    for y in range(snap.height):
        row = "".join(
            "@" if (x, y) == pos else _GLYPH.get(snap.classify_at(x, y), "?")
            for x in range(snap.width)
        )
        print(f"    {y:2d} {row}")
    # Report which border cells are path-reachable from pos.
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            for d, (dx, dy) in DELTAS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < snap.width and 0 <= ny < snap.height:
                    continue
                reachable = (plan_path_grid(snap, pos, (x, y)) is not None) or (x, y) == pos
                print(f"    border {(x, y)} {d}: reachable={reachable}")


# ---------------------------------------------------------------------------
# Crossing primitives
# ---------------------------------------------------------------------------

def _straight_shot(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    entry_pos: tuple[int, int],
    mtf: object,
    predict: object,
) -> str | None:
    """Hold `direction` from the clean entry position up to _HOLD_MAX_PRESSES times.

    This is the primary descent primitive. We do NOT call explore_grid first because
    it wanders into the ledge maze and strands the player far from the south exit.
    Wild encounters are cleared by the Fighter so grass does not block progress.
    Returns a note string on success, None if no map transition occurred.
    """
    # Track the axis that should advance: y for down, x for left/right, y for up.
    _axis = 0 if direction in ("left", "right") else 1
    prev_coord = entry_pos[_axis]
    stall_count = 0
    for press in range(_HOLD_MAX_PRESSES):
        # Clear any wild encounter before pressing the d-pad.
        handle_battle_interruption(emu, reader, mtf, predict)
        before = snapshot_settled(reader)
        if before is None:
            continue
        if before.map_id != from_map:
            # Departed from_map (possibly during battle handling).
            if before.map_id == to_map:
                memory.record_portal(from_map, entry_pos, direction, before.map_id, True, before.pos)
                return f"STRAIGHT-SHOT ({direction}, press {press}, departed during battle clear)"
            return None
        outcome = probe_step(emu, reader, before, direction)
        after = snapshot_settled(reader)
        if after is None:
            continue
        if after.map_id != from_map:
            memory.record_portal(from_map, before.pos, direction, after.map_id, True, after.pos)
            print(f"    straight-shot: crossed at press {press + 1}, "
                  f"cell={before.pos} -> {after.map_id}@{after.pos}")
            if after.map_id == to_map:
                return f"STRAIGHT-SHOT ({direction}, {press + 1} press(es) from {before.pos})"
            return None  # crossed to unexpected map — caller will report stall
        # Track whether the movement axis is advancing to detect a stall.
        cur_coord = after.pos[_axis]
        if cur_coord == prev_coord:
            stall_count += 1
            if stall_count >= _STALL_PATIENCE:
                print(f"    straight-shot: axis stalled at {cur_coord} "
                      f"for {stall_count} presses — giving up")
                return None
        else:
            stall_count = 0
            prev_coord = cur_coord
        if press < 6 or press % 5 == 0:
            print(f"    press {press + 1}: outcome={outcome!r} pos={after.pos}")
    return None


def _column_scan(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    mtf: object,
    predict: object,
) -> str | None:
    """Column-scan fallback: navigate to southmost reachable border cells and hold DOWN.

    Used when the straight-shot stalls because the entry column is not the exit column.
    Enumerates standable border cells (southmost-first), navigates to each, holds
    _SCAN_HOLD_PRESSES times. First cell that produces a map transition wins.
    """
    here = snapshot_settled(reader)
    if here is None:
        return None
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return None
    dx, dy = DELTAS[direction]
    # Collect standable cells whose direction-neighbor is off-map (= border cells).
    border: list[tuple[int, int]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < snap.width and 0 <= ny < snap.height:
                continue
            if plan_path_grid(snap, here.pos, (x, y)) is not None or (x, y) == here.pos:
                border.append((x, y))
    # Sort border-edge-first: for DOWN sort southmost (largest y), for LEFT sort westmost
    # (smallest x) so we try the most promising exits early.
    if direction == "down":
        border.sort(key=lambda c: -c[1])
    elif direction == "left":
        border.sort(key=lambda c: c[0])
    else:
        border.sort(key=lambda c: c[1])
    border = border[:_SCAN_MAX_CANDIDATES]
    print(f"    column-scan {direction}: {len(border)} border candidates (southmost first)")
    for cell in border:
        handle_battle_interruption(emu, reader, mtf, predict)
        arrived = navigate_grid(emu, reader, cell, memory=memory, move_type_fn=mtf, predict=predict)
        if arrived != "arrived":
            print(f"      {cell}: nav={arrived!r}")
            continue
        for press in range(_SCAN_HOLD_PRESSES):
            handle_battle_interruption(emu, reader, mtf, predict)
            before = snapshot_settled(reader)
            if before is None or before.map_id != from_map:
                break
            probe_step(emu, reader, before, direction)
            after = snapshot_settled(reader)
            if after is None:
                continue
            if after.map_id != from_map:
                memory.record_portal(from_map, before.pos, direction, after.map_id, True, after.pos)
                print(f"      {cell} press {press + 1}: crossed -> {after.map_id}@{after.pos}")
                if after.map_id == to_map:
                    return f"COLUMN-SCAN ({direction} from {cell}, press {press + 1})"
                return None
        print(f"      {cell}: no transition after {_SCAN_HOLD_PRESSES} presses")
    return None


def _nav_then_probe(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    mtf: object,
    predict: object,
) -> str | None:
    """Single-probe border scan: navigate to each reachable border cell + one probe_step.

    Used for door warps (lab hop UP) where a single step into the door suffices.
    """
    here = snapshot_settled(reader)
    if here is None:
        return None
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return None
    dx, dy = DELTAS[direction]
    candidates: list[tuple[int, int]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < snap.width and 0 <= ny < snap.height:
                continue
            if plan_path_grid(snap, here.pos, (x, y)) is not None or (x, y) == here.pos:
                candidates.append((x, y))
    print(f"    nav+probe {direction}: {len(candidates)} candidate border cells")
    for cell in candidates:
        arrived = navigate_grid(emu, reader, cell, memory=memory, move_type_fn=mtf, predict=predict)
        if arrived != "arrived":
            print(f"      {cell}: nav={arrived!r}")
            continue
        before = snapshot_settled(reader)
        if before is None or before.pos != cell:
            continue
        outcome = probe_step(emu, reader, before, direction)
        landed = snapshot_settled(reader)
        land_map = landed.map_id if landed else None
        land_pos = landed.pos if landed else None
        print(f"      {cell} -> {outcome!r} land={land_map}@{land_pos}")
        if landed is not None and land_map != from_map:
            memory.record_portal(from_map, cell, direction, land_map, True, land_pos)
            if land_map == to_map:
                return f"NAV+PROBE ({direction} from {cell})"
    return None


# ---------------------------------------------------------------------------
# Portal crossing via navigate_grid (mirrors _cross in probe_north_exit_truth.py)
# ---------------------------------------------------------------------------

def _cross_portal(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    portal: Portal,
    mtf: object,
    predict: object,
) -> str:
    """Navigate to portal.from_cell then one step into the neighbor cell.

    Returns "left_map" / "arrived" / navigate_grid outcome string.
    """
    r = navigate_grid(emu, reader, portal.from_cell, memory=memory, move_type_fn=mtf, predict=predict)
    if r == "left_map":
        return "left_map"
    if r != "arrived":
        return r
    dx, dy = DELTAS[portal.direction]
    nb = (portal.from_cell[0] + dx, portal.from_cell[1] + dy)
    return navigate_grid(emu, reader, nb, memory=memory, move_type_fn=mtf, predict=predict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_state(emu: GbaEmulator, state_path: str) -> None:
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)


def _hop_via_explore_then_scan(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    mtf: object,
    predict: object,
) -> str | None:
    """Standard hop: explore_grid sweep, then cross discovered portal, then column-scan."""
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != from_map:
        return None
    entry_pos = snap.pos

    print(f"    Running explore_grid on {from_map}...")
    eg_result = explore_grid(emu, reader, memory, from_map, move_type_fn=mtf, predict=predict)
    snap_eg = snapshot_settled(reader)
    pos_eg = snap_eg.pos if snap_eg else None
    map_eg = snap_eg.map_id if snap_eg else None
    print(f"    explore_grid -> {eg_result!r} | now {map_eg}@{pos_eg}")

    portals = memory.outgoing_portals(from_map)
    print(f"    outgoing_portals({from_map}) [{len(portals)} total]:")
    for p in portals:
        print(
            f"      from_cell={p.from_cell} dir={p.direction!r} "
            f"to_map={p.to_map} to_cell={p.to_cell} rev={p.reversible}"
        )

    if map_eg == to_map:
        memory.record_portal(from_map, entry_pos, direction, to_map, True, pos_eg)
        return f"EXPLORE_GRID-AUTO (landed on {to_map}@{pos_eg})"

    target_portal: Portal | None = next((p for p in portals if p.to_map == to_map), None)
    if target_portal is not None:
        print(
            f"    Found portal to {to_map}: "
            f"{target_portal.from_cell} {target_portal.direction!r} -> "
            f"{target_portal.to_map}@{target_portal.to_cell}"
        )
        snap_now = snapshot_settled(reader)
        if snap_now is not None and snap_now.map_id == from_map:
            _heal_party(emu)
            crossed = _cross_portal(emu, reader, memory, target_portal, mtf, predict)
            print(f"    _cross_portal -> {crossed!r}")
            snap_cross = snapshot_settled(reader)
            if snap_cross is not None and snap_cross.map_id == to_map:
                return (
                    f"EXPLORE+CROSS ({target_portal.direction!r} "
                    f"from {target_portal.from_cell})"
                )

    # Fallback: straight-shot then column-scan.
    snap_now = snapshot_settled(reader)
    if snap_now is not None and snap_now.map_id == from_map:
        print(f"    Trying straight-shot ({direction}) from {snap_now.pos}...")
        note = _straight_shot(
            emu, reader, memory, from_map, to_map, direction, snap_now.pos, mtf, predict
        )
        if note is not None:
            return note

    snap_now = snapshot_settled(reader)
    if snap_now is not None and snap_now.map_id == from_map:
        print("    Straight-shot stalled — trying column-scan fallback...")
        return _column_scan(emu, reader, memory, from_map, to_map, direction, mtf, predict)

    return None



def _attempt_oldale_south_entry(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    starter_state: str,
    mtf: object,
    predict: object,
) -> bool:
    """Single attempt: reload starter_state, use explore_grid to find north exit, probe back.

    explore_grid sweeps ALL reachable border cells (including those behind ledge
    barriers unreachable by plain A*), so it reliably finds the route_101->Oldale
    north portal even when navigate_grid times out heading north.

    Returns True if both portals were recorded and player is on route_101.
    """
    _load_state(emu, starter_state)
    _heal_party(emu)
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != ROUTE_101:
        print(f"    ERROR: expected ROUTE_101 after reload, got {snap.map_id if snap else None}")
        return False
    print(f"    After reload: map={snap.map_id} pos={snap.pos}")

    # explore_grid sweeps the full border; discovers the north Oldale portal and
    # may auto-cross into Oldale (explore_grid records reversibility itself).
    print(f"    Running explore_grid on {ROUTE_101}...")
    eg_result = explore_grid(
        emu, reader, memory, ROUTE_101, move_type_fn=mtf, predict=predict
    )
    _heal_party(emu)
    snap_eg = snapshot_settled(reader)
    map_eg = snap_eg.map_id if snap_eg else None
    pos_eg = snap_eg.pos if snap_eg else None
    print(f"    explore_grid -> {eg_result!r} | now {map_eg}@{pos_eg}")

    portals_101 = memory.outgoing_portals(ROUTE_101)
    print(f"    outgoing_portals(ROUTE_101) [{len(portals_101)} total]:")
    for p in portals_101:
        print(f"      {p.from_cell} {p.direction!r} -> {p.to_map}@{p.to_cell} rev={p.reversible}")

    # Case A: explore_grid crossed into Oldale — probe DOWN to record the return portal.
    if map_eg == OLDALE:
        print(f"    explore_grid landed on Oldale@{pos_eg}")
        return _probe_oldale_down_and_record(emu, reader, memory, pos_eg)

    # Case B: portal to Oldale was discovered; navigate to its from_cell, probe UP.
    north_portal = next((p for p in portals_101 if p.to_map == OLDALE), None)
    if north_portal is not None and map_eg == ROUTE_101:
        print(f"    Found north portal: {north_portal.from_cell} up -> Oldale@{north_portal.to_cell}")
        _heal_party(emu)
        r_nav = navigate_grid(
            emu, reader, north_portal.from_cell, max_steps=600,
            memory=memory, move_type_fn=mtf, predict=predict,
        )
        snap_at = snapshot_settled(reader)
        pos_at = snap_at.pos if snap_at else None
        print(f"    Nav to {north_portal.from_cell}: {r_nav!r} | pos={pos_at}")
        if snap_at is not None and snap_at.map_id == ROUTE_101 and snap_at.pos == north_portal.from_cell:
            before_up = snapshot_settled(reader)
            if before_up is None:
                return False
            outcome_up = probe_step(emu, reader, before_up, "up")
            snap_oldale = snapshot_settled(reader)
            oldale_map = snap_oldale.map_id if snap_oldale else None
            oldale_pos = snap_oldale.pos if snap_oldale else None
            print(f"    Probe UP from {north_portal.from_cell}: {outcome_up!r} -> {oldale_map}@{oldale_pos}")
            if oldale_map == OLDALE:
                memory.record_portal(ROUTE_101, north_portal.from_cell, "up", OLDALE, True, oldale_pos)
                return _probe_oldale_down_and_record(emu, reader, memory, oldale_pos)

    print("    Could not reach or probe Oldale from route_101")
    return False


def _probe_oldale_down_and_record(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    oldale_pos: tuple[int, int],
) -> bool:
    """Probe DOWN from current Oldale position; record portal; return to route_101."""
    before_down = snapshot_settled(reader)
    if before_down is None:
        return False
    outcome_down = probe_step(emu, reader, before_down, "down")
    snap_back = snapshot_settled(reader)
    back_map = snap_back.map_id if snap_back else None
    back_pos = snap_back.pos if snap_back else None
    print(f"    Probe DOWN from Oldale@{oldale_pos}: {outcome_down!r} -> {back_map}@{back_pos}")
    if back_map != ROUTE_101:
        print(f"    ERROR: DOWN from Oldale returned {back_map}, not route_101")
        return False
    memory.record_portal(OLDALE, oldale_pos, "down", ROUTE_101, True, back_pos)
    print("    Both portals recorded. Player on route_101.")
    return True


def _hop_oldale_via_south_entry(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    starter_state: str,
    mtf: object,
    predict: object,
) -> bool:
    """Discover Oldale<->route_101 portals via state-reload south-entry, with retries.

    The Oldale south border (y=19) is only live when the player entered from
    route_101 (south). After entering from route_103 (north), the connection
    strip is not activated. Fix: reload post_starter.state (player on route_101),
    walk north to (11,1), probe UP into Oldale, probe DOWN back.
    """
    print("\n--- Oldale<->route_101 via south-entry reload ---")
    for attempt in range(3):
        print(f"    Attempt {attempt + 1}/3: loading {starter_state}...")
        ok = _attempt_oldale_south_entry(emu, reader, memory, starter_state, mtf, predict)
        if ok:
            return True
        print(f"    Attempt {attempt + 1} failed — retrying")
    print("    All attempts failed")
    return False


_KEY_FOR: dict[str, int] = {
    "up": _buttons.KEY_UP,
    "down": _buttons.KEY_DOWN,
    "left": _buttons.KEY_LEFT,
    "right": _buttons.KEY_RIGHT,
}
# Probe-confirmed lab entrance: standing at (8,17) in Littleroot and pressing UP
# triggers the warp into the lab.  navigate_grid (STEP_FRAMES=24) overshoots in
# short-distance navigation on Littleroot's complex topology, so we use 4-frame
# precision steps driven by plan_path_grid instead.
_LAB_DOOR_CELL = (8, 17)
_PRECISION_STEP_FRAMES = 4
_PRECISION_RELEASE_FRAMES = 32
_PRECISION_MAX_STEPS = 600


def _precision_step(emu: GbaEmulator, key: int) -> None:
    """Press a direction for 4 frames (1-tile precision) then release."""
    emu.step(key, _PRECISION_STEP_FRAMES)
    emu.step(0, _PRECISION_RELEASE_FRAMES)


def _precision_walk_to(
    emu: GbaEmulator,
    reader: WorldReader,
    snap: GridSnapshot,
    target: tuple[int, int],
    from_map: tuple[int, int],
) -> bool:
    """Walk to target using plan_path_grid + 4-frame steps.

    Returns True if the player arrived at target on from_map.
    navigate_grid uses STEP_FRAMES=24 which causes 2-tile overshoots for
    close targets on Littleroot; 4-frame steps move exactly 1 tile per press
    (after the initial turn press), making short-range navigation reliable.
    """
    for _ in range(_PRECISION_MAX_STEPS):
        here = snapshot_settled(reader)
        if here is None or here.map_id != from_map:
            return False
        if here.pos == target:
            return True
        path = plan_path_grid(snap, here.pos, target)
        if path is None:
            return False
        _precision_step(emu, _KEY_FOR[path[0]])
    return False


_MB_WARP = 0x60   # pokeemerald metatile_behaviors.h MB_WARP

def _find_warp_cells(reader: WorldReader, map_id: tuple[int, int]) -> list[tuple[int, int]]:
    """Return all FREE cells in map_id whose tile behavior is MB_WARP (0x60).

    Building entrances in Gen-3 Emerald are warp tiles: collision=0, behavior=MB_WARP.
    Walking onto such a tile triggers the map transition.
    """
    snap = GridSnapshot.from_reader(reader.grid_reader, map_id)
    if snap is None:
        return []
    mgr = reader.grid_reader
    results = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            beh = mgr.tile_behavior_at(x, y)
            if beh == _MB_WARP:
                results.append((x, y))
    return results


def _warp_scan_up(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    mtf: object,
    predict: object,
) -> str | None:
    """Enter the lab from Littleroot by finding and stepping onto the door warp tile.

    In Gen-3 Emerald, building entrances are FREE tiles with behavior MB_WARP (0x60).
    Walking onto the warp tile (not pressing UP from in front of it) triggers the
    transition. We scan for all MB_WARP tiles in Littleroot, then walk onto each one
    using precision steps (4 frames) to avoid overshooting.
    """
    here = snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return None
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return None

    handle_battle_interruption(emu, reader, mtf, predict)
    _heal_party(emu)

    # Find all MB_WARP (0x60) tiles — lab door candidates.
    warp_cells = _find_warp_cells(reader, from_map)
    print(f"    MB_WARP cells in {from_map}: {warp_cells}")

    print(f"    MB_WARP cells: {warp_cells}")

    # Strategy: in Gen-3 Emerald, warp events fire when the player steps onto the
    # warp event tile. The lab door is triggered by walking UP from the row below
    # the building's south face. Scan every FREE tile in rows 13-18 that has a
    # WALL cell directly above it (candidate doorstep) — walk there, press UP 24fr.
    candidates: list[tuple[int, int]] = []
    for y in range(snap.height - 1, -1, -1):   # southmost first
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            above = snap.classify_at(x, y - 1) if y > 0 else None
            if above is TileKind.WALL:
                candidates.append((x, y))

    print(f"    WALL-above candidates in {from_map}: {len(candidates)} cells")
    for cell in candidates:
        here_now = snapshot_settled(reader)
        if here_now is None or here_now.map_id != from_map:
            return None
        snap2 = GridSnapshot.from_reader(reader.grid_reader, from_map)
        if snap2 is None:
            return None
        ok = _precision_walk_to(emu, reader, snap2, cell, from_map)
        here_now = snapshot_settled(reader)
        cur_pos = here_now.pos if here_now else None
        # If _precision_walk_to already crossed to to_map, record and return.
        if here_now is not None and here_now.map_id == to_map:
            memory.record_portal(from_map, cell, "up", to_map, True, here_now.pos)
            print(f"      {cell}: walk itself triggered warp -> {to_map}@{here_now.pos}")
            return f"WARP-STEP (walk onto {cell})"
        if not ok or here_now is None or here_now.map_id != from_map or cur_pos != cell:
            print(f"      {cell}: precision-walk failed (pos={cur_pos} map={here_now.map_id if here_now else None})")
            continue
        # Press UP with 24-frame hold + extra settle to let the warp engine fire.
        emu.step(_KEY_FOR["up"], 24)
        emu.step(0, 64)   # longer release to let warp transition complete
        landed = snapshot_settled(reader)
        land_map = landed.map_id if landed else None
        land_pos = landed.pos if landed else None
        if land_map == to_map:
            memory.record_portal(from_map, cell, "up", land_map, True, land_pos)
            print(f"      {cell} UP -> {land_map}@{land_pos}  *** WARP FOUND ***")
            return f"WARP-STEP (UP from {cell})"
        if land_map != from_map:
            print(f"      {cell} UP -> unexpected {land_map}@{land_pos}")
            return None
        # Moved to an intermediate cell (e.g. inside building row).
        # Try one more settle cycle — warp may need more time.
        emu.step(0, 64)
        landed2 = snapshot_settled(reader)
        l2_map = landed2.map_id if landed2 else None
        l2_pos = landed2.pos if landed2 else None
        print(f"      {cell} UP -> {land_map}@{land_pos} -> settle -> {l2_map}@{l2_pos}")
        if l2_map == to_map:
            memory.record_portal(from_map, cell, "up", l2_map, True, l2_pos)
            return f"WARP-STEP (UP from {cell}, delayed)"
        if l2_map != from_map:
            print(f"      {cell} delayed settle: unexpected {l2_map}")
            return None

    return None


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    rival_state = sys.argv[1] if len(sys.argv) > 1 else "states/post_rival.state"
    # post_starter.state is used for the Oldale south-entry reload.
    starter_state = sys.argv[2] if len(sys.argv) > 2 else "states/post_starter.state"
    fighter_ckpt = "checkpoints/fighter/ppo_fighter_final.zip"

    emu = GbaEmulator(rom)
    _load_state(emu, rival_state)

    reader = WorldReader(emu.read_bytes)
    memory = MapMemory()

    print(f"Loading Fighter from {fighter_ckpt}...")
    model = PPO.load(fighter_ckpt, device="cpu")
    mtf = make_move_type_fn(emu)

    def predict(obs: object) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    _heal_party(emu)

    snap = snapshot_settled(reader)
    if snap is None:
        print("ERROR: snapshot returned None at startup")
        sys.exit(1)
    print(f"Start: map={snap.map_id} pos={snap.pos}")

    results: list[str] = []

    # -----------------------------------------------------------------------
    # HOP 1: route_103 -> Oldale (explore_grid sweep + column-scan)
    # -----------------------------------------------------------------------
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != ROUTE_103:
        print(f"ERROR: expected ROUTE_103 {ROUTE_103}, got {snap.map_id if snap else None}")
        sys.exit(1)

    print(f"\n=== HOP {ROUTE_103} -> {OLDALE} (direction=down) ===")
    print(f"  Entry: map={ROUTE_103} pos={snap.pos}")
    _dump(reader, snap.pos)
    _heal_party(emu)

    note = _hop_via_explore_then_scan(
        emu, reader, memory, ROUTE_103, OLDALE, "down", mtf, predict
    )
    snap_after = snapshot_settled(reader)
    if note is None or snap_after is None or snap_after.map_id != OLDALE:
        actual = snap_after.map_id if snap_after else None
        _stall_exit(memory, ROUTE_103, OLDALE, snap_after, note)

    results.append(f"{ROUTE_103} -> {OLDALE}: {note}")
    print(f"  => CROSSED: {note}")

    # -----------------------------------------------------------------------
    # HOP 2: Oldale -> route_101 (state-reload south-entry approach)
    # -----------------------------------------------------------------------
    print(f"\n=== HOP {OLDALE} -> {ROUTE_101} (south-entry reload) ===")
    ok = _hop_oldale_via_south_entry(emu, reader, memory, starter_state, mtf, predict)
    snap_after = snapshot_settled(reader)
    if not ok or snap_after is None or snap_after.map_id != ROUTE_101:
        actual = snap_after.map_id if snap_after else None
        print(f"\n  STALLED on hop {OLDALE} -> {ROUTE_101} (actual map: {actual})")
        print(f"\nREACHED lab = False")
        _print_portals(memory)
        sys.exit(1)

    results.append(f"{OLDALE} -> {ROUTE_101}: SOUTH-ENTRY-RELOAD")
    print(f"  => CROSSED: SOUTH-ENTRY-RELOAD (player on route_101@{snap_after.pos})")

    # -----------------------------------------------------------------------
    # HOP 3: route_101 -> Littleroot (column-scan down, skip explore_grid)
    # explore_grid battle-timeouts on grass-heavy route_101; go straight to
    # column-scan on the known south-exit columns (10,19) and (11,19).
    # -----------------------------------------------------------------------
    print(f"\n=== HOP {ROUTE_101} -> {LITTLEROOT} (direction=down) ===")
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != ROUTE_101:
        print(f"ERROR: expected on ROUTE_101, got {snap.map_id if snap else None}")
        _stall_exit(memory, ROUTE_101, LITTLEROOT, snap, None)
    print(f"  Entry: map={ROUTE_101} pos={snap.pos}")
    _dump(reader, snap.pos)
    _heal_party(emu)

    note = _column_scan(
        emu, reader, memory, ROUTE_101, LITTLEROOT, "down", mtf, predict
    )
    snap_after = snapshot_settled(reader)
    if note is None or snap_after is None or snap_after.map_id != LITTLEROOT:
        # Fallback: try explore_grid in case column-scan missed a portal.
        _heal_party(emu)
        snap_now = snapshot_settled(reader)
        if snap_now is not None and snap_now.map_id == ROUTE_101:
            note = _hop_via_explore_then_scan(
                emu, reader, memory, ROUTE_101, LITTLEROOT, "down", mtf, predict
            )
        snap_after = snapshot_settled(reader)
    if note is None or snap_after is None or snap_after.map_id != LITTLEROOT:
        _stall_exit(memory, ROUTE_101, LITTLEROOT, snap_after, note)

    results.append(f"{ROUTE_101} -> {LITTLEROOT}: {note}")
    print(f"  => CROSSED: {note}")

    # -----------------------------------------------------------------------
    # HOP 4: Littleroot -> lab (interior door warp scan, direction=up)
    # The lab door is an interior warp tile, not a map-edge crossing, so
    # _nav_then_probe (border-cell scan) misses it.  We probe UP from every
    # reachable standable cell adjacent to a WALL tile above it (buildings).
    # -----------------------------------------------------------------------
    print(f"\n=== HOP {LITTLEROOT} -> {LAB} (direction=up) ===")
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != LITTLEROOT:
        print(f"ERROR: expected on LITTLEROOT, got {snap.map_id if snap else None}")
        _stall_exit(memory, LITTLEROOT, LAB, snap, None)
    print(f"  Entry: map={LITTLEROOT} pos={snap.pos}")
    _dump(reader, snap.pos)
    _heal_party(emu)

    note = _warp_scan_up(emu, reader, memory, LITTLEROOT, LAB, mtf, predict)
    if note is None:
        # Fallback to border scan in case the warp is at the north edge.
        _heal_party(emu)
        snap_now = snapshot_settled(reader)
        if snap_now is not None and snap_now.map_id == LITTLEROOT:
            note = _nav_then_probe(emu, reader, memory, LITTLEROOT, LAB, "up", mtf, predict)
    snap_after = snapshot_settled(reader)
    if note is None or snap_after is None or snap_after.map_id != LAB:
        _stall_exit(memory, LITTLEROOT, LAB, snap_after, note)

    results.append(f"{LITTLEROOT} -> {LAB}: {note}")
    print(f"  => CROSSED: {note}")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    snap_final = snapshot_settled(reader)
    on_lab = snap_final is not None and snap_final.map_id == LAB
    print(f"\n{'='*60}")
    print(f"REACHED lab = {on_lab}")
    if snap_final:
        print(f"Final map={snap_final.map_id} pos={snap_final.pos}")

    print("\nCrossing primitive per hop:")
    for line in results:
        print(f"  {line}")

    _print_portals(memory)

    if not on_lab:
        print("\nREACHED lab = False (post-loop)")
        sys.exit(1)


def _stall_exit(
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    snap_after: object,
    note: str | None,
) -> None:
    """Print stall diagnostics and exit 1."""
    actual = snap_after.map_id if snap_after else None
    print(f"\n  STALLED on hop {from_map} -> {to_map} (actual map: {actual}). note={note!r}")
    stall_portals = memory.outgoing_portals(from_map)
    print(f"\n  outgoing_portals({from_map}) at stall [{len(stall_portals)} total]:")
    for p in stall_portals:
        print(
            f"    from_cell={p.from_cell} dir={p.direction!r} "
            f"to_map={p.to_map} to_cell={p.to_cell} rev={p.reversible}"
        )
    print(f"\nREACHED lab = False")
    _print_portals(memory)
    sys.exit(1)


def _print_portals(memory: MapMemory) -> None:
    """Print the recorded portal chain in _PortalSeed order."""
    print("\nRecorded return portals (in _PortalSeed order):")
    for a, b in zip(CHAIN_ORDER, CHAIN_ORDER[1:]):
        p = memory.portal(a, b)
        if p is None:
            print(f"  {a} -> {b}: NOT RECORDED")
        else:
            print(f"  _PortalSeed({a}, {p.from_cell}, {p.direction!r}, "
                  f"{b}, {p.reversible}, {p.to_cell}),")


if __name__ == "__main__":
    main()
