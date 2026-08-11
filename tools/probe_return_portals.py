"""Throwaway probe: settle the southbound return crossing primitive and record the real chain.

Loads states/post_rival.state (player on route_103 after beating the rival) and
drives a per-hop descent: route_103 -> Oldale -> route_101 -> Littleroot -> lab.

Per hop, two primitives are tried in order:
  1. explore_grid — the proven northbound sweep, ledge-aware. If the map changes,
     the sweep carried us across by stepping through every reachable border cell.
     This is the natural way to discover portals including ledge/muret crossings.
  2. Hold-direction fallback — from the entry position, press the hop direction
     continuously (bounded cap, no navigate_grid first), letting a muret auto-carry
     the player across the map border if a gap exists on the correct column.
     This covers the case where explore_grid stays on the same map (e.g., a
     non-reversible muret that the sweep avoids because it can't return).

The crux (user domain knowledge): a ledge muret is crossed by WALKING INTO it, not
by routing to the border cell and issuing a single probe_step. The hold fallback
holds the direction continuously so the muret's auto-carry mechanism can trigger.

Run (from the main repo so the ROM/venv/states are available):
  cd /Users/_eloi/Projets/Emu
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
      .venv/bin/python tools/probe_return_portals.py states/post_rival.state
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator.gba import GbaEmulator
from env.grid_explorer import explore_grid
from env.grid_navigator import DELTAS, navigate_grid, plan_path_grid, probe_step, snapshot_settled
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory
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
RETURN_CHAIN = [
    (ROUTE_103, OLDALE, "down"),
    (OLDALE, ROUTE_101, "down"),
    (ROUTE_101, LITTLEROOT, "down"),
    (LITTLEROOT, LAB, "up"),     # lab door is north of the player in Littleroot
]
CHAIN_ORDER = [ROUTE_103, OLDALE, ROUTE_101, LITTLEROOT, LAB]

# Max presses for the hold-direction fallback per hop.
_HOLD_MAX_PRESSES = 40

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

def _try_explore_grid(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    mtf: object,
    predict: object,
) -> str | None:
    """Run explore_grid on from_map and check whether we ended up on to_map.

    Returns a note string if successful, None if we stayed on from_map.
    Records portals into memory as a side effect.
    """
    result = explore_grid(emu, reader, memory, from_map, move_type_fn=mtf, predict=predict)
    print(f"    explore_grid => {result!r}")
    snap = snapshot_settled(reader)
    if snap is not None and snap.map_id == to_map:
        return f"SWEEP (explore_grid, result={result!r})"
    if snap is not None and snap.map_id != from_map:
        print(f"    [note] explore_grid left to unexpected {snap.map_id} (wanted {to_map})")
    return None


def _try_hold_direction(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    entry_pos: tuple[int, int],
) -> str | None:
    """Hold `direction` up to _HOLD_MAX_PRESSES times from the current position.

    This lets a muret auto-carry the player across the map border. Does NOT
    navigate to a specific cell first — holds from wherever the player landed
    after the sweep (or the entry position if the sweep didn't move far).
    Records the portal on transition. Returns a note string on success, None on stall.
    """
    for press in range(_HOLD_MAX_PRESSES):
        before = snapshot_settled(reader)
        if before is None:
            continue
        if before.map_id != from_map:
            # Already left during a previous press.
            if before.map_id == to_map:
                return f"HOLD ({direction}, press {press}, arrived on prev step)"
            return None
        outcome = probe_step(emu, reader, before, direction)
        after = snapshot_settled(reader)
        if after is None:
            continue
        if after.map_id != from_map:
            memory.record_portal(from_map, before.pos, direction, after.map_id, True, after.pos)
            print(f"    hold fallback: crossed after {press + 1} press(es), "
                  f"cell={before.pos} -> {after.map_id}@{after.pos}")
            if after.map_id == to_map:
                return f"HOLD ({direction}, {press + 1} press(es) from {before.pos})"
            return None   # crossed to unexpected map
        if outcome == "blocked" and press > 8:
            # Jammed into a wall with no hope; abort early rather than spin.
            print(f"    hold fallback: blocked for {press + 1} presses, giving up")
            break
    return None


def _try_navigate_then_probe(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    mtf: object,
    predict: object,
) -> str | None:
    """Exhaustive border probe: navigate to each reachable border cell in direction
    and probe_step once. Covers the case where the player must be positioned
    exactly at the correct border cell before crossing.
    """
    here = snapshot_settled(reader)
    if here is None:
        return None
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return None
    dx, dy = DELTAS[direction]
    candidates = []
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
        arrived = navigate_grid(emu, reader, cell, memory=memory,
                                move_type_fn=mtf, predict=predict)
        if arrived != "arrived":
            print(f"      {cell}: nav={arrived}")
            continue
        before = snapshot_settled(reader)
        if before is None or before.pos != cell:
            continue
        outcome = probe_step(emu, reader, before, direction)
        landed = snapshot_settled(reader)
        land_map = landed.map_id if landed else None
        land_pos = landed.pos if landed else None
        print(f"      {cell} -> {outcome} land={land_map}@{land_pos}")
        if outcome == "transition" and landed is not None and land_map != from_map:
            memory.record_portal(from_map, cell, direction, land_map, True, land_pos)
            if land_map == to_map:
                return f"NAV+PROBE ({direction} from {cell})"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state_path = sys.argv[1] if len(sys.argv) > 1 else "states/post_rival.state"
    fighter_ckpt = "checkpoints/fighter/ppo_fighter_final.zip"

    emu = GbaEmulator(rom)
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

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
    reached = False

    for from_map, to_map, direction in RETURN_CHAIN:
        snap = snapshot_settled(reader)
        if snap is None or snap.map_id != from_map:
            actual = snap.map_id if snap else None
            print(f"\nHOP {from_map} -> {to_map}: not on from_map (actual: {actual})")
            _dump(reader, snap.pos if snap else (0, 0))
            break

        print(f"\n=== HOP {from_map} -> {to_map} (direction={direction}) ===")
        entry_pos = snap.pos
        _dump(reader, entry_pos)
        _heal_party(emu)

        note = None

        # --- Primitive 1: explore_grid sweep ---
        note = _try_explore_grid(emu, reader, memory, from_map, to_map, mtf, predict)

        # Check if we already left from_map unexpectedly (explore_grid may have crossed elsewhere).
        if note is None:
            snap_now = snapshot_settled(reader)
            if snap_now is not None and snap_now.map_id != from_map:
                print(f"    [warn] explore_grid left to {snap_now.map_id}, not {to_map} — abort")
                _dump(reader, snap_now.pos)
                break

        # --- Primitive 2: hold-direction fallback (from wherever explore_grid left us) ---
        if note is None:
            snap_now = snapshot_settled(reader)
            cur_pos = snap_now.pos if snap_now else entry_pos
            print(f"    Trying hold-direction fallback ({direction}) from {cur_pos}...")
            note = _try_hold_direction(emu, reader, memory, from_map, to_map, direction, cur_pos)

        # --- Primitive 3 (last resort): navigate to each border cell + single probe_step ---
        if note is None:
            snap_now = snapshot_settled(reader)
            if snap_now is not None and snap_now.map_id == from_map:
                print(f"    Trying nav+probe fallback...")
                note = _try_navigate_then_probe(
                    emu, reader, memory, from_map, to_map, direction, mtf, predict
                )

        snap_after = snapshot_settled(reader)
        if note is not None and snap_after is not None and snap_after.map_id == to_map:
            results.append(f"{from_map} -> {to_map}: {note}")
            print(f"  => CROSSED: {note}")
        else:
            actual = snap_after.map_id if snap_after else None
            print(f"\n  STALLED on hop {from_map} -> {to_map} "
                  f"(actual map: {actual}). Grid dump:")
            _dump(reader, snap_after.pos if snap_after else entry_pos)
            print(f"\nREACHED lab = False")
            _print_portals(memory)
            sys.exit(1)

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
