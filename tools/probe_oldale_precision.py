"""Final decisive de-risk probe: memory-free precision-walk crosses Oldale -> route_101?

Findings so far on map (0,10) [entered from route_103/north at (11,0)]:
  - raw hold-DOWN stalls: column 11 is walled at (11,9); it is NOT a straight shot.
  - navigate_grid returns 'unreachable' for every south-border cell because
    explore_grid poisons the MapMemory blocked-set.
  - plan_path_grid (raw snapshot, NO memory) reports all 12 south-border cells
    (x=4..15, y=19) reachable — the map IS geometrically crossable south.

This probe tests the remaining viable primitive: a memory-free precision walk
(plan_path_grid + 4-frame steps, the same primitive that crosses Littleroot->lab)
to each south-border cell, then one probe_step DOWN. The known connection column
(from the reload-recorded portal) is x=11, so try x=11 first.

If a precision walk reaches a south cell and DOWN crosses to route_101 -> the
durable _cross_toward must use precision-walk (NOT hold-DOWN, NOT navigate_grid)
for this ledge/wall leg, and Approach 1 stands with a primitive swap.
If EVERY reachable south cell fails to cross -> the north-entry position cannot
reach the live connection column; Approach 1 needs rework (escalate).

Run (from the main repo):
  cd /Users/_eloi/Projets/Emu
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \\
      .venv/bin/python tools/probe_oldale_precision.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator.gba import GbaEmulator
from env.grid_navigator import (
    DELTAS,
    handle_battle_interruption,
    plan_path_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory
from env.world_reader import WorldReader
from tools.probe_return_portals import (
    OLDALE,
    ROUTE_101,
    ROUTE_103,
    _heal_party,
    _hop_via_explore_then_scan,
    _load_state,
    _precision_walk_to,
)

_STANDABLE = {TileKind.FREE, TileKind.GRASS}


def _precision_south_cross(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    mtf: object,
    predict: object,
) -> tuple[int, int] | None:
    """Memory-free precision walk to each south-border cell, then probe DOWN.

    Enumerates south-border cells via plan_path_grid on the raw snapshot (no
    memory), tries the known connection column (x=11) first, then the rest
    ordered by distance from x=11. Returns the crossing from_cell or None.
    """
    here = snapshot_settled(reader)
    if here is None:
        return None
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return None
    dx, dy = DELTAS["down"]
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
    # Try the known connection column (x=11) first, then nearest-to-11.
    border.sort(key=lambda c: (abs(c[0] - 11), -c[1]))
    print(f"  precision south-border candidates [{len(border)}]: {border}")
    for cell in border:
        handle_battle_interruption(emu, reader, mtf, predict)
        snap_now = GridSnapshot.from_reader(reader.grid_reader, from_map)
        if snap_now is None:
            continue
        ok = _precision_walk_to(emu, reader, snap_now, cell, from_map)
        here_now = snapshot_settled(reader)
        cur = here_now.pos if here_now else None
        cur_map = here_now.map_id if here_now else None
        if here_now is not None and cur_map == to_map:
            print(f"    {cell}: precision-walk itself crossed -> {to_map}@{cur}")
            memory.record_portal(from_map, cell, "down", to_map, True, cur)
            return cell
        if not ok or cur_map != from_map or cur != cell:
            print(f"    {cell}: walk failed (pos={cur} map={cur_map})")
            continue
        before = snapshot_settled(reader)
        if before is None:
            continue
        outcome = probe_step(emu, reader, before, "down")
        after = snapshot_settled(reader)
        am = after.map_id if after else None
        ap = after.pos if after else None
        print(f"    {cell}: reached, DOWN -> {am}@{ap} ({outcome!r})")
        if after is not None and am == to_map:
            memory.record_portal(from_map, cell, "down", to_map, True, ap)
            return cell
    return None


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    fighter_ckpt = "checkpoints/fighter/ppo_fighter_final.zip"

    emu = GbaEmulator(rom)
    _load_state(emu, "states/post_rival.state")
    reader = WorldReader(emu.read_bytes)
    memory = MapMemory()

    print(f"Loading Fighter from {fighter_ckpt}...")
    model = PPO.load(fighter_ckpt, device="cpu")
    mtf = make_move_type_fn(emu)

    def predict(obs: object) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    _heal_party(emu)
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != ROUTE_103:
        print(f"ERROR: expected ROUTE_103, got {snap.map_id if snap else None}")
        sys.exit(1)
    print(f"Start: map={snap.map_id} pos={snap.pos}")

    print(f"\n=== HOP {ROUTE_103} -> {OLDALE} (explore_grid) ===")
    note = _hop_via_explore_then_scan(emu, reader, memory, ROUTE_103, OLDALE, "down", mtf, predict)
    snap = snapshot_settled(reader)
    if note is None or snap is None or snap.map_id != OLDALE:
        print(f"ERROR: failed to reach Oldale (note={note!r})")
        sys.exit(1)
    print(f"  On Oldale @ {snap.pos} (entered from route_103/north)")

    print(f"\n=== CRUX: memory-free precision walk {OLDALE} -> {ROUTE_101} ===")
    _heal_party(emu)
    crossed = _precision_south_cross(emu, reader, memory, OLDALE, ROUTE_101, mtf, predict)

    final = snapshot_settled(reader)
    ok = crossed is not None and final is not None and final.map_id == ROUTE_101
    print(f"\n{'=' * 60}")
    print(f"PRECISION-WALK Oldale->route_101 = {ok}")
    if ok:
        print(f"  crossing cell = {crossed}, landed route_101@{final.pos}")
        p = memory.portal(OLDALE, ROUTE_101)
        if p is not None:
            print(f"  _PortalSeed({OLDALE}, {p.from_cell}, 'down', "
                  f"{ROUTE_101}, {p.reversible}, {p.to_cell}),")
    else:
        print(f"  final: {final.map_id if final else None}@{final.pos if final else None}"
              f" — north-entry cannot reach the live south connection; Approach 1 needs rework")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
