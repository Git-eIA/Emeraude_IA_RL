"""Focused de-risk probe: can Oldale -> route_101 be crossed CONTINUOUSLY (no reload)?

The main return probe (tools/probe_return_portals.py) crossed this hop via a
STATE-RELOAD to post_starter.state, claiming the Oldale south border "only
activates on south-entry". reach_map runs continuously from post_rival.state and
CANNOT reload mid-descent, so this probe settles whether a plain continuous DOWN
crossing works after entering Oldale from route_103 (north).

Method:
  1. load post_rival.state (player on route_103), heal, load Fighter.
  2. cross route_103 -> Oldale via explore_grid (the proven northbound sweep).
  3. WITHOUT any reload, exhaustively try EVERY reachable south-border cell of
     Oldale (southmost-first, NO candidate cap), navigate to it, hold DOWN, and
     report the landing map. First cell that lands on route_101 = SUCCESS.

Run (from the main repo):
  cd /Users/_eloi/Projets/Emu
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \\
      .venv/bin/python tools/probe_oldale_south.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
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
from env.map_memory import MapMemory
from env.world_reader import WorldReader
from tools.probe_return_portals import (
    OLDALE,
    ROUTE_101,
    ROUTE_103,
    _dump,
    _heal_party,
    _hop_via_explore_then_scan,
    _load_state,
)

_STANDABLE = {TileKind.FREE, TileKind.GRASS}
_HOLD_PRESSES = 20


def _exhaustive_south_cross(
    emu: GbaEmulator,
    reader: WorldReader,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    mtf: object,
    predict: object,
) -> tuple[int, int] | None:
    """Try EVERY reachable south-border cell (southmost-first, no cap), hold DOWN.

    Returns the from_cell that crossed to to_map, or None if no cell crossed.
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
    border.sort(key=lambda c: -c[1])  # southmost first
    print(f"  south-border reachable cells [{len(border)}]: {border}")
    for cell in border:
        handle_battle_interruption(emu, reader, mtf, predict)
        arrived = navigate_grid(emu, reader, cell, memory=memory, move_type_fn=mtf, predict=predict)
        if arrived != "arrived":
            print(f"    {cell}: nav={arrived!r}")
            continue
        for press in range(_HOLD_PRESSES):
            handle_battle_interruption(emu, reader, mtf, predict)
            before = snapshot_settled(reader)
            if before is None or before.map_id != from_map:
                break
            probe_step(emu, reader, before, "down")
            after = snapshot_settled(reader)
            if after is None:
                continue
            if after.map_id != from_map:
                print(f"    {cell} press {press + 1}: crossed -> {after.map_id}@{after.pos}")
                if after.map_id == to_map:
                    memory.record_portal(from_map, cell, "down", after.map_id, True, after.pos)
                    return cell
                return None  # crossed to an unexpected map
        print(f"    {cell}: no transition after {_HOLD_PRESSES} presses")
    return None


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    rival_state = "states/post_rival.state"
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
    if snap is None or snap.map_id != ROUTE_103:
        print(f"ERROR: expected ROUTE_103, got {snap.map_id if snap else None}")
        sys.exit(1)
    print(f"Start: map={snap.map_id} pos={snap.pos}")

    # HOP 1: route_103 -> Oldale (proven explore_grid sweep).
    print(f"\n=== HOP {ROUTE_103} -> {OLDALE} (explore_grid) ===")
    note = _hop_via_explore_then_scan(emu, reader, memory, ROUTE_103, OLDALE, "down", mtf, predict)
    snap = snapshot_settled(reader)
    if note is None or snap is None or snap.map_id != OLDALE:
        print(f"ERROR: failed to reach Oldale (note={note!r}, map={snap.map_id if snap else None})")
        sys.exit(1)
    print(f"  On Oldale @ {snap.pos} (entered from route_103/north)")

    # Reproduce the design's primitive: explore_grid repositions the player (the
    # ledge-aware greedy sweep can traverse one-way ledges that plain A* cannot),
    # so the south-connection column (x~11) may become reachable from the
    # post-explore position even though it is unreachable from the (11,0) north entry.
    print(f"\n=== explore_grid on Oldale (reposition before south cross) ===")
    _heal_party(emu)
    eg = explore_grid(emu, reader, memory, OLDALE, move_type_fn=mtf, predict=predict)
    snap_eg = snapshot_settled(reader)
    print(f"  explore_grid -> {eg!r} | now {snap_eg.map_id if snap_eg else None}"
          f"@{snap_eg.pos if snap_eg else None}")
    for p in memory.outgoing_portals(OLDALE):
        print(f"    portal {p.from_cell} {p.direction!r} -> {p.to_map}@{p.to_cell} rev={p.reversible}")
    # explore_grid may have auto-crossed into route_101 already.
    if snap_eg is not None and snap_eg.map_id == ROUTE_101:
        print("  explore_grid auto-crossed into route_101!")
        crossed = snap_eg.pos
        final = snap_eg
        _report(memory, crossed, final)
        return

    # The crux test: exhaustive continuous DOWN crossing, NO reload.
    print(f"\n=== CRUX: {OLDALE} -> {ROUTE_101} continuous DOWN (no reload) ===")
    _heal_party(emu)
    snap = snapshot_settled(reader)
    if snap is None or snap.map_id != OLDALE:
        print(f"ERROR: not on Oldale after explore_grid (map={snap.map_id if snap else None})")
        sys.exit(1)
    _dump(reader, snap.pos)
    crossed = _exhaustive_south_cross(emu, reader, memory, OLDALE, ROUTE_101, mtf, predict)
    _report(memory, crossed, snapshot_settled(reader))


def _report(memory: MapMemory, crossed: tuple[int, int] | None, final: object) -> None:
    """Print the crux verdict and exit 0 (success) / 1 (failure)."""
    ok = crossed is not None and final is not None and final.map_id == ROUTE_101
    print(f"\n{'=' * 60}")
    print(f"CONTINUOUS Oldale->route_101 = {ok}")
    if ok:
        print(f"  crossing cell = {crossed}, landed route_101@{final.pos}")
        p = memory.portal(OLDALE, ROUTE_101)
        if p is not None:
            print(f"  _PortalSeed({OLDALE}, {p.from_cell}, 'down', "
                  f"{ROUTE_101}, {p.reversible}, {p.to_cell}),")
    else:
        fm = final.map_id if final else None
        print(f"  final map={fm} — continuous descent FAILED; reload workaround was real")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
