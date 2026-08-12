"""Decisive de-risk probe: does a RAW hold-DOWN cross Oldale -> route_101?

Both prior tests routed the south crossing through navigate_grid, which returns
'unreachable' for every south-border cell after explore_grid poisons the
MapMemory blocked-set. That is a nav-machinery failure, not proof the connection
is uncrossable. The spec's own premise (user-confirmed) is that the descent is a
STRAIGHT SHOT: hold DOWN on the correct column, no A* routing.

This probe tests exactly that: reach Oldale from route_103 (north entry), then
hold DOWN raw (probe_step, clearing wild battles) straight down the landing
column with NO navigate_grid and NO explore_grid poisoning. _straight_shot
tracks the y axis and gives up if it stalls against a wall, reporting where.

If it crosses to route_101 -> the durable primitive is a raw hold-DOWN, and the
continuous descent works (Approach 1 stands, _cross_toward must NOT use
navigate_grid for ledge legs).
If it stalls -> report the wall row so we know whether column 11 is physically
blocked from the north (genuine architecture problem).

Run (from the main repo):
  cd /Users/_eloi/Projets/Emu
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \\
      .venv/bin/python tools/probe_oldale_straightshot.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator.gba import GbaEmulator
from env.grid_navigator import handle_battle_interruption, probe_step, snapshot_settled
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
    _straight_shot,
)

# Raw hold budget: Oldale is ~20 tall, so 30 DOWN presses covers the full column.
_RAW_MAX_PRESSES = 30


def _raw_hold_down(
    emu: GbaEmulator,
    reader: WorldReader,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
) -> str:
    """Hold DOWN raw from the current cell; report the full trajectory.

    Pure probe_step down (battles cleared each press). No navigate_grid, no
    memory. Returns a verdict string. Prints every landing so a wall row is visible.
    """
    for press in range(_RAW_MAX_PRESSES):
        before = snapshot_settled(reader)
        if before is None:
            continue
        if before.map_id != from_map:
            return f"crossed to {before.map_id}@{before.pos} before press {press}"
        outcome = probe_step(emu, reader, before, "down")
        after = snapshot_settled(reader)
        ap = after.pos if after else None
        am = after.map_id if after else None
        print(f"    raw DOWN press {press + 1}: {before.pos} -> {am}@{ap} ({outcome!r})")
        if after is not None and after.map_id == to_map:
            return f"CROSSED to route_101@{after.pos} at press {press + 1}"
        if after is not None and after.map_id != from_map:
            return f"crossed to UNEXPECTED {after.map_id}@{after.pos} at press {press + 1}"
    final = snapshot_settled(reader)
    return f"NO CROSS after {_RAW_MAX_PRESSES} presses, stuck on {final.map_id if final else None}@{final.pos if final else None}"


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

    # Reach Oldale from route_103 (proven explore_grid sweep). This poisons memory,
    # but the raw hold below does NOT consult memory, so that is irrelevant here.
    print(f"\n=== HOP {ROUTE_103} -> {OLDALE} (explore_grid) ===")
    note = _hop_via_explore_then_scan(emu, reader, memory, ROUTE_103, OLDALE, "down", mtf, predict)
    snap = snapshot_settled(reader)
    if note is None or snap is None or snap.map_id != OLDALE:
        print(f"ERROR: failed to reach Oldale (note={note!r})")
        sys.exit(1)
    landing = snap.pos
    print(f"  On Oldale @ {landing} (entered from route_103/north)")
    _dump(reader, landing)

    # TEST 1: pure raw hold-DOWN straight from the landing column.
    print(f"\n=== TEST 1: raw hold-DOWN from landing {landing} ===")
    _heal_party(emu)
    verdict_raw = _raw_hold_down(emu, reader, OLDALE, ROUTE_101)
    print(f"  RAW verdict: {verdict_raw}")

    final = snapshot_settled(reader)
    crossed_raw = final is not None and final.map_id == ROUTE_101

    # TEST 2 (only if raw stalled and we are still on Oldale): _straight_shot with
    # its stall-patience + battle handling, which may nudge onto a live column.
    verdict_ss = None
    if not crossed_raw and final is not None and final.map_id == OLDALE:
        print(f"\n=== TEST 2: _straight_shot(down) from {final.pos} ===")
        _heal_party(emu)
        verdict_ss = _straight_shot(
            emu, reader, memory, OLDALE, ROUTE_101, "down", final.pos, mtf, predict
        )
        print(f"  _straight_shot verdict: {verdict_ss!r}")
        final = snapshot_settled(reader)
        crossed_raw = final is not None and final.map_id == ROUTE_101

    print(f"\n{'=' * 60}")
    print(f"RAW STRAIGHT-SHOT Oldale->route_101 = {crossed_raw}")
    print(f"  landing column was {landing}")
    print(f"  raw verdict: {verdict_raw}")
    if verdict_ss is not None:
        print(f"  straight_shot verdict: {verdict_ss!r}")
    fm = final.map_id if final else None
    fp = final.pos if final else None
    print(f"  final: {fm}@{fp}")
    sys.exit(0 if crossed_raw else 1)


if __name__ == "__main__":
    main()
