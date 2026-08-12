"""Throwaway: directed DOWN descent route_101 -> Littleroot -> lab from post_starter.

Replaces the greedy explore_grid sweep (which exits route_101 NORTH to Oldale) with the
directional _column_scan (southmost border + press DOWN). Then _warp_scan_up for the lab.
Stdout only; nothing committed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.grid_navigator import snapshot_settled
from env.map_memory import MapMemory
from env.world_reader import WorldReader
from tools.probe_return_portals import (
    LAB,
    LITTLEROOT,
    ROUTE_101,
    _column_scan,
    _load_state,
    _warp_scan_up,
)


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    _load_state(emu, "states/post_starter.state")
    world = WorldReader(emu.read_bytes)
    game = EmeraldReader(emu.read_bytes)

    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn

    model = PPO.load("checkpoints/fighter/ppo_fighter_final.zip", device="cpu")
    mtf = make_move_type_fn(emu)

    def predict(obs: object) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    memory = MapMemory()
    here = snapshot_settled(world)
    print(f"start: {here.map_id if here else None} @ {here.pos if here else None}")

    note = _column_scan(emu, world, memory, ROUTE_101, LITTLEROOT, "down", mtf, predict)
    here = snapshot_settled(world)
    on_littleroot = here is not None and here.map_id == LITTLEROOT
    print(f"route_101 --DOWN--> note={note!r} now={here.map_id if here else None}@{here.pos if here else None} on_littleroot={on_littleroot}")

    if on_littleroot:
        note = _warp_scan_up(emu, world, memory, LITTLEROOT, LAB, mtf, predict)
        here = snapshot_settled(world)
        on_lab = here is not None and here.map_id == LAB
        print(f"Littleroot --UP--> note={note!r} now={here.map_id if here else None}@{here.pos if here else None} on_lab={on_lab}")
    else:
        on_lab = False
        print("Littleroot --UP--> SKIPPED (not on Littleroot)")

    print(f"RESULT reached_littleroot={on_littleroot} reached_lab={on_lab}")
    sys.exit(0 if on_lab else 1)


if __name__ == "__main__":
    main()
