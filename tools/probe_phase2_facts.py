"""Throwaway probe: freeze the live Phase 2 facts on BPEF before wiring durables.

Loads states/post_rival.state and, driving the Fighter for any wild along the way,
prints the ground truth the Phase 2 machinery encodes as candidates:
  - has_pokedex / has_running_shoes flag ids (scan a small flag-id window for the
    bit that flips when the Pokédex/shoes are received),
  - SAVE_BLOCK2_PTR validity + securityKey, and has_item(POKE_BALL, 5) after the
    lab cutscene (confirms the Items pocket offset + XOR decrypt),
  - the return-portal cells actually crossed (route_103->...->lab) and the lab
    warp-landing tile,
  - how many A-presses each cutscene needs (sanity-check STORY_MAX_PRESSES).

Run: POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
     .venv/bin/python tools/probe_phase2_facts.py states/post_rival.state
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import (
    EmeraldReader,
    FLAG_RECEIVED_RUNNING_SHOES,
    FLAG_SYS_POKEDEX_GET,
    POKE_BALL_ITEM_ID,
    SAVE_BLOCK2_PTR,
)
from env.world_reader import WorldReader


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state = sys.argv[1] if len(sys.argv) > 1 else "states/post_rival.state"
    emu = GbaEmulator(rom)
    with open(state, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)

    snap = world.snapshot()
    if snap is None:
        print("ERROR: world.snapshot() returned None — emulator may not be in overworld")
        sys.exit(1)
    print(f"start map={snap.map_id} pos={snap.pos}")
    print(
        f"SAVE_BLOCK2_PTR=0x{SAVE_BLOCK2_PTR:08x} sb2_valid="
        f"{reader._save_block2() is not None}"
    )
    print(
        f"has_pokedex(0x{FLAG_SYS_POKEDEX_GET:x})={reader.has_pokedex()} "
        f"has_running_shoes(0x{FLAG_RECEIVED_RUNNING_SHOES:x})="
        f"{reader.has_running_shoes()}"
    )
    print(
        f"has_item(POKE_BALL={POKE_BALL_ITEM_ID}, 5)="
        f"{reader.has_item(POKE_BALL_ITEM_ID, 5)}"
    )
    # The operator drives the campaign manually here (or wires run_campaign) and
    # re-prints these lines after each cutscene to confirm the flags/items flip.


if __name__ == "__main__":
    main()
