"""Throwaway capture tool: spawn the flag-gated route_103 rival and mint trainer_battle.state.

The rival (gObjectEvents template obj[10]: gfx 0x40, tile (7,3), trainerType 0) is HIDDEN by
SaveBlock1 hide-flag 0x0382. This tool clears that flag in RAM (cheat-spawn, for the ARTIFACT
only), forces a map reload so the object respawns, navigates the sand to a cell adjacent to
(7,3), and spams A through any dialogue until a trainer battle starts -- then saves
states/trainer_battle.state. A bounded retry loop reloads the state fresh (party restored) and
perturbs RNG to absorb wild-encounter attrition on the northward trip.

The legit path (Birch's post-lab dialog clearing 0x0382) is the deferred scripted campaign
(Option B); this tool only produces the artifact so the two trainer-battle ROM smokes become
load-bearing.

Run in the MAIN repo:
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
    .venv/bin/python tools/capture_trainer_battle.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import SAVE_BLOCK1_PTR
from env.map_grid_reader import TileKind
from env.world_reader import WorldReader

INPUT_STATE = "states/route_103_reached.state"
OUTPUT_STATE = "states/trainer_battle.state"
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"

ROUTE_103 = (0, 18)
RIVAL_FLAG = 0x0382
RIVAL_TILE = (7, 3)

_FLAGS_OFF = 0x1270           # SaveBlock1.flags[] offset (Emerald)
_STANDABLE = {TileKind.FREE, TileKind.GRASS}
_LOST = ("battle_lost", "battle_timeout", "battle_interrupted")

_RELOAD_BUDGET = 30           # bounded map-crossing loop
_NAV_MAX = 250               # bounded nav loop
_TALK_A_PRESSES = 30         # A-spam budget through pre-battle dialogue
_MAX_ATTEMPTS = 5            # retry-RNG budget
_RNG_PERTURB_FRAMES = 17     # idle frames * attempt to vary wild rolls


def _clear_flag(emu, flag_id: int) -> int:
    """Clear a SaveBlock1 flag bit in RAM; return the bit value read back (0 = cleared)."""
    ptr = int.from_bytes(emu.read_bytes(SAVE_BLOCK1_PTR, 4), "little")
    addr = ptr + _FLAGS_OFF + flag_id // 8
    val = emu.read_bytes(addr, 1)[0] & ~(1 << (flag_id % 8))
    emu._core._core.rawWrite8(emu._core._core, addr, -1, val)
    return (emu.read_bytes(addr, 1)[0] >> (flag_id % 8)) & 1


def main() -> int:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    with open(INPUT_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    here = reader.snapshot()
    print(f"start map={here.map_id} pos={here.pos}")
    bit = _clear_flag(emu, RIVAL_FLAG)
    print(f"cleared flag 0x{RIVAL_FLAG:04x} -> bit now {bit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
