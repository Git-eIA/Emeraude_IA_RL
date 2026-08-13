"""Gated ROM smoke: the retail BPEF Pokédex flag id is load-bearing.

Loads states/lab_arrival.state (player warped into Birch's lab, VAR_BIRCH_LAB_STATE == 4,
pre-cutscene, has_pokedex False). The lab OnFrame GivePokedexEvent auto-fires: the player is
auto-walked and A-spam advances the dialogue until VAR_BIRCH_LAB_STATE flips 4 -> 5 (the
GivePokedex script completes and the Pokédex is delivered). Asserts has_pokedex() is then
True -- this only holds when FLAG_SYS_POKEDEX_GET carries the correct retail id (0x861); the
old 0x801 reads a clear bit and returns False even though the Pokédex was delivered.

The pure tests in test_game_state.py exercise FLAG_SYS_POKEDEX_GET symbolically, so they
pass for any numeric value; only this smoke proves the concrete id decodes a real armed
state. Double-skips without ROM / lab_arrival.state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
LAB_STATE = "states/lab_arrival.state"

# pret/pokeemerald include/constants/vars.h -- the lab OnFrame gate.
VAR_BIRCH_LAB_STATE = 0x4084

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(LAB_STATE).exists(), reason="lab_arrival.state missing"),
]


def test_pokedex_flag_id_matches_retail_cutscene() -> None:
    from emulator.gba import GbaEmulator
    from env.game_state import EmeraldReader

    try:
        from pokemon_env import buttons
        key_a = buttons.KEY_A
    except Exception:
        key_a = 1

    emu = GbaEmulator(ROM)
    with open(LAB_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = EmeraldReader(emu.read_bytes)

    def birch_state() -> int | None:
        sb1 = reader._save_block1()
        return None if sb1 is None else reader._var(sb1, VAR_BIRCH_LAB_STATE)

    # Precondition: armed, pre-cutscene, Pokédex not yet flagged.
    assert birch_state() == 4, birch_state()
    assert reader.has_pokedex() is False

    # Let the OnFrame cutscene auto-walk (no input), then A-spam its dialogue to completion.
    for _ in range(30):
        emu.step(0, 8)
        if birch_state() != 4:
            break
    for _ in range(400):
        emu.step(key_a, 4)
        emu.step(0, 12)
        if birch_state() == 5:
            break

    # The GivePokedex script ran to completion regardless of the detector id...
    assert birch_state() == 5, birch_state()
    # ...so has_pokedex() must be True -- load-bearing on the retail flag id (0x861).
    assert reader.has_pokedex() is True
