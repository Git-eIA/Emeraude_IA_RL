"""Gated ROM smoke: the Balls pocket detector reads the real SaveBlock1 offset.

The unit tests plant a ball at _POKEBALLS_POCKET_OFFSET and read it back — circular:
they prove the scan logic, not that 0x650 is where Emerald actually keeps Poke Balls.
This is the ONLY non-circular proof of the offset. post_pokedex.state is dumped by the
A2 smoke at the FIRST frame has_pokedex() is True — ~50 A-presses BEFORE the same lab
GivePokedex cutscene runs its AddBagItem(POKE_BALL, 5). So we finish the cutscene tail
(bounded A-spam) until the balls land, then assert has_poke_balls(5) True (and 6 False,
pinning the delivered count at exactly 5) against real post-delivery RAM.
Double-skips without ROM / post_pokedex.state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = "states/post_pokedex.state"

# The Pokedex flag flips before the AddBagItem; the balls landed ~55 A-presses
# later in the probe. Bound generously so the cutscene tail always completes.
_MAX_ASPAM = 120

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(STATE).exists(), reason="post_pokedex.state missing"),
]


def test_post_pokedex_cutscene_delivers_five_poke_balls() -> None:
    from emulator import buttons
    from emulator.gba import GbaEmulator
    from env.game_state import EmeraldReader

    emu = GbaEmulator(ROM)
    with open(STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

    reader = EmeraldReader(emu.read_bytes)
    assert reader.has_pokedex() is True  # dumped mid-cutscene, dex already flagged

    # Finish the GivePokedex cutscene tail until the balls are delivered.
    for _ in range(_MAX_ASPAM):
        if reader.has_poke_balls(5):
            break
        emu.step(buttons.KEY_A, 8)
        emu.step(0, 8)

    assert reader.has_poke_balls(5) is True
    assert reader.has_poke_balls(6) is False  # exactly 5 delivered, offset is right
