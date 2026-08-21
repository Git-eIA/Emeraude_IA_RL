"""Gated ROM smoke: run_shoes_leg delivers the running shoes end to end.

From a healthy (post-release) post_pokedex.state: exit the lab, walk north into the
scripted mom interception, drain the event until FLAG 0x112 + town_state==4, verify
control, and dump states/post_shoes.state — then reload the dump and re-verify control
(anti-false-lock pin: the dump must be healthy, not just the live session). No Fighter:
Littleroot and the lab have no wild grass, so no battle can interrupt the leg.
Double-skips without ROM / post_pokedex.state. Run AFTER test_pokedex_return_rom.py,
which re-dumps post_pokedex.state healthy (the pre-existing dump was mid-cutscene).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = "states/post_pokedex.state"

pytestmark = pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")


def test_run_shoes_leg_delivers_the_running_shoes() -> None:
    from emulator.gba import GbaEmulator
    from env.campaign import run_shoes_leg
    from env.game_state import EmeraldReader
    from env.map_memory import MapMemory
    from env.world_reader import WorldReader
    from tests.conftest import control_returns

    # Checked at RUN time, not collection time: in a fresh single-session run,
    # test_pokedex_return_rom.py dumps this state AFTER collection happened —
    # a collection-time skipif would silently skip this smoke forever.
    if not Path(STATE).exists():
        pytest.skip("post_pokedex.state missing (run test_pokedex_return_rom.py first)")

    emu = GbaEmulator(ROM)
    with open(STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)
    memory = MapMemory()

    # Precondition: the consumed dump must be the HEALTHY post-release state
    # (fail loudly here rather than deep inside run_shoes_leg).
    assert reader.has_pokedex() is True
    assert reader.has_poke_balls(5) is True
    assert reader.birch_lab_state() == 5

    # run_shoes_leg needs snapshot/grid_reader (world) AND flags/vars (reader);
    # a thin adapter forwards both, matching test_pokedex_return_rom.py.
    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_shoes_leg(emu, _Reader(), memory)

    assert result == "shoes_delivered", result
    assert reader.has_running_shoes() is True
    ps = reader.player_state()
    assert ps is not None and ps.town_state == 4, ps

    # Dump FIRST, then reload and verify control on the reloaded session (the pin
    # must prove the DUMP is healthy; verification never mutates the dumped state).
    state_bytes = emu.save_state()
    Path("states/post_shoes.state").write_bytes(state_bytes)

    emu2 = GbaEmulator(ROM)
    emu2.load_state(state_bytes)
    emu2.step(0, 4)
    assert control_returns(emu2), "reloaded post_shoes.state is control-locked"
