"""map_traveler: ROM smoke test — real emulator wiring (gated on POKEMON_EMERALD_ROM)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")

# states/ lives in the main repo, not the worktree; resolve it absolutely.
_STATE = Path(__file__).resolve().parents[2] / "Emu" / "states" / "initial.state"


@pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")
def test_travel_same_map_arrives_on_real_rom() -> None:
    from emulator.gba import GbaEmulator
    from env.map_memory import MapMemory
    from env.map_traveler import travel_to
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    emu.load_state(_STATE.read_bytes())
    reader = WorldReader(emu.read_bytes)
    snap = reader.snapshot()
    assert snap is not None

    # Same-map travel to the current cell must arrive immediately (delegates to
    # navigate_grid, which returns "arrived" when pos == target). This exercises
    # the plan_route([here]) + map_id == goal_map branch on the real emulator.
    result = travel_to(
        emu, reader, MapMemory(),
        goal_map=snap.map_id, goal_cell=snap.pos,
    )
    assert result == "arrived"
