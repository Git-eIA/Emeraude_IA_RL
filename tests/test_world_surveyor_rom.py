"""ROM-gated smoke for survey_world on a real open-map savestate.

Double skip: POKEMON_EMERALD_ROM unset OR states/open_map.state missing.
The state and ROM are LOCAL, gitignored artifacts in the main repo ~/Projets/Emu.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.gba import GbaEmulator
from env.local_navigator import DIRECTIONS, WallMap
from env.map_memory import MapMemory
from env.world_reader import WorldReader
from env.world_surveyor import survey_world

POKEMON_EMERALD_ROM = os.environ.get("POKEMON_EMERALD_ROM")

# states/ lives in the main repo, not the worktree — resolve absolutely.
_STATE = Path.home() / "Projets" / "Emu" / "states" / "open_map.state"


@pytest.mark.skipif(not POKEMON_EMERALD_ROM, reason="requires POKEMON_EMERALD_ROM")
@pytest.mark.skipif(not _STATE.exists(), reason="requires states/open_map.state")
def test_survey_world_smoke_is_coherent_and_learns() -> None:
    emulator = GbaEmulator(POKEMON_EMERALD_ROM)
    emulator.load_state(_STATE.read_bytes())
    reader = WorldReader(emulator.read_bytes)

    start = reader.snapshot()
    assert start is not None, "open_map.state should sit on a readable map"
    start_map = start.map_id
    start_cell = start.pos

    memory = MapMemory()
    wallmap = WallMap()

    report = survey_world(emulator, reader, memory, wallmap, max_maps=2)

    # Report is coherent: at least the starting map was attempted.
    assert report.surveyed or report.failed

    # Learning is externally visible via public API: wall learned or portal recorded.
    learned_wall = any(
        wallmap.is_blocked(start_map, start_cell, d) for d in DIRECTIONS
    )
    learned_portal = bool(memory.outgoing_portals(start_map)) if report.surveyed else False
    assert learned_wall or learned_portal or report.failed, (
        "survey_world should learn a wall, record a portal, or record a failure"
    )
