"""ROM-gated smoke for map_map on a real open map.

Loads states/open_map.state (produced by tools/capture_open_map.py), runs a
short survey, and asserts a load-bearing result: a legal outcome without
crashing AND that learning actually happened, checked only through externally
visible state (reached is private to map_map): the player moved, or a wall was
learned, or a portal was recorded.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.local_navigator import DIRECTIONS, WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory
from env.world_reader import WorldReader

POKEMON_EMERALD_ROM = os.environ.get("POKEMON_EMERALD_ROM")
# states/ lives in the main repo, not the worktree — resolve absolutely.
_STATE = Path.home() / "Projets" / "Emu" / "states" / "open_map.state"


@pytest.mark.skipif(not POKEMON_EMERALD_ROM, reason="requires POKEMON_EMERALD_ROM")
@pytest.mark.skipif(not _STATE.exists(), reason="requires states/open_map.state")
def test_map_map_learns_something_on_a_real_open_map():
    emulator = GbaEmulator(POKEMON_EMERALD_ROM)
    emulator.load_state(_STATE.read_bytes())
    reader = WorldReader(EmeraldReader(emulator.read_bytes))

    start = reader.snapshot()
    assert start is not None, "open_map.state should sit on a readable map"
    target_map = start.map_id
    start_cell = start.pos

    memory = MapMemory()
    wallmap = WallMap()
    result = map_map(emulator, reader, memory, wallmap, target_map, max_steps=40)

    assert result in ("complete", "budget_exhausted", "left_map")

    moved = reader.snapshot() is not None and reader.snapshot().pos != start_cell
    learned_wall = any(
        wallmap.is_blocked(target_map, start_cell, d) for d in DIRECTIONS
    )
    recorded_portal = len(memory.edges()) > 0
    assert moved or learned_wall or recorded_portal, (
        "survey should move, learn a wall, or record a portal on a real map"
    )
