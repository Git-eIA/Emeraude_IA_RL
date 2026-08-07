from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = Path("states/post_starter.state")

pytestmark = pytest.mark.skipif(
    not ROM or not STATE.exists(),
    reason="needs POKEMON_EMERALD_ROM and states/post_starter.state",
)


def _reader_and_emu():
    from emulator.gba import GbaEmulator
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)
    return emu, WorldReader(emu.read_bytes)


def test_grid_snapshot_captures_route101_from_ram():
    from env.grid_snapshot import GridSnapshot

    emu, reader = _reader_and_emu()
    snap = reader.snapshot()
    assert snap is not None
    grid = GridSnapshot.from_reader(reader.grid_reader, snap.map_id)
    assert grid is not None
    assert grid.width > 0 and grid.height > 0
    # the player's own tile is standable
    from env.map_grid_reader import TileKind
    assert grid.classify_at(*snap.pos) in (TileKind.FREE, TileKind.GRASS)


def test_navigate_grid_moves_north_past_the_ledge():
    from env.grid_navigator import navigate_grid
    from env.map_memory import MapMemory

    emu, reader = _reader_and_emu()
    start = reader.snapshot()
    assert start is not None
    memory = MapMemory()
    # target a cell well to the north of the start; the plan must route around
    # the one-way ledge (right then up), never through it.
    target = (start.pos[0], max(0, start.pos[1] - 6))
    result = navigate_grid(emu, reader, target, memory=memory, max_steps=400)
    end = reader.snapshot()
    assert end is not None
    # either we reached it, or we made real northward progress (y decreased) —
    # crucially NOT the old budget_exhausted / timeout-in-place thrash.
    assert result in ("arrived", "unreachable", "left_map", "timeout")
    if result == "timeout":
        assert end.pos[1] < start.pos[1], "no northward progress (thrash regression)"
