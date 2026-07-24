"""WorldReader: RAM snapshot of the Explorer's world state (no navigation)."""
from __future__ import annotations

from env.game_state import EmeraldReader
from env.world_reader import WorldReader, WorldSnapshot
from tests.conftest import FakeEmulator


def _reader(emu: FakeEmulator) -> WorldReader:
    return WorldReader(EmeraldReader(emu.read_bytes))


def test_snapshot_reads_map_id_and_position() -> None:
    emu = FakeEmulator()
    emu.map_group, emu.map_num = 0, 16  # Route 101
    emu.x, emu.y = 3, 7
    snap = _reader(emu).snapshot()
    assert isinstance(snap, WorldSnapshot)
    assert snap.map_id == (0, 16)
    assert snap.pos == (3, 7)


def test_tile_behavior_is_none_until_probed() -> None:
    snap = _reader(FakeEmulator()).snapshot()
    assert snap is not None
    assert snap.tile_behavior is None


def test_snapshot_is_none_while_save_blocks_relocate() -> None:
    emu = FakeEmulator()
    emu._sb1 = 0x00000000  # out of EWRAM range -> player_state() returns None
    assert _reader(emu).snapshot() is None
