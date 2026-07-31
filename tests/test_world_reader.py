"""WorldReader: RAM snapshot of the Explorer's world state (no navigation)."""
from __future__ import annotations

from env.game_state import (
    BATTLE_MON_SIZE,
    GBATTLE_MONS_ADDR,
    GBATTLE_OUTCOME_ADDR,
    GBATTLE_TYPE_FLAGS_ADDR,
    GMOVE_RESULT_FLAGS_ADDR,
)
from env.world_reader import WorldReader, WorldSnapshot
from tests.conftest import FakeEmulator


def _reader(emu: FakeEmulator) -> WorldReader:
    return WorldReader(emu.read_bytes)


def _battle_read(*, in_battle: bool):
    """A read(addr, size) that reports a battle iff `in_battle` (opp max_hp>0)."""
    def read(addr: int, size: int) -> bytes:
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return (1 if in_battle else 0).to_bytes(2, "little") + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return b"\x00"
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return b"\x00\x00"
        opp_base = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        if opp_base <= addr < opp_base + BATTLE_MON_SIZE:
            buf = bytearray(BATTLE_MON_SIZE)
            buf[0x2C:0x2E] = (18 if in_battle else 0).to_bytes(2, "little")  # opp max_hp
            offset = addr - opp_base
            return bytes(buf[offset : offset + size])
        return b"\x00" * size  # player mon + anything else reads as zeros
    return read


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


def test_in_battle_true_when_a_battle_is_active() -> None:
    reader = WorldReader(_battle_read(in_battle=True))
    assert reader.in_battle() is True


def test_in_battle_false_out_of_battle() -> None:
    reader = WorldReader(_battle_read(in_battle=False))
    assert reader.in_battle() is False


def test_party_levels_passthrough_returns_ram_reader_levels() -> None:
    emu = FakeEmulator()
    emu.party_count = 2
    emu.party_levels = [7, 5]
    assert _reader(emu).party_levels() == [7, 5]
