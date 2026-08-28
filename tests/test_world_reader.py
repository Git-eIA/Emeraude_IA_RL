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


def test_snapshot_tile_behavior_reads_current_tile() -> None:
    from env.map_grid_reader import (
        BACKUP_MAP_LAYOUT_ADDR,
        MAP_HEADER_ADDR,
        MB_TALL_GRASS,
        _BML_HEIGHT,
        _BML_MAP_PTR,
        _BML_WIDTH,
    )

    emu = FakeEmulator()
    emu.map_group, emu.map_num = 0, 16
    emu.x, emu.y = 3, 7
    base = emu.read_bytes

    # Overlay a minimal coherent map on top of FakeEmulator's reads.
    map_buf = 0x02030000
    layout = 0x0203A000
    primary_ts = 0x0203B000
    primary_attr = 0x08300000
    pw, ph = 25, 34  # -> logical 10x20 (contains x=3,y=7)
    overlay: dict[tuple[int, int], bytes] = {}

    def _u32(addr: int, value: int) -> None:
        overlay[(addr, 4)] = value.to_bytes(4, "little")

    def _u16(addr: int, value: int) -> None:
        overlay[(addr, 2)] = value.to_bytes(2, "little")

    _u32(BACKUP_MAP_LAYOUT_ADDR + _BML_WIDTH, pw)
    _u32(BACKUP_MAP_LAYOUT_ADDR + _BML_HEIGHT, ph)
    _u32(BACKUP_MAP_LAYOUT_ADDR + _BML_MAP_PTR, map_buf)
    _u32(MAP_HEADER_ADDR + 0x00, layout)
    _u32(layout + 0x10, primary_ts)
    _u32(primary_ts + 0x0C, primary_attr)
    # player tile (3,7) -> padded (10,14); metatile 5 -> tall grass
    px, py = 3 + 7, 7 + 7
    _u16(map_buf + 2 * (py * pw + px), 0x0005)
    _u16(primary_attr + 2 * 0x0005, MB_TALL_GRASS)

    def read(addr: int, size: int) -> bytes:
        if (addr, size) in overlay:
            return overlay[(addr, size)]
        return base(addr, size)

    snap = WorldReader(read).snapshot()
    assert snap is not None
    assert snap.tile_behavior == MB_TALL_GRASS


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


def test_grid_reader_exposes_the_map_grid_reader():
    from env.map_grid_reader import MapGridReader

    def read(_addr, _size):
        return b"\x00" * _size

    reader = WorldReader(read)
    assert isinstance(reader.grid_reader, MapGridReader)
    # same instance each access (no re-construction)
    assert reader.grid_reader is reader.grid_reader


def test_world_reader_battle_starting_delegates() -> None:
    def read(addr: int, size: int) -> bytes:
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return (0x0004).to_bytes(2, "little")
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([0])
        return bytes(size)

    assert WorldReader(read).battle_starting() is True


def test_world_and_emerald_reader_public_apis_only_overlap_on_passthroughs() -> None:
    """Review I4: the ROM smokes' _Reader adapters resolve WorldReader BEFORE
    EmeraldReader, so any shared public name is silently shadowed by the world
    side. Today the only shared names are party_hp/party_levels, which
    WorldReader implements as pure passthroughs to its internal EmeraldReader —
    equivalent either way. Pin that set: a NEW overlapping name must be resolved
    consciously (rename it or extend this allowlist with a passthrough proof),
    never absorbed silently by the adapters' resolution order."""
    from env.game_state import EmeraldReader

    world_api = {name for name in dir(WorldReader) if not name.startswith("_")}
    reader_api = {name for name in dir(EmeraldReader) if not name.startswith("_")}
    assert world_api & reader_api == {"party_hp", "party_levels"}
