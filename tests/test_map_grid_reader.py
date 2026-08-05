"""MapGridReader: decode the loaded Emerald map grid from RAM (no ROM)."""
from __future__ import annotations

import pytest  # noqa: F401

from env.map_grid_reader import (
    BACKUP_MAP_LAYOUT_ADDR,
    MAP_HEADER_ADDR,
    MAP_OFFSET,
    MB_JUMP_EAST,
    MB_TALL_GRASS,
    TileKind,
    _BML_HEIGHT,
    _BML_MAP_PTR,
    _BML_WIDTH,
    _CORRUPTION_ID,
    MapGridReader,
)

_MAP_BUF = 0x02030000  # arbitrary in-EWRAM buffer for the tile array
_LAYOUT = 0x0203A000
_PRIMARY_TS = 0x0203B000
_SECONDARY_TS = 0x0203B100
_PRIMARY_ATTR = 0x08300000
_SECONDARY_ATTR = 0x08300800
_ML_PRIMARY_TILESET_PTR = 0x10
_ML_SECONDARY_TILESET_PTR = 0x14
_TS_METATILE_ATTR_PTR = 0x0C
_MH_MAP_LAYOUT_PTR = 0x00


class FakeMem:
    """Dict-backed read(addr,size) with little-endian write helpers."""

    def __init__(self) -> None:
        self._m: dict[int, int] = {}

    def write_u16(self, addr: int, value: int) -> None:
        self._m[addr] = value & 0xFF
        self._m[addr + 1] = (value >> 8) & 0xFF

    def write_u32(self, addr: int, value: int) -> None:
        for i in range(4):
            self._m[addr + i] = (value >> (8 * i)) & 0xFF

    def read(self, addr: int, size: int) -> bytes:
        return bytes(self._m.get(addr + i, 0) for i in range(size))


def _build(pw: int, ph: int) -> FakeMem:
    """Wire gBackupMapLayout -> buffer + gMapHeader -> tileset attr tables."""
    mem = FakeMem()
    mem.write_u32(BACKUP_MAP_LAYOUT_ADDR + _BML_WIDTH, pw)
    mem.write_u32(BACKUP_MAP_LAYOUT_ADDR + _BML_HEIGHT, ph)
    mem.write_u32(BACKUP_MAP_LAYOUT_ADDR + _BML_MAP_PTR, _MAP_BUF)
    mem.write_u32(MAP_HEADER_ADDR + _MH_MAP_LAYOUT_PTR, _LAYOUT)
    mem.write_u32(_LAYOUT + _ML_PRIMARY_TILESET_PTR, _PRIMARY_TS)
    mem.write_u32(_LAYOUT + _ML_SECONDARY_TILESET_PTR, _SECONDARY_TS)
    mem.write_u32(_PRIMARY_TS + _TS_METATILE_ATTR_PTR, _PRIMARY_ATTR)
    mem.write_u32(_SECONDARY_TS + _TS_METATILE_ATTR_PTR, _SECONDARY_ATTR)
    return mem


def _set_tile(mem: FakeMem, pw: int, x: int, y: int, entry: int) -> None:
    """Write a raw u16 map entry at padded coords (x,y)."""
    mem.write_u16(_MAP_BUF + 2 * (y * pw + x), entry)


def _set_behavior(mem: FakeMem, metatile_id: int, behavior: int) -> None:
    if metatile_id < 0x200:
        mem.write_u16(_PRIMARY_ATTR + 2 * metatile_id, behavior)
    else:
        mem.write_u16(_SECONDARY_ATTR + 2 * (metatile_id - 0x200), behavior)


def test_dimensions_strips_border_padding() -> None:
    # padded 25x34 -> logical 10x20 (route_101 shape)
    mem = _build(25, 34)
    reader = MapGridReader(mem.read)
    assert reader.dimensions() == (10, 20)


def test_dimensions_none_when_map_ptr_out_of_ram() -> None:
    mem = _build(25, 34)
    mem.write_u32(BACKUP_MAP_LAYOUT_ADDR + _BML_MAP_PTR, 0x00000000)
    assert MapGridReader(mem.read).dimensions() is None


def test_dimensions_none_when_size_aberrant() -> None:
    mem = _build(0, 34)  # width 0 -> nonsense
    assert MapGridReader(mem.read).dimensions() is None


def test_classify_collision_is_wall() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, 0x0400)  # collision bit set
    assert MapGridReader(mem.read).classify_at(0, 0) is TileKind.WALL


def test_classify_grass() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, 0x0005)  # metatile 5, no collision
    _set_behavior(mem, 0x0005, MB_TALL_GRASS)
    assert MapGridReader(mem.read).classify_at(0, 0) is TileKind.GRASS


def test_classify_ledge_east_is_right() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, 0x0006)
    _set_behavior(mem, 0x0006, MB_JUMP_EAST)
    assert MapGridReader(mem.read).classify_at(0, 0) is TileKind.LEDGE_RIGHT


def test_classify_plain_is_free() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, 0x0007)
    _set_behavior(mem, 0x0007, 0x00)  # MB_NORMAL
    assert MapGridReader(mem.read).classify_at(0, 0) is TileKind.FREE


def test_classify_secondary_tileset_boundary() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, 0x0201)  # secondary metatile
    _set_behavior(mem, 0x0201, MB_TALL_GRASS)
    assert MapGridReader(mem.read).classify_at(0, 0) is TileKind.GRASS


def test_classify_corruption_marker_is_wall() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, _CORRUPTION_ID)  # 0x3FF, no collision
    assert MapGridReader(mem.read).classify_at(0, 0) is TileKind.WALL


def test_classify_out_of_bounds_is_none() -> None:
    mem = _build(25, 34)
    assert MapGridReader(mem.read).classify_at(-1, 0) is None
    assert MapGridReader(mem.read).classify_at(10, 0) is None  # w==10 -> x in 0..9
