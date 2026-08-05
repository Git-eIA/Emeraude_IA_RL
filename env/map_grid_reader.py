"""MapGridReader: decode the currently-loaded Emerald map grid from RAM.

Follows the project's injected-read pattern: takes a read(addr, size) -> bytes
callable, no ROM/SB3 dependency, testable with crafted bytes. Decodes the
pokeemerald map-memory model:

  gBackupMapLayout = { s32 width; s32 height; u16 *map }  (padded grid)
  per tile u16: metatile_id = entry & 0x03FF
                collision   = (entry & 0x0C00) >> 10   (!=0 -> wall)
                elevation   = bits 12-15                (ignored)
  behavior: metatile_id -> tileset metatileAttributes[u16], low byte.
            id < 0x200 -> primary tileset, else secondary (id - 0x200).

The BPEF addresses + border padding below are probe-owned: seeded from BPEE,
confirmed on BPEF by tools/probe_map_grid.py. See the design spec's
circular-validation warning: only the ROM bump-test proves these VALUES.
"""
from __future__ import annotations

from enum import Enum

from env.game_state import ReadFn

# --- Probe-owned BPEF constants (seeded BPEE; confirmed by probe_map_grid) ---
BACKUP_MAP_LAYOUT_ADDR = 0x03005DC0
MAP_HEADER_ADDR = 0x02037318
MAP_OFFSET = 7          # MAP_OFFSET border padding (pokeemerald)
_WIDTH_PAD = 15         # padded_width  - logical_width  (2*MAP_OFFSET + 1)
_HEIGHT_PAD = 14        # padded_height - logical_height (2*MAP_OFFSET)

# struct offsets
_BML_WIDTH = 0x00
_BML_HEIGHT = 0x04
_BML_MAP_PTR = 0x08
_MH_MAP_LAYOUT_PTR = 0x00
_ML_PRIMARY_TILESET_PTR = 0x10
_ML_SECONDARY_TILESET_PTR = 0x14
_TS_METATILE_ATTR_PTR = 0x0C

# tile-entry bitfields
_METATILE_ID_MASK = 0x03FF
_COLLISION_MASK = 0x0C00
_COLLISION_SHIFT = 10
_CORRUPTION_ID = 0x3FF
_SECONDARY_TILESET_START = 0x200
_BEHAVIOR_MASK = 0x00FF

# metatile behaviors (pokeemerald include/constants/metatile_behaviors.h)
MB_TALL_GRASS = 0x02
MB_LONG_GRASS = 0x03
MB_JUMP_EAST = 0x38
MB_JUMP_WEST = 0x39
MB_JUMP_NORTH = 0x3A
MB_JUMP_SOUTH = 0x3B

# memory regions
_EWRAM_START, _EWRAM_END = 0x02000000, 0x02040000
_IWRAM_START, _IWRAM_END = 0x03000000, 0x03008000
_ROM_START, _ROM_END = 0x08000000, 0x0E000000
_DIM_MIN, _DIM_MAX = 1, 256


class TileKind(Enum):
    FREE = 0
    WALL = 1
    GRASS = 2
    LEDGE_UP = 3
    LEDGE_DOWN = 4
    LEDGE_LEFT = 5
    LEDGE_RIGHT = 6


_LEDGE_BY_BEHAVIOR = {
    MB_JUMP_NORTH: TileKind.LEDGE_UP,
    MB_JUMP_SOUTH: TileKind.LEDGE_DOWN,
    MB_JUMP_WEST: TileKind.LEDGE_LEFT,
    MB_JUMP_EAST: TileKind.LEDGE_RIGHT,
}
_GRASS_BEHAVIORS = {MB_TALL_GRASS, MB_LONG_GRASS}


class MapGridReader:
    """Decodes the loaded map grid; conservative (WALL) on any corruption."""

    def __init__(self, read: ReadFn) -> None:
        self._read = read

    def dimensions(self) -> tuple[int, int] | None:
        """Logical (width, height) of the loaded map, or None if unreadable."""
        pw = self._s32(BACKUP_MAP_LAYOUT_ADDR + _BML_WIDTH)
        ph = self._s32(BACKUP_MAP_LAYOUT_ADDR + _BML_HEIGHT)
        map_ptr = self._u32(BACKUP_MAP_LAYOUT_ADDR + _BML_MAP_PTR)
        if not self._is_ram(map_ptr):
            return None
        w, h = pw - _WIDTH_PAD, ph - _HEIGHT_PAD
        if not (_DIM_MIN <= w <= _DIM_MAX and _DIM_MIN <= h <= _DIM_MAX):
            return None
        return (w, h)

    def _layout(self) -> tuple[int, int, int] | None:
        """(padded_width, padded_height, map_buffer_ptr) or None."""
        pw = self._s32(BACKUP_MAP_LAYOUT_ADDR + _BML_WIDTH)
        ph = self._s32(BACKUP_MAP_LAYOUT_ADDR + _BML_HEIGHT)
        map_ptr = self._u32(BACKUP_MAP_LAYOUT_ADDR + _BML_MAP_PTR)
        if not self._is_ram(map_ptr):
            return None
        if not (_DIM_MIN <= pw <= _DIM_MAX + _WIDTH_PAD):
            return None
        if not (_DIM_MIN <= ph <= _DIM_MAX + _HEIGHT_PAD):
            return None
        return (pw, ph, map_ptr)

    @staticmethod
    def _is_ram(addr: int) -> bool:
        return _EWRAM_START <= addr < _EWRAM_END or _IWRAM_START <= addr < _IWRAM_END

    @staticmethod
    def _is_ptr(addr: int) -> bool:
        return (
            _EWRAM_START <= addr < _EWRAM_END
            or _IWRAM_START <= addr < _IWRAM_END
            or _ROM_START <= addr < _ROM_END
        )

    def _u16(self, addr: int) -> int:
        b = self._read(addr, 2)
        return b[0] | (b[1] << 8)

    def _u32(self, addr: int) -> int:
        b = self._read(addr, 4)
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    def _s32(self, addr: int) -> int:
        v = self._u32(addr)
        return v - 0x100000000 if v & 0x80000000 else v

    def classify_at(self, x: int, y: int) -> TileKind | None:
        """Passability of the tile at logical (x,y). None if off-map/unreadable.

        Collision is read FIRST: any collision -> WALL and behavior is never
        consulted. Corruption marker -> WALL (conservative).
        """
        entry = self._raw_entry(x, y)
        if entry is None:
            return None
        metatile_id = entry & _METATILE_ID_MASK
        if (entry & _COLLISION_MASK) >> _COLLISION_SHIFT != 0:
            return TileKind.WALL
        if metatile_id == _CORRUPTION_ID:
            return TileKind.WALL
        behavior = self._behavior(metatile_id)
        if behavior is None:
            return TileKind.WALL
        if behavior in _LEDGE_BY_BEHAVIOR:
            return _LEDGE_BY_BEHAVIOR[behavior]
        if behavior in _GRASS_BEHAVIORS:
            return TileKind.GRASS
        return TileKind.FREE

    def _raw_entry(self, x: int, y: int) -> int | None:
        """The u16 map entry at logical (x,y), or None if off-map/unreadable."""
        dims = self.dimensions()
        layout = self._layout()
        if dims is None or layout is None:
            return None
        w, h = dims
        if not (0 <= x < w and 0 <= y < h):
            return None
        pw, _ph, map_ptr = layout
        px, py = x + MAP_OFFSET, y + MAP_OFFSET
        return self._u16(map_ptr + 2 * (py * pw + px))

    def _behavior(self, metatile_id: int) -> int | None:
        """Low-byte behavior of a metatile, or None if the attr table is bad."""
        table = self._attr_table_for(metatile_id)
        if table is None:
            return None
        if metatile_id < _SECONDARY_TILESET_START:
            index = metatile_id
        else:
            index = metatile_id - _SECONDARY_TILESET_START
        return self._u16(table + 2 * index) & _BEHAVIOR_MASK

    def _attr_table_for(self, metatile_id: int) -> int | None:
        """Resolve the metatileAttributes table ptr for a metatile id."""
        layout_ptr = self._u32(MAP_HEADER_ADDR + _MH_MAP_LAYOUT_PTR)
        if not self._is_ptr(layout_ptr):
            return None
        if metatile_id < _SECONDARY_TILESET_START:
            ts = self._u32(layout_ptr + _ML_PRIMARY_TILESET_PTR)
        else:
            ts = self._u32(layout_ptr + _ML_SECONDARY_TILESET_PTR)
        if not self._is_ptr(ts):
            return None
        table = self._u32(ts + _TS_METATILE_ATTR_PTR)
        return table if self._is_ptr(table) else None
