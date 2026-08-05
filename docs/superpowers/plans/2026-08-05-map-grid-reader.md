# Map-Grid Reader (Brique 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A RAM-only reader that decodes the loaded map grid into per-tile `FREE`/`WALL`/`GRASS`/`LEDGE_<dir>` classifications plus the raw metatile behavior, wired into `WorldReader._tile_behavior()`.

**Architecture:** New `env/map_grid_reader.py` follows the project's injected-`read(addr,size)` reader pattern (zero ROM/SB3 dependency, testable with crafted bytes). The pokeemerald map-memory decode logic is fully specified below and pure-tested against synthetic buffers. The only genuine unknowns are the BPEF RAM addresses + border padding; these are isolated into probe-owned module constants (seeded from pokeemerald BPEE) and confirmed by a throwaway discovery/validation tool before any dependent code trusts them.

**Tech Stack:** Python 3.12, pytest, ruff (line-length 100), mgba emulator (`emulator.gba.GbaEmulator`) for the gated ROM smoke only.

**Spec:** `docs/superpowers/specs/2026-08-05-map-grid-reader-design.md`

---

## Critical Invariants (read before every task)

1. **Two distinct logics.** `classify_at` = passability: read **collision FIRST**; if `collision != 0` return `WALL` and NEVER read behavior. `tile_behavior_at` = RAW behavior: **ignore collision entirely** (a wall-tree tile still returns its behavior byte, not `None`). This split is the whole point — `WorldReader._tile_behavior()` wants the raw behavior.
2. **Probe-owned constants are NOT trusted until Task 1 validates them.** `BACKUP_MAP_LAYOUT_ADDR`, `MAP_HEADER_ADDR`, and the padding are seeded from BPEE. If Task 1's probe cannot PASS on `states/post_starter.state`, STOP and escalate to the user — do not silently guess.
3. **Circular-validation guard.** Pure tests validate decode *arithmetic* against the reader's own constants (true by construction). Only the ROM **bump-test cross-check** (Task 7, live WallMap bump vs grid `WALL`) is load-bearing evidence the constant *values* are right. Do not add a pure test that "proves" the addresses.
4. **Elevation bits (12-15) are ignored** in Brique 1.
5. **Conservative on corruption / unresolved reads → `WALL`** (never crash, never return a bogus FREE).

## File Structure

- Create: `env/map_grid_reader.py` — the reader (decode logic + injected read).
- Create: `tests/test_map_grid_reader.py` — pure unit tests vs synthetic memory.
- Create: `tools/probe_map_grid.py` — throwaway BPEF discovery/validation probe.
- Create: `tools/dump_map_grid.py` — throwaway ASCII grid dumper (annex utility).
- Create: `tests/test_map_grid_reader_rom.py` — gated, load-bearing ROM bump-test.
- Modify: `env/world_reader.py` — wire `_tile_behavior()` to `MapGridReader`.
- Modify: `tests/test_world_reader.py` — replace the None-stub test.

**Execution-order caveat:** Task 1's probe imports `MapGridReader`, so **write Task 2 before running Task 1**. The plan lists the probe first because it owns the discovery, but the module skeleton must exist for it to run.

---

### Task 1: Discovery/validation probe

**Files:**
- Create: `tools/probe_map_grid.py`

Exploratory tool, no unit test. It confirms the probe-owned constants hold on BPEF before any dependent code trusts them.

- [ ] **Step 1: Write the probe**

```python
"""Throwaway BPEF map-grid discovery/validation probe.

Loads a mid-overworld savestate, prints the gBackupMapLayout struct + the
gMapHeader tileset chain, then exercises MapGridReader at the player's tile and
sweeps the grid for LEDGE_*/GRASS tiles. Exits 0 only if the reader decodes a
sane map (dimensions resolve AND the player's own tile classifies FREE). A
non-zero exit means the seeded BPEE addresses/padding are wrong for BPEF ->
escalate to the user before trusting the reader.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.map_grid_reader import MapGridReader, TileKind


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state_path = sys.argv[1] if len(sys.argv) > 1 else "states/post_starter.state"
    emu = GbaEmulator(rom)
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)  # let one frame settle so the map layout is coherent

    grid = MapGridReader(emu.read_bytes)
    player = EmeraldReader(emu.read_bytes).player_state()
    dims = grid.dimensions()
    print(f"dimensions() = {dims}")
    if player is not None:
        px, py = player.x, player.y
        print(f"player at ({px},{py})")
        print(f"  classify_at = {grid.classify_at(px, py)}")
        print(f"  tile_behavior_at = {grid.tile_behavior_at(px, py)}")
    else:
        px = py = None
        print("player_state() is None")

    if dims is not None:
        w, h = dims
        ledges = []
        grasses = []
        for y in range(h):
            for x in range(w):
                kind = grid.classify_at(x, y)
                if kind in (TileKind.LEDGE_UP, TileKind.LEDGE_DOWN,
                            TileKind.LEDGE_LEFT, TileKind.LEDGE_RIGHT):
                    ledges.append((x, y, kind.name))
                elif kind is TileKind.GRASS:
                    grasses.append((x, y))
        print(f"ledges found: {ledges[:20]}{' ...' if len(ledges) > 20 else ''}")
        print(f"grass tiles: {len(grasses)} (e.g. {grasses[:10]})")

    ok = (
        dims is not None
        and player is not None
        and grid.classify_at(px, py) is TileKind.FREE
    )
    print("VALIDATION:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after Task 2 exists)**

Run: `POKEMON_EMERALD_ROM=<rom> python tools/probe_map_grid.py`
Expected: prints sane dimensions (~10x20 for route_101), player tile FREE, and `VALIDATION: PASS`. If `FAIL` or garbage dimensions → STOP, report to the user which constant looks wrong (addresses vs padding), do not proceed.

- [ ] **Step 3: Commit**

```bash
git add tools/probe_map_grid.py
git commit -m "$(cat <<'EOF'
tools: throwaway BPEF map-grid discovery/validation probe

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Module skeleton + dimensions()

**Files:**
- Create: `env/map_grid_reader.py`
- Test: `tests/test_map_grid_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
"""MapGridReader: decode the loaded Emerald map grid from RAM (no ROM)."""
from __future__ import annotations

import pytest

from env.map_grid_reader import (
    BACKUP_MAP_LAYOUT_ADDR,
    MAP_HEADER_ADDR,
    _BML_HEIGHT,
    _BML_MAP_PTR,
    _BML_WIDTH,
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: FAIL — `env.map_grid_reader` does not exist.

- [ ] **Step 3: Write the module (skeleton + dimensions)**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add env/map_grid_reader.py tests/test_map_grid_reader.py
git commit -m "$(cat <<'EOF'
feat: MapGridReader skeleton + dimensions() (border-padding strip)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: classify_at() — passability

**Files:**
- Modify: `env/map_grid_reader.py`
- Test: `tests/test_map_grid_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
from env.map_grid_reader import (  # add to existing imports
    MB_JUMP_EAST,
    MB_TALL_GRASS,
    TileKind,
)


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
```

Add `_CORRUPTION_ID` to the imports at the top of the test file.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: FAIL — `classify_at` not defined.

- [ ] **Step 3: Implement**

Add to `MapGridReader`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add env/map_grid_reader.py tests/test_map_grid_reader.py
git commit -m "$(cat <<'EOF'
feat: MapGridReader.classify_at (collision-first passability + ledges/grass)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: tile_behavior_at() — raw behavior (collision ignored)

**Files:**
- Modify: `env/map_grid_reader.py`
- Test: `tests/test_map_grid_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_tile_behavior_ignores_collision() -> None:
    # A wall-tree tile: collision bit SET, but behavior is tall grass.
    # classify_at -> WALL (collision-first), tile_behavior_at -> the raw byte.
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET, 0x0400 | 0x000A)
    _set_behavior(mem, 0x000A, MB_TALL_GRASS)
    reader = MapGridReader(mem.read)
    assert reader.classify_at(0, 0) is TileKind.WALL
    assert reader.tile_behavior_at(0, 0) == MB_TALL_GRASS


def test_tile_behavior_out_of_bounds_is_none() -> None:
    mem = _build(25, 34)
    assert MapGridReader(mem.read).tile_behavior_at(-1, 0) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: FAIL — `tile_behavior_at` not defined.

- [ ] **Step 3: Implement**

Add to `MapGridReader`:

```python
    def tile_behavior_at(self, x: int, y: int) -> int | None:
        """Raw metatile behavior at (x,y), collision IGNORED. None if off-map.

        This is what WorldReader._tile_behavior wants: the behavior byte of the
        tile the player stands on, regardless of passability.
        """
        entry = self._raw_entry(x, y)
        if entry is None:
            return None
        return self._behavior(entry & _METATILE_ID_MASK)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add env/map_grid_reader.py tests/test_map_grid_reader.py
git commit -m "$(cat <<'EOF'
feat: MapGridReader.tile_behavior_at (raw behavior, collision ignored)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: grid() — full classified rectangle

**Files:**
- Modify: `env/map_grid_reader.py`
- Test: `tests/test_map_grid_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_grid_returns_full_classified_rectangle() -> None:
    mem = _build(25, 34)
    _set_tile(mem, 25, MAP_OFFSET + 1, MAP_OFFSET, 0x0400)  # (1,0) wall
    _set_tile(mem, 25, MAP_OFFSET, MAP_OFFSET + 1, 0x0005)  # (0,1) grass
    _set_behavior(mem, 0x0005, MB_TALL_GRASS)
    g = MapGridReader(mem.read).grid()
    assert g is not None
    assert len(g) == 20 and len(g[0]) == 10
    assert g[0][1] is TileKind.WALL
    assert g[1][0] is TileKind.GRASS
    assert g[0][0] is TileKind.FREE


def test_grid_none_when_no_map() -> None:
    mem = _build(25, 34)
    mem.write_u32(BACKUP_MAP_LAYOUT_ADDR + _BML_MAP_PTR, 0x00000000)
    assert MapGridReader(mem.read).grid() is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: FAIL — `grid` not defined.

- [ ] **Step 3: Implement**

Add to `MapGridReader`:

```python
    def grid(self) -> list[list[TileKind]] | None:
        """Full [y][x] classified rectangle, or None if unreadable.

        Any cell that would classify as None (shouldn't happen inside bounds) is
        pinned to WALL so the returned grid never contains None. NOT for hot
        loops -- reads the whole map every call; callers cache it.
        """
        dims = self.dimensions()
        if dims is None:
            return None
        w, h = dims
        return [
            [self.classify_at(x, y) or TileKind.WALL for x in range(w)]
            for y in range(h)
        ]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_map_grid_reader.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add env/map_grid_reader.py tests/test_map_grid_reader.py
git commit -m "$(cat <<'EOF'
feat: MapGridReader.grid full classified rectangle (WALL-pinned, no None cells)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Wire WorldReader._tile_behavior

**Files:**
- Modify: `env/world_reader.py`
- Test: `tests/test_world_reader.py`

- [ ] **Step 1: Replace the None-stub test**

In `tests/test_world_reader.py`, DELETE `test_tile_behavior_is_none_until_probed` and add:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_world_reader.py -v`
Expected: FAIL — `tile_behavior` is still None (stub).

- [ ] **Step 3: Wire the reader**

In `env/world_reader.py`, add the import and build the grid reader in the ctor:

```python
from env.map_grid_reader import MapGridReader
```

```python
    def __init__(self, read: ReadFn) -> None:
        self._reader = EmeraldReader(read)
        self._battle = BattleReader(read)
        self._grid = MapGridReader(read)
```

Replace the `_tile_behavior` stub with:

```python
    def _tile_behavior(self) -> int | None:
        """Raw metatile behavior of the tile the player stands on, or None."""
        ps = self._reader.player_state()
        if ps is None:
            return None
        return self._grid.tile_behavior_at(ps.x, ps.y)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_world_reader.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add env/world_reader.py tests/test_world_reader.py
git commit -m "$(cat <<'EOF'
feat: wire WorldReader._tile_behavior to MapGridReader

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: ROM bump-test cross-check + dump tool

**Files:**
- Create: `tools/dump_map_grid.py`
- Create: `tests/test_map_grid_reader_rom.py`

This is the ONLY load-bearing evidence the constant VALUES are right.

- [ ] **Step 1: Write the ASCII dump tool**

```python
"""Throwaway ASCII dump of the loaded map grid (visual sanity check)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.map_grid_reader import MapGridReader, TileKind

_GLYPH = {
    TileKind.FREE: ".",
    TileKind.WALL: "#",
    TileKind.GRASS: '"',
    TileKind.LEDGE_UP: "^",
    TileKind.LEDGE_DOWN: "v",
    TileKind.LEDGE_LEFT: "<",
    TileKind.LEDGE_RIGHT: ">",
}


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state_path = sys.argv[1] if len(sys.argv) > 1 else "states/post_starter.state"
    emu = GbaEmulator(rom)
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    grid = MapGridReader(emu.read_bytes).grid()
    if grid is None:
        print("no map")
        sys.exit(1)
    ps = EmeraldReader(emu.read_bytes).player_state()
    for y, row in enumerate(grid):
        line = []
        for x, kind in enumerate(row):
            if ps is not None and (x, y) == (ps.x, ps.y):
                line.append("@")
            else:
                line.append(_GLYPH[kind])
        print("".join(line))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the gated bump-test**

```python
"""Load-bearing ROM cross-check: the grid's WALL agrees with a live wall bump.

The pure tests validate decode arithmetic against the reader's own constants
(true by construction). This is the ONLY test that proves the probe-owned BPEF
addresses/padding are the RIGHT VALUES: it bumps the player into a wall live and
asserts MapGridReader independently classifies that tile as WALL.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from emulator.buttons import KEY_UP
from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.map_grid_reader import MapGridReader, TileKind
from tests.conftest import requires_rom

_STATE = Path("states/post_starter.state")


@requires_rom
@pytest.mark.skipif(not _STATE.exists(), reason="post_starter.state missing")
def test_grid_reader_agrees_with_a_live_wall_bump(rom_path: Path) -> None:
    emu = GbaEmulator(str(rom_path))
    emu.load_state(_STATE.read_bytes())
    emu.step(0, 4)
    er = EmeraldReader(emu.read_bytes)
    grid = MapGridReader(emu.read_bytes)

    before = er.player_state()
    assert before is not None
    # Walk into a wall: press UP repeatedly; if the player never advances north,
    # the tile above is impassable -> the grid must classify it WALL.
    for _ in range(6):
        emu.step(KEY_UP, 24)
        emu.step(0, 8)
    after = er.player_state()
    assert after is not None

    if (after.x, after.y) == (before.x, before.y):
        above = grid.classify_at(before.x, before.y - 1)
        assert above is TileKind.WALL
    else:
        pytest.skip("player advanced; no wall bump to cross-check here")
```

Note: if this always skips, the start tile isn't wall-adjacent north — pick a bump direction/target where the WALL branch runs (e.g. try KEY_LEFT toward the known west wall), so the assertion is exercised. The test is not done until the WALL branch runs green at least once locally.

- [ ] **Step 3: Run with ROM**

Run: `POKEMON_EMERALD_ROM=<rom> python -m pytest tests/test_map_grid_reader_rom.py -v`
Also run the probe (Task 1) and the dump tool now to eyeball the grid.
Expected: PASS (WALL branch exercised), and the dump shows a plausible route_101 with grass/ledges.

- [ ] **Step 4: Commit**

```bash
git add tools/dump_map_grid.py tests/test_map_grid_reader_rom.py
git commit -m "$(cat <<'EOF'
test: load-bearing ROM bump-test + ASCII map-grid dump tool

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Pure suite + lint**

Run: `python -m pytest -q` (no ROM) and `ruff check env/map_grid_reader.py env/world_reader.py tests/test_map_grid_reader.py tests/test_world_reader.py tools/probe_map_grid.py tools/dump_map_grid.py tests/test_map_grid_reader_rom.py`
Expected: all green, ruff clean.

- [ ] **Step 2: With ROM**

Symlink `roms`/`states`/`checkpoints` from the main repo, then:
Run: `POKEMON_EMERALD_ROM=<rom> python -m pytest -q`
Expected: prior count + new tests passing, the ROM smoke load-bearing (not skipped).

---

## Self-Review

- **Spec coverage:** RAM-only reader (Tasks 2-5), two distinct logics classify vs behavior (Tasks 3-4), FREE/WALL/GRASS/LEDGE_<dir> (Task 3), wired into `_tile_behavior` (Task 6), discovery probe for BPEF unknowns (Task 1), circular-validation honored — only ROM bump-test load-bearing (Task 7), conservative-on-corruption (Tasks 3/5), annex dump tool (Task 7). Elevation ignored (invariant 4). All spec sections map to a task.
- **Placeholder scan:** the only unverified values are the probe-owned constants, intentionally isolated and gated by Task 1 with a user-escalation branch; the ROM test's possible-skip is called out with a "not done until the WALL branch runs" instruction. No TODO/TBD left in shipped code.
- **Type consistency:** `classify_at`/`grid`/`tile_behavior_at` names stable across Tasks 3-7; `TileKind` members `LEDGE_UP/DOWN/LEFT/RIGHT` consistent; struct-offset constants (`_BML_*`, `_ML_*`, `_TS_METATILE_ATTR_PTR`, `_MH_MAP_LAYOUT_PTR`) match between module and tests; `MapGridReader(read)` ctor signature matches every call site (probe, dump, WorldReader, tests).
