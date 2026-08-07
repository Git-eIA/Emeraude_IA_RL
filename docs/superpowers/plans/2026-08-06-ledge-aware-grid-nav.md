# Ledge-Aware Grid Navigation (Brique 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blind bump-learned navigation with grid-based navigation that consumes Brique 1's RAM map ground truth, so route_101's one-way ledge is routed *around* (not treated as a wall) and the `map_map` frontier thrash is eliminated.

**Architecture:** Five new units (`GridSnapshot` value object, `plan_path_grid` pure A* with a faithful 2-tile jump model, `MapMemory.remember_grid/grid_for`, live `navigate_grid`, discovery `explore_grid`) plus a migration cascade that drops the `WallMap` parameter from `travel_to` / `execute_order` / `run_campaign`, rewires `survey_world` onto `explore_grid`, exposes a public `WorldReader.grid_reader`, and deletes the orphaned bump-nav (`local_navigator` / `live_navigator` / `map_explorer`) once unreferenced.

**Tech Stack:** Python 3.12 (`from __future__ import annotations`, `int | None`, frozen dataclasses, `pathlib`), pytest (no network, crafted-bytes fakes + one gated ROM smoke), ruff (line-length 100). Emerald FR (BPEF) ROM at `roms/pokemon_emerald_fr.gba`.

---

## File Structure

**New files**
- `env/grid_snapshot.py` — `GridSnapshot` immutable classified-grid value object (own module to avoid a `grid_navigator` ↔ `map_memory` import cycle).
- `env/grid_navigator.py` — direction constants (`DIRECTIONS`/`DELTAS`), `plan_path_grid` (pure A* with jump edges), the live-nav primitives that outlive bump-nav (`snapshot_settled`, `handle_battle_interruption`, `probe_step`, timing/key constants, `resolve_move`), and `navigate_grid` (live ledge-aware nav with a per-run transient-block set).
- `env/grid_explorer.py` — `explore_grid` (portal-probing discovery loop).
- `tests/test_grid_snapshot.py`, `tests/test_grid_navigator.py`, `tests/test_grid_explorer.py`, `tests/test_ledge_aware_nav_rom.py` — new tests.

**Modified files**
- `env/map_memory.py` — add `remember_grid` / `grid_for`.
- `env/world_reader.py` — add public `grid_reader` property.
- `env/map_traveler.py` — `travel_to` calls `navigate_grid`, drops `wallmap`, imports `DELTAS` from `grid_navigator`.
- `env/orders.py` — `execute_order` + `_execute_heal` / `_execute_grind` / `_execute_level_up` / `_execute_battle_trainer` drop `wallmap`.
- `env/campaign.py` — `run_campaign` drops `wallmap`.
- `env/world_surveyor.py` — `survey_world` calls `explore_grid` (not `map_map`), drops `wallmap`, imports `snapshot_settled` from `grid_navigator`.
- Their unit + ROM tests (`test_map_traveler*.py`, `test_orders.py`, `test_campaign*.py`, `test_world_surveyor*.py`, `test_battle_proof_survey_rom.py`).

**Deleted files (once unreferenced)**
- `env/local_navigator.py`, `env/live_navigator.py`, `env/map_explorer.py`
- `tests/test_local_navigator.py`, `tests/test_live_navigator.py`, `tests/test_live_navigator_rom.py`, `tests/test_map_explorer.py`, `tests/test_map_explorer_rom.py`

**Test command convention (used throughout):**
- Pure suite: `.venv/bin/python -m pytest <path> -v`
- Full pure suite: `.venv/bin/python -m pytest -q`
- ROM suite: `export POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba && .venv/bin/python -m pytest -q`
- Lint: `.venv/bin/ruff check <paths>`

---

## Task 1: `GridSnapshot` value object

**Files:**
- Create: `env/grid_snapshot.py`
- Test: `tests/test_grid_snapshot.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid_snapshot.py
from __future__ import annotations

from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind


def _grid_3x2() -> GridSnapshot:
    # tiles[y][x]; width 3, height 2
    tiles = (
        (TileKind.FREE, TileKind.WALL, TileKind.GRASS),
        (TileKind.LEDGE_DOWN, TileKind.FREE, TileKind.FREE),
    )
    return GridSnapshot(map_id=(0, 16), width=3, height=2, tiles=tiles)


def test_classify_at_returns_the_tile_at_xy():
    g = _grid_3x2()
    assert g.classify_at(0, 0) is TileKind.FREE
    assert g.classify_at(1, 0) is TileKind.WALL
    assert g.classify_at(2, 0) is TileKind.GRASS
    assert g.classify_at(0, 1) is TileKind.LEDGE_DOWN


def test_classify_at_out_of_bounds_is_wall():
    g = _grid_3x2()
    assert g.classify_at(-1, 0) is TileKind.WALL
    assert g.classify_at(3, 0) is TileKind.WALL
    assert g.classify_at(0, -1) is TileKind.WALL
    assert g.classify_at(0, 2) is TileKind.WALL


def test_from_reader_captures_the_reader_grid():
    class FakeGridReader:
        def grid(self):
            return [
                [TileKind.FREE, TileKind.WALL],
                [TileKind.GRASS, TileKind.FREE],
            ]

    snap = GridSnapshot.from_reader(FakeGridReader(), map_id=(0, 16))
    assert snap is not None
    assert snap.map_id == (0, 16)
    assert snap.width == 2
    assert snap.height == 2
    assert snap.classify_at(1, 0) is TileKind.WALL
    assert isinstance(snap.tiles, tuple)
    assert isinstance(snap.tiles[0], tuple)


def test_from_reader_returns_none_when_map_not_ready():
    class NotReadyReader:
        def grid(self):
            return None

    assert GridSnapshot.from_reader(NotReadyReader(), map_id=(0, 16)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.grid_snapshot'`

- [ ] **Step 3: Write minimal implementation**

```python
# env/grid_snapshot.py
"""GridSnapshot: an immutable classified-grid value object.

Captured from a MapGridReader once per navigation step and consumed by the pure
planner (plan_path_grid) and stored by MapMemory. Standalone module on purpose:
both the navigator and map_memory depend on it, so keeping it here avoids a
grid_navigator <-> map_memory import cycle. Emerald (BPEF) only.
"""
from __future__ import annotations

from dataclasses import dataclass

from env.map_grid_reader import TileKind


@dataclass(frozen=True)
class GridSnapshot:
    map_id: tuple[int, int]
    width: int
    height: int
    tiles: tuple[tuple[TileKind, ...], ...]  # [y][x], WALL-pinned, never None

    def classify_at(self, x: int, y: int) -> TileKind:
        """Bounds-checked tile lookup; out-of-range returns WALL."""
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.tiles[y][x]
        return TileKind.WALL

    @classmethod
    def from_reader(cls, grid_reader, map_id: tuple[int, int]) -> "GridSnapshot | None":
        """Capture grid_reader.grid(); None if the map is not ready.

        map_id is a label the caller supplies; the grid reader always decodes the
        currently-loaded map and does not verify the two agree — the caller reads
        pos and map_id from the same WorldSnapshot.
        """
        rows = grid_reader.grid()
        if rows is None:
            return None
        tiles = tuple(tuple(row) for row in rows)
        height = len(tiles)
        width = len(tiles[0]) if height else 0
        return cls(map_id=map_id, width=width, height=height, tiles=tiles)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_grid_snapshot.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add env/grid_snapshot.py tests/test_grid_snapshot.py
git commit -m "$(cat <<'EOF'
feat: GridSnapshot immutable classified-grid value object

Own module to avoid a grid_navigator <-> map_memory import cycle. classify_at
is WALL-pinned out of bounds; from_reader captures MapGridReader.grid() or None.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `plan_path_grid` (pure A* with the ledge jump model)

**Files:**
- Create: `env/grid_navigator.py`
- Test: `tests/test_grid_navigator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid_navigator.py
from __future__ import annotations

from env.grid_navigator import plan_path_grid
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind

F = TileKind.FREE
W = TileKind.WALL
G = TileKind.GRASS
LU = TileKind.LEDGE_UP
LD = TileKind.LEDGE_DOWN


def _snap(rows: list[list[TileKind]]) -> GridSnapshot:
    tiles = tuple(tuple(r) for r in rows)
    return GridSnapshot(
        map_id=(0, 16), width=len(rows[0]), height=len(rows), tiles=tiles
    )


def test_straight_line_path():
    # 1x3 corridor; walk right from (0,0) to (2,0)
    snap = _snap([[F, F, F]])
    assert plan_path_grid(snap, (0, 0), (2, 0)) == ["right", "right"]


def test_routes_around_a_wall():
    # (1,0) is a wall; go down, right, up
    snap = _snap([
        [F, W, F],
        [F, F, F],
    ])
    path = plan_path_grid(snap, (0, 0), (2, 0))
    assert path is not None
    # ends on the goal
    assert _walk((0, 0), path) == (2, 0)
    # never steps onto the wall
    assert (1, 0) not in _cells((0, 0), path)


def test_one_way_ledge_descend_is_allowed():
    # standing at (0,0); (0,1) is LEDGE_DOWN; (0,2) is the FREE landing.
    # descending the ledge is a single directed jump edge (0,0)->(0,2).
    snap = _snap([
        [F],
        [LD],
        [F],
    ])
    assert plan_path_grid(snap, (0, 0), (0, 2)) == ["down"]


def test_one_way_ledge_climb_is_blocked():
    # same LEDGE_DOWN column, but now going UP from (0,2) to (0,0):
    # the ledge only accepts "down", so there is no path up.
    snap = _snap([
        [F],
        [LD],
        [F],
    ])
    assert plan_path_grid(snap, (0, 2), (0, 0)) is None


def test_routes_around_a_ledge_to_the_right_then_up():
    # A LEDGE_DOWN wall spans the middle row across x=0..1 (cannot climb it).
    # The right column x=2 is open FREE, so the way up is: right along the
    # bottom, up the open right column, then left along the top.
    snap = _snap([
        [F, F, F],
        [LD, LD, F],
        [F, F, F],
    ])
    path = plan_path_grid(snap, (0, 2), (0, 0))
    assert path is not None
    assert _walk((0, 2), path) == (0, 0)
    # the climb must use the open right column, never a ledge upward
    assert "up" in path


def test_unreachable_goal_returns_none():
    # goal (2,0) is walled off entirely
    snap = _snap([
        [F, W, F],
        [F, W, F],
    ])
    assert plan_path_grid(snap, (0, 0), (2, 0)) is None


def test_blocked_edge_forces_a_detour():
    # open 3x2 grid; block the direct (0,0)->right edge so A* detours down.
    snap = _snap([
        [F, F, F],
        [F, F, F],
    ])
    blocked = {((0, 0), "right")}
    path = plan_path_grid(snap, (0, 0), (1, 0), blocked=blocked)
    assert path is not None
    assert _walk((0, 0), path) == (1, 0)
    assert path[0] != "right"


# --- helpers: replay a direction list over the 2-tile jump model ---
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_LEDGE = {
    "up": TileKind.LEDGE_UP,
    "down": TileKind.LEDGE_DOWN,
    "left": TileKind.LEDGE_LEFT,
    "right": TileKind.LEDGE_RIGHT,
}


def _step(snap: GridSnapshot, cell, d):
    dx, dy = _DELTA[d]
    adj = (cell[0] + dx, cell[1] + dy)
    if snap.classify_at(*adj) is _LEDGE[d]:
        return (cell[0] + 2 * dx, cell[1] + 2 * dy)
    return adj


def _walk(start, path):
    # NOTE: needs the snapshot; tests bind it via closure below.
    raise NotImplementedError
```

> **Implementer note:** the two helpers `_walk` / `_cells` need the snapshot in scope. Replace the placeholder `_walk` above and add `_cells` as closures inside each test, or refactor to pass `snap` explicitly. Use this exact form:

```python
def _walk_on(snap, start, path):
    cell = start
    for d in path:
        cell = _step(snap, cell, d)
    return cell


def _cells_on(snap, start, path):
    cell = start
    out = [cell]
    for d in path:
        cell = _step(snap, cell, d)
        out.append(cell)
    return out
```

Then in the tests call `_walk_on(snap, ...)` / `_cells_on(snap, ...)` instead of `_walk` / `_cells`. Delete the placeholder `_walk` stub.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid_navigator.py -v`
Expected: FAIL with `ImportError: cannot import name 'plan_path_grid'`

- [ ] **Step 3: Write minimal implementation**

```python
# env/grid_navigator.py
"""grid_navigator: ledge-aware planning + live navigation over a RAM grid.

plan_path_grid is a pure A* over a GridSnapshot with a faithful 2-tile jump
model: a one-way ledge is traversable only in its arrow direction, so ledges are
strictly one-way by construction (no false walls, no bump-learning). navigate_grid
drives the emulator using that plan, keeping a per-run transient-block set so a
live NPC on a static-grid FREE tile degrades to a detour, not a hang.

This module also owns the live-nav primitives that outlive the deleted bump-nav
(snapshot_settled, handle_battle_interruption, probe_step, resolve_move, timing/
key constants); grid_explorer reuses them. Emerald (BPEF) only.
"""
from __future__ import annotations

import heapq
from typing import Any

from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# Grid convention: x grows right, y grows down. up decreases y.
DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# The ledge tile that a given direction may descend through.
_LEDGE_FOR: dict[str, TileKind] = {
    "up": TileKind.LEDGE_UP,
    "down": TileKind.LEDGE_DOWN,
    "left": TileKind.LEDGE_LEFT,
    "right": TileKind.LEDGE_RIGHT,
}

_STANDABLE: frozenset[TileKind] = frozenset({TileKind.FREE, TileKind.GRASS})


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_path_grid(
    grid: GridSnapshot,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[tuple[int, int], str]] | None = None,
) -> list[str] | None:
    """A* over a GridSnapshot; list of directions start->goal, or None.

    Nodes are only standable tiles (FREE/GRASS); LEDGE_*/WALL are never nodes.
    From node C in direction d (delta D): adjacent FREE/GRASS -> normal edge cost
    1; adjacent LEDGE_d with a FREE/GRASS landing at C+2D -> directed jump edge
    C->C+2D cost 1; otherwise blocked. `blocked` is an optional set of directed
    edges (cell, direction) to skip (the live navigator's transient NPC-avoidance
    set). Bounded by the finite grid (node set is width*height).
    """
    if start == goal:
        return []
    skip = blocked if blocked is not None else set()

    open_heap: list[tuple[int, tuple[int, int]]] = [(_manhattan(start, goal), start)]
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        for direction in DIRECTIONS:
            if (current, direction) in skip:
                continue
            neighbour = _edge_target(grid, current, direction)
            if neighbour is None:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbour, 1 << 30):
                g_score[neighbour] = tentative
                came_from[neighbour] = (current, direction)
                f_score = tentative + _manhattan(neighbour, goal)
                heapq.heappush(open_heap, (f_score, neighbour))
    return None


def _edge_target(
    grid: GridSnapshot, cell: tuple[int, int], direction: str
) -> tuple[int, int] | None:
    """The standable cell reached from `cell` going `direction`, or None.

    Normal step onto FREE/GRASS, or a one-tile ledge jump landing on FREE/GRASS.
    """
    dx, dy = DELTAS[direction]
    adj = (cell[0] + dx, cell[1] + dy)
    adj_kind = grid.classify_at(*adj)
    if adj_kind in _STANDABLE:
        return adj
    if adj_kind is _LEDGE_FOR[direction]:
        landing = (cell[0] + 2 * dx, cell[1] + 2 * dy)
        if grid.classify_at(*landing) in _STANDABLE:
            return landing
    return None


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]],
    goal: tuple[int, int],
) -> list[str]:
    directions: list[str] = []
    cell = goal
    while cell in came_from:
        prev, direction = came_from[cell]
        directions.append(direction)
        cell = prev
    directions.reverse()
    return directions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_grid_navigator.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add env/grid_navigator.py tests/test_grid_navigator.py
git commit -m "$(cat <<'EOF'
feat: plan_path_grid A* with faithful one-way-ledge jump model

Nodes are standable tiles only; a LEDGE_d is a directed 2-tile jump edge in its
arrow direction (descend OK, climb blocked). Optional blocked set skips directed
edges for the live NPC detour. Bounded by the finite grid, no MAX_EXPANSIONS.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `MapMemory.remember_grid` / `grid_for`

**Files:**
- Modify: `env/map_memory.py`
- Test: `tests/test_map_memory.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_map_memory.py`:

```python
def test_remember_grid_then_grid_for_returns_it():
    from env.grid_snapshot import GridSnapshot
    from env.map_grid_reader import TileKind

    mem = MapMemory()
    snap = GridSnapshot(
        map_id=(0, 16), width=1, height=1, tiles=((TileKind.FREE,),)
    )
    assert mem.grid_for((0, 16)) is None
    mem.remember_grid(snap)
    assert mem.grid_for((0, 16)) is snap


def test_remember_grid_is_last_write_wins_per_map():
    from env.grid_snapshot import GridSnapshot
    from env.map_grid_reader import TileKind

    mem = MapMemory()
    first = GridSnapshot((0, 16), 1, 1, ((TileKind.FREE,),))
    second = GridSnapshot((0, 16), 1, 1, ((TileKind.WALL,),))
    mem.remember_grid(first)
    mem.remember_grid(second)
    assert mem.grid_for((0, 16)) is second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_map_memory.py -k remember_grid -v`
Expected: FAIL with `AttributeError: 'MapMemory' object has no attribute 'remember_grid'`

- [ ] **Step 3: Write minimal implementation**

In `env/map_memory.py`, add a `TYPE_CHECKING` import near the top (after the existing `from env.world_reader import WorldSnapshot`):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.grid_snapshot import GridSnapshot
```

In `MapMemory.__init__`, add the store (next to `self._labeled_cells`):

```python
        # map_id -> remembered classified grid (last-write-wins).
        self._grids: dict[tuple[int, int], "GridSnapshot"] = {}
```

Add two methods to `MapMemory` (e.g. after `healing_spots`):

```python
    def remember_grid(self, snap: "GridSnapshot") -> None:
        """Store the snapshot keyed by its map_id; last-write-wins."""
        self._grids[snap.map_id] = snap

    def grid_for(self, map_id: tuple[int, int]) -> "GridSnapshot | None":
        """Return the remembered grid for map_id, or None if never seen."""
        return self._grids.get(map_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_map_memory.py -q`
Expected: PASS (all map_memory tests, incl. the 2 new)

- [ ] **Step 5: Commit**

```bash
git add env/map_memory.py tests/test_map_memory.py
git commit -m "$(cat <<'EOF'
feat: MapMemory.remember_grid / grid_for (per-map remembered grid)

Additive, last-write-wins. GridSnapshot imported under TYPE_CHECKING only
(remember_grid reads snap.map_id at runtime, no runtime import) so map_memory
stays free of any import cycle.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `WorldReader.grid_reader` public property

**Files:**
- Modify: `env/world_reader.py`
- Test: `tests/test_world_reader.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_world_reader.py`:

```python
def test_grid_reader_exposes_the_map_grid_reader():
    from env.map_grid_reader import MapGridReader

    def read(_addr, _size):
        return b"\x00" * _size

    reader = WorldReader(read)
    assert isinstance(reader.grid_reader, MapGridReader)
    # same instance each access (no re-construction)
    assert reader.grid_reader is reader.grid_reader
```

> If `WorldReader` is not already imported at the top of `tests/test_world_reader.py`, add `from env.world_reader import WorldReader`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_world_reader.py -k grid_reader -v`
Expected: FAIL with `AttributeError: 'WorldReader' object has no attribute 'grid_reader'`

- [ ] **Step 3: Write minimal implementation**

In `env/world_reader.py`, add a property to `WorldReader` (after `__init__`):

```python
    @property
    def grid_reader(self) -> MapGridReader:
        """The MapGridReader decoding the currently-loaded map (Brique 1)."""
        return self._grid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_world_reader.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/world_reader.py tests/test_world_reader.py
git commit -m "$(cat <<'EOF'
feat: WorldReader.grid_reader public property

Exposes the private MapGridReader so navigate_grid / explore_grid read the grid
off the same reader with no re-construction or duplicate read-fn wiring.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: live-nav primitives + `navigate_grid`

**Files:**
- Modify: `env/grid_navigator.py` (add primitives + `navigate_grid`)
- Test: `tests/test_grid_navigator.py` (append live-fake tests)

**Context:** `navigate_grid` replaces `navigate_to`. Because the deleted bump-nav owned `snapshot_settled` / `handle_battle_interruption` / `probe_step` / `resolve_move` / the d-pad key map / timing constants, this task moves faithful copies into `grid_navigator` (grid_explorer will reuse them in Task 6). The loop: snapshot → `GridSnapshot.from_reader` → `remember_grid` → `plan_path_grid(..., blocked=blocked)` → press first move → classify. A press that fails to move records the directed edge in the per-run `blocked` set so the next plan detours.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grid_navigator.py`:

```python
from env.map_grid_reader import TileKind as _TK
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot


class _FakeGridReader:
    """Serves a fixed classified grid regardless of the loaded map."""

    def __init__(self, rows):
        self._rows = rows

    def grid(self):
        return [list(r) for r in self._rows]


class _LedgeWorld:
    """Emulator + reader double. The player walks a small grid; a LEDGE_DOWN at
    (0,1) may be descended (down from (0,0) lands (0,2)) but never climbed.

    Each 'down'/'up'/... press updates pos per the 2-tile jump model; a press
    into a WALL leaves pos unchanged (a 'blocked' outcome).
    """

    def __init__(self, rows, start):
        self._rows = rows
        self._pos = start
        self._grid = _FakeGridReader(rows)
        self._blocked_npc: set[tuple[tuple[int, int], str]] = set()

    # --- reader surface ---
    def snapshot(self):
        return WorldSnapshot(map_id=(0, 16), pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return False

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    # --- emulator surface ---
    def step(self, key, _frames):
        from emulator import buttons

        keymap = {
            buttons.KEY_UP: "up",
            buttons.KEY_DOWN: "down",
            buttons.KEY_LEFT: "left",
            buttons.KEY_RIGHT: "right",
        }
        d = keymap.get(key)
        if d is None:
            return
        self._pos = self._resolve(self._pos, d)

    def _classify(self, x, y):
        if 0 <= y < len(self._rows) and 0 <= x < len(self._rows[0]):
            return self._rows[y][x]
        return _TK.WALL

    def _resolve(self, cell, d):
        if (cell, d) in self._blocked_npc:
            return cell  # a phantom NPC stands here: press does not move
        delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        ledge = {
            "up": _TK.LEDGE_UP,
            "down": _TK.LEDGE_DOWN,
            "left": _TK.LEDGE_LEFT,
            "right": _TK.LEDGE_RIGHT,
        }[d]
        adj = (cell[0] + delta[0], cell[1] + delta[1])
        kind = self._classify(*adj)
        if kind in (_TK.FREE, _TK.GRASS):
            return adj
        if kind is ledge:
            land = (cell[0] + 2 * delta[0], cell[1] + 2 * delta[1])
            if self._classify(*land) in (_TK.FREE, _TK.GRASS):
                return land
        return cell  # wall / wrong-arrow ledge: no move


def test_navigate_grid_descends_a_ledge_in_the_correct_direction():
    from env.grid_navigator import navigate_grid

    rows = [
        [_TK.FREE],
        [_TK.LEDGE_DOWN],
        [_TK.FREE],
    ]
    world = _LedgeWorld(rows, start=(0, 0))
    assert navigate_grid(world, world, target=(0, 2)) == "arrived"
    assert world._pos == (0, 2)


def test_navigate_grid_refuses_to_climb_a_one_way_ledge():
    from env.grid_navigator import navigate_grid

    rows = [
        [_TK.FREE],
        [_TK.LEDGE_DOWN],
        [_TK.FREE],
    ]
    world = _LedgeWorld(rows, start=(0, 2))
    # climbing back up is impossible: no path -> unreachable, no hang.
    assert navigate_grid(world, world, target=(0, 0)) == "unreachable"


def test_navigate_grid_detours_around_a_phantom_npc():
    from env.grid_navigator import navigate_grid

    # open 3x2 grid; an NPC blocks the direct (0,0)->right press. navigate_grid
    # must add that edge to its transient set and reroute down/right/up.
    rows = [
        [_TK.FREE, _TK.FREE, _TK.FREE],
        [_TK.FREE, _TK.FREE, _TK.FREE],
    ]
    world = _LedgeWorld(rows, start=(0, 0))
    world._blocked_npc.add(((0, 0), "right"))
    assert navigate_grid(world, world, target=(1, 0)) == "arrived"
    assert world._pos == (1, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid_navigator.py -k navigate_grid -v`
Expected: FAIL with `ImportError: cannot import name 'navigate_grid'`

- [ ] **Step 3: Write minimal implementation**

Append to `env/grid_navigator.py`:

```python
from emulator import buttons
from env.battle_player import play_battle
from env.encounter_detector import EncounterWatcher
from env.heal_detector import HealWatcher
from env.map_memory import MapMemory, WorldEvent

_DIRECTION_KEYS: dict[str, int] = {
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
}

STEP_FRAMES = 24      # hold a d-pad key ~0.4 s: one walking tile
RELEASE_FRAMES = 8    # idle after each press so the GBA doesn't fuse presses
TURN_RETRIES = 2      # a first press may only turn; retry to tell turn from wall
SETTLE_TRIES = 4      # re-read snapshot to skip SaveBlock None frames


def snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap


def resolve_move(before: Any, after: Any) -> str:
    """Classify one attempted step: 'moved' | 'blocked' | 'transition'."""
    if before.map_id != after.map_id:
        return "transition"
    if before.pos != after.pos:
        return "moved"
    return "blocked"


def handle_battle_interruption(
    emulator: Any, reader: Any, move_type_fn: Any, predict: Any
) -> str | None:
    """If a wild battle is in progress, hand it to the Fighter and report.

    None when there is no battle (or it was won) so the caller resumes; a
    terminal outcome otherwise: "battle_interrupted" (no Fighter), "battle_lost",
    "battle_timeout".
    """
    if not reader.in_battle():
        return None
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result == "won":
        return None
    return "battle_lost" if result == "lost" else "battle_timeout"


def probe_step(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    """Press `direction`, retrying so a first-press turn isn't read as a wall."""
    outcome = "blocked"
    for _ in range(TURN_RETRIES):
        emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
        emulator.step(0, RELEASE_FRAMES)
        after = snapshot_settled(reader)
        if after is None:
            return "blocked"
        outcome = resolve_move(before, after)
        if outcome != "blocked":
            return outcome
    return outcome


def navigate_grid(
    emulator: Any,
    reader: Any,
    target: tuple[int, int],
    max_steps: int = 200,
    memory: MapMemory | None = None,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Walk the player to `target` on its current map, planning over the RAM grid.

    Zero false walls, one-way ledges respected by construction. A per-run
    transient-block set catches a planned press that fails to move (a live NPC on
    a static-grid FREE tile) and reroutes; it is per-call only, never persisted.
    When `memory` is given, the live grid is remembered, a border crossing is
    recorded as a portal, a heal tags a healing spot, and a battle tags has_grass.

    Returns 'arrived' | 'unreachable' | 'left_map' | 'timeout' | 'battle_lost' |
    'battle_timeout' | 'battle_interrupted'.
    """
    heal_watcher = HealWatcher()
    enc_watcher = EncounterWatcher()
    blocked: set[tuple[tuple[int, int], str]] = set()

    for _ in range(max_steps):
        before = snapshot_settled(reader)
        if before is None:
            emulator.step(0, RELEASE_FRAMES)   # relocating; idle and retry
            continue
        if memory is not None:
            if heal_watcher.observe(reader.party_hp()):
                memory.observe(before, WorldEvent(healed=True))
            if enc_watcher.observe(reader.in_battle()):
                memory.observe(before, WorldEvent(encounter_started=True))
        interruption = handle_battle_interruption(
            emulator, reader, move_type_fn, predict
        )
        if interruption is not None:
            return interruption
        if before.pos == target:
            return "arrived"

        snap = GridSnapshot.from_reader(reader.grid_reader, before.map_id)
        if snap is None:
            emulator.step(0, RELEASE_FRAMES)   # map not ready; idle and retry
            continue
        if memory is not None:
            memory.remember_grid(snap)

        path = plan_path_grid(snap, before.pos, target, blocked=blocked)
        if path is None:
            return "unreachable"
        direction = path[0]
        outcome = probe_step(emulator, reader, before, direction)
        if outcome == "transition":
            if memory is not None:
                landed = snapshot_settled(reader)
                if landed is not None:
                    memory.record_portal(
                        before.map_id, before.pos, direction, landed.map_id,
                        False, landed.pos,
                    )
            return "left_map"
        if outcome == "blocked":
            blocked.add((before.pos, direction))   # transient: NPC / surprise
    return "timeout"
```

> **Import-cycle note:** `grid_navigator` importing `MapMemory` is fine — `map_memory` imports `GridSnapshot` only under `TYPE_CHECKING`, so there is no runtime cycle.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_grid_navigator.py -q`
Expected: PASS (all: pure planner + 3 live-fake)

- [ ] **Step 5: Commit**

```bash
git add env/grid_navigator.py tests/test_grid_navigator.py
git commit -m "$(cat <<'EOF'
feat: navigate_grid live ledge-aware nav + per-run transient-block set

Replaces navigate_to: plans over the live RAM GridSnapshot (zero false walls,
one-way ledges by construction), remembers the grid, records portals, plays wild
battles. A planned press that fails to move is added to a per-call blocked set so
a live NPC degrades to a detour, not a timeout. Moves the surviving live-nav
primitives (snapshot_settled/handle_battle_interruption/probe_step/resolve_move)
into grid_navigator; grid_explorer will reuse them.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `explore_grid` discovery loop

**Files:**
- Create: `env/grid_explorer.py`
- Test: `tests/test_grid_explorer.py`

**Context:** `explore_grid` replaces `map_map`. With the RAM grid, geometry is known the instant the map loads — nothing left to discover about terrain. The grid holds no warp destinations, so the only thing to probe is **portals**: which border tiles warp and to where. `explore_grid` reads + remembers the grid once, then routes with `navigate_grid`/`plan_path_grid` to each reachable border FREE/GRASS cell and steps outward off the map edge. A `transition` records a portal (with the step-back reversibility check reused from `map_map`); a `blocked`/no-op means that edge is not a portal. `complete` = every reachable border-edge candidate has been tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid_explorer.py
from __future__ import annotations

from env.grid_explorer import explore_grid
from env.map_grid_reader import TileKind as TK
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot

F, W = TK.FREE, TK.WALL


class _FakeGridReader:
    def __init__(self, rows):
        self._rows = rows

    def grid(self):
        return [list(r) for r in self._rows]


class _ExploreWorld:
    """A tiny map with exactly one warp: stepping off the top edge from the
    single top-row FREE cell transitions to another map; every other border step
    is blocked. Records how many outward border steps were attempted so the test
    can assert there is no thrash (each candidate tested at most once).
    """

    def __init__(self, rows, start, warp_cell, warp_dir):
        self._rows = rows
        self._pos = start
        self._map = (0, 16)
        self._grid = _FakeGridReader(rows)
        self._warp_cell = warp_cell
        self._warp_dir = warp_dir
        self._returned = True
        self.border_attempts = 0

    def snapshot(self):
        return WorldSnapshot(map_id=self._map, pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return False

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    def step(self, key, _frames):
        from emulator import buttons

        keymap = {
            buttons.KEY_UP: "up",
            buttons.KEY_DOWN: "down",
            buttons.KEY_LEFT: "left",
            buttons.KEY_RIGHT: "right",
        }
        d = keymap.get(key)
        if d is None:
            return
        delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        target = (self._pos[0] + delta[0], self._pos[1] + delta[1])
        on_map = 0 <= target[1] < len(self._rows) and 0 <= target[0] < len(self._rows[0])
        if on_map:
            if self._rows[target[1]][target[0]] is F:
                self._pos = target
            return
        # stepping off the edge:
        self.border_attempts += 1
        if self._map == (0, 16) and self._pos == self._warp_cell and d == self._warp_dir:
            self._map = (0, 17)              # warped to the neighbour map
            self._pos = (0, 0)
        # else: blocked, stay put (not a warp)

    # step-back after a transition returns us to the origin map:
    def _step_back_supported(self):
        return True


def test_explore_grid_records_a_border_portal_without_thrash():
    # 3-wide, 2-tall open map; the only warp is UP from (1,0).
    rows = [
        [F, F, F],
        [F, F, F],
    ]
    world = _ExploreWorld(rows, start=(1, 1), warp_cell=(1, 0), warp_dir="up")
    memory = MapMemory()
    result = explore_grid(world, world, memory, target_map=(0, 16))
    # a portal from (0,16) up to (0,17) was recorded
    assert memory.portal((0, 16), (0, 17)) is not None
    # the remembered grid is stored
    assert memory.grid_for((0, 16)) is not None
    # bounded, no infinite re-probing: far fewer than max_steps border attempts
    assert world.border_attempts < 50
    assert result in ("complete", "left_map")
```

> **Implementer note:** the ROM-faithful step-back reversibility check (press the opposite direction, confirm we returned to `target_map`) mirrors `map_map`'s transition branch. For a fake that warps one-way you may land on `(0,17)` and not return — model whichever your `explore_grid` needs; the assertion only requires the portal be recorded. Keep the fake's `step` honest about which edge warps.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid_explorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.grid_explorer'`

- [ ] **Step 3: Write minimal implementation**

```python
# env/grid_explorer.py
"""grid_explorer: probe a map's border tiles for portals over the RAM grid.

explore_grid replaces map_map. With the RAM grid the terrain (walls, grass,
ledges) is known the instant the map loads, so nothing about geometry needs
probing. The grid holds no warp destinations, so the only thing left to discover
is portals: explore_grid reads + remembers the grid once, routes to each reachable
border FREE/GRASS cell, and steps outward off the map edge. A transition records
a portal (with a step-back reversibility check); a blocked step means that edge is
not a portal. complete = every reachable border candidate tested. The RAM grid
kills the map_map thrash: geometry is never re-probed and blocked edges are never
re-proposed. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from env.grid_navigator import (
    DELTAS,
    RELEASE_FRAMES,
    handle_battle_interruption,
    navigate_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory

_STANDABLE = (TileKind.FREE, TileKind.GRASS)


def explore_grid(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    target_map: tuple[int, int],
    max_steps: int = 2000,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Probe `target_map`'s reachable border tiles for portals.

    Returns:
      "complete"            — every reachable border candidate tested
      "budget_exhausted"    — hit max_steps before candidates emptied
      "left_map"            — crossed a non-reversible border and could not return
      "battle_interrupted" / "battle_lost" / "battle_timeout"
    """
    here = snapshot_settled(reader)
    if here is None or here.map_id != target_map:
        return "left_map"
    snap = GridSnapshot.from_reader(reader.grid_reader, target_map)
    if snap is None:
        return "budget_exhausted"
    memory.remember_grid(snap)

    candidates = _border_candidates(snap)
    tested: set[tuple[tuple[int, int], str]] = set()
    steps = 0
    for cell, direction in candidates:
        if steps >= max_steps:
            return "budget_exhausted"
        steps += 1
        if (cell, direction) in tested:
            continue
        tested.add((cell, direction))

        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle

        arrived = navigate_grid(
            emulator, reader, cell, memory=memory,
            move_type_fn=move_type_fn, predict=predict,
        )
        if arrived in ("battle_lost", "battle_timeout", "battle_interrupted"):
            return arrived
        if arrived == "left_map":
            return "left_map"
        if arrived != "arrived":
            continue   # unreachable/timeout candidate: skip, no re-probe

        before = snapshot_settled(reader)
        if before is None or before.pos != cell:
            continue
        outcome = probe_step(emulator, reader, before, direction)
        if outcome != "transition":
            continue   # not a portal (blocked / no-op)
        landed = snapshot_settled(reader)
        if landed is None:
            return "left_map"
        probe_step(emulator, reader, landed, _opposite(direction))
        returned = snapshot_settled(reader)
        reversible = returned is not None and returned.map_id == target_map
        memory.record_portal(
            target_map, cell, direction, landed.map_id, reversible, landed.pos
        )
        if not reversible:
            return "left_map"
    return "complete"


_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _opposite(direction: str) -> str:
    return _OPPOSITE[direction]


def _border_candidates(
    snap: GridSnapshot,
) -> list[tuple[tuple[int, int], str]]:
    """Every standable border cell paired with the outward direction off its edge."""
    out: list[tuple[tuple[int, int], str]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            for direction, (dx, dy) in DELTAS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < snap.width and 0 <= ny < snap.height):
                    out.append(((x, y), direction))
    return out
```

> **Note on `RELEASE_FRAMES` import:** it is imported for symmetry with the bump-nav idle pattern; if unused after implementation, remove it to keep ruff clean (F401).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_grid_explorer.py -v`
Expected: PASS

Then lint the new module:
Run: `.venv/bin/ruff check env/grid_explorer.py env/grid_navigator.py env/grid_snapshot.py`
Expected: no errors (remove any unused import flagged).

- [ ] **Step 5: Commit**

```bash
git add env/grid_explorer.py tests/test_grid_explorer.py
git commit -m "$(cat <<'EOF'
feat: explore_grid portal-probing discovery loop (replaces map_map)

RAM grid makes geometry free, so explore_grid only probes border tiles for
portals: read+remember the grid once, route to each reachable border cell, step
outward, record a portal on transition (step-back reversibility check). No
frontier BFS over unproven edges -> no budget_exhausted thrash.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: migrate `travel_to` / `execute_order` / `run_campaign` off `WallMap`

**Files:**
- Modify: `env/map_traveler.py`, `env/orders.py`, `env/campaign.py`
- Modify: `tests/test_map_traveler.py`, `tests/test_orders.py`, `tests/test_campaign.py`
- Modify (ROM): `tests/test_map_traveler_rom.py`, `tests/test_campaign_rom.py`

**Context (why one task):** `wallmap` threads callee→caller through `travel_to` (callee) into `execute_order`+5 helpers and `run_campaign` (callers). Dropping the parameter breaks callers until all change together, so this is one cohesive, green-at-the-end task. `world_surveyor` (the other `travel_to` caller) is migrated in Task 8. No backwards-compat shim — the parameter is removed outright.

- [ ] **Step 1: Rewrite `travel_to` onto `navigate_grid`**

In `env/map_traveler.py`, replace the imports and every `navigate_to(...)` call. New top-of-file imports:

```python
from env.grid_navigator import DELTAS, navigate_grid, snapshot_settled
from env.map_memory import MapMemory
from env.route_planner import plan_route
```

Change `travel_to`'s signature to drop `wallmap` and rewrite its body. Full new function:

```python
def travel_to(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    goal_map: tuple[int, int],
    goal_cell: tuple[int, int],
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Walk map-by-map to goal_cell on goal_map over known territory.

    Returns 'arrived' | 'unknown_route' | 'unreachable' | 'lost' | 'timeout'
    | 'battle_lost' | 'battle_timeout' | 'battle_interrupted'.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return navigate_grid(
                emulator, reader, goal_cell,
                move_type_fn=move_type_fn, predict=predict,
            )

        route = plan_route(memory, here.map_id, goal_map)
        if route is None or len(route) < 2:
            return "unknown_route"
        next_map = route[1]
        crossing = memory.portal(here.map_id, next_map)
        if crossing is None:
            return "unknown_route"

        reached = navigate_grid(
            emulator, reader, crossing.from_cell,
            move_type_fn=move_type_fn, predict=predict,
        )
        if reached in BATTLE_OUTCOMES:
            return reached
        if reached in ("unreachable", "timeout"):
            return reached
        if reached == "left_map":
            continue

        dx, dy = DELTAS[crossing.direction]
        neighbour = (crossing.from_cell[0] + dx, crossing.from_cell[1] + dy)
        crossed = navigate_grid(
            emulator, reader, neighbour, memory=memory,
            move_type_fn=move_type_fn, predict=predict,
        )
        if crossed in BATTLE_OUTCOMES:
            return crossed
        if crossed in ("unreachable", "timeout"):
            return crossed

        landed = _snapshot_settled(reader)
        if landed is None or landed.map_id != next_map:
            return "lost"
    return "timeout"
```

Keep the module's own `_snapshot_settled` helper as-is (it is independent of the deleted modules). Delete the now-unused `from env.live_navigator import navigate_to` and `from env.local_navigator import DELTAS, WallMap` lines (replaced above).

- [ ] **Step 2: Drop `wallmap` from `orders.py`**

In `env/orders.py`, remove the `wallmap` parameter from `execute_order`, `_execute_heal`, `_execute_grind`, `_execute_level_up`, `_execute_battle_trainer`, and remove it from every internal call. Concretely:

- `execute_order(order, emulator, reader, memory, wallmap, max_hops=20, ...)` → `execute_order(order, emulator, reader, memory, max_hops=20, ...)`.
- Each dispatch call loses `wallmap`, e.g.
  `return _execute_heal(emulator, reader, memory, wallmap, max_hops=max_hops)` →
  `return _execute_heal(emulator, reader, memory, max_hops=max_hops)`.
- Same for `_execute_grind`, `_execute_level_up`, `_execute_battle_trainer` dispatch calls.
- Each helper signature drops `wallmap` (the `wallmap: Any,` line).
- Each `travel_to(emulator, reader, memory, wallmap, goal_map, goal_cell, ...)` call → `travel_to(emulator, reader, memory, goal_map, goal_cell, ...)`.
- In `_execute_level_up`, the recursive `_execute_grind(...)` / `_execute_heal(...)` calls drop `wallmap`.

- [ ] **Step 3: Drop `wallmap` from `campaign.py`**

In `env/campaign.py`, remove `wallmap` from `run_campaign`'s signature and from every `order_fn(...)` call:

- `def run_campaign(emulator, reader, memory, wallmap, curriculum=CAMPAIGN, ...)` → `def run_campaign(emulator, reader, memory, curriculum=CAMPAIGN, ...)`.
- Each of the three `order_fn(Order(...), emulator, reader, memory, wallmap, ...)` calls → `order_fn(Order(...), emulator, reader, memory, ...)`.

- [ ] **Step 4: Update the unit tests**

In `tests/test_map_traveler.py`, `tests/test_orders.py`, `tests/test_campaign.py`:
- Remove every `from env.local_navigator import WallMap` (and any `WallMap` construction like `wallmap = WallMap()`).
- Remove the `wallmap` argument from every `travel_to(...)`, `execute_order(...)`, `run_campaign(...)` call.
- The fakes in these tests double as both emulator and reader. If a fake reader lacks a `grid_reader` property (navigate_grid now needs it whenever it actually plans), add a minimal one. **Most traveler/orders tests never reach `navigate_grid`'s planning** (they arrive immediately or fail at route planning), but any test that walks a real path must expose `grid_reader`. For a fake that should "arrive immediately" (pos already == target), no grid is consulted. For a fake that walks, give it a `grid_reader` returning an all-FREE grid large enough to contain the path, e.g.:

```python
class _AllFreeGridReader:
    def __init__(self, w, h):
        self._w, self._h = w, h

    def grid(self):
        from env.map_grid_reader import TileKind
        return [[TileKind.FREE] * self._w for _ in range(self._h)]
```

and add `grid_reader` as a property on the fake returning `_AllFreeGridReader(...)`.

> **Implementer:** run the target test file after edits and fix each `TypeError` (unexpected `wallmap`) / `AttributeError` (`grid_reader`) it surfaces. These are mechanical.

- [ ] **Step 5: Update the ROM tests**

In `tests/test_map_traveler_rom.py` and `tests/test_campaign_rom.py`:
- Remove `WallMap` import + construction.
- Drop the `wallmap` argument from the `travel_to(...)` / `run_campaign(...)` calls.
- The real `WorldReader` already exposes `grid_reader` (Task 4), so no fake grid is needed.

- [ ] **Step 6: Run the migrated tests**

Run: `.venv/bin/python -m pytest tests/test_map_traveler.py tests/test_orders.py tests/test_campaign.py -q`
Expected: PASS (all migrated). Fix any residual `wallmap`/`grid_reader` errors until green.

Lint: `.venv/bin/ruff check env/map_traveler.py env/orders.py env/campaign.py`
Expected: no errors (no unused `WallMap`/`navigate_to` imports remain).

- [ ] **Step 7: Commit**

```bash
git add env/map_traveler.py env/orders.py env/campaign.py \
  tests/test_map_traveler.py tests/test_orders.py tests/test_campaign.py \
  tests/test_map_traveler_rom.py tests/test_campaign_rom.py
git commit -m "$(cat <<'EOF'
refactor: drop WallMap from travel_to / execute_order / run_campaign

travel_to now drives navigate_grid (RAM-grid ledge-aware nav) instead of the
bump-learned navigate_to; the WallMap parameter is removed outright from
travel_to, execute_order (+ all 5 mode helpers) and run_campaign, and from every
caller and test. No shim.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: rewire `survey_world` onto `explore_grid`

**Files:**
- Modify: `env/world_surveyor.py`
- Modify: `tests/test_world_surveyor.py`, `tests/test_world_surveyor_rom.py`, `tests/test_battle_proof_survey_rom.py`

**Context:** `survey_world` is the last `travel_to`/`map_map` caller. It drops `wallmap`, imports `snapshot_settled` from `grid_navigator` (was `live_navigator`), and calls `explore_grid` where it called `map_map`.

- [ ] **Step 1: Rewire `world_surveyor.py`**

Replace the imports:

```python
from env.grid_explorer import explore_grid
from env.grid_navigator import snapshot_settled
from env.map_memory import MapMemory, Portal
from env.map_traveler import BATTLE_OUTCOMES, travel_to
```

(Delete `from env.live_navigator import snapshot_settled`, `from env.local_navigator import WallMap`, `from env.map_explorer import map_map`.)

Change `survey_world`'s signature to drop `wallmap`:

```python
def survey_world(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    max_maps: int = 50,
    move_type_fn: Any = None,
    predict: Any = None,
) -> SurveyReport:
```

In the body, drop `wallmap` from the `travel_to(...)` call, and replace the `map_map(...)` call:

```python
        result = explore_grid(
            emulator, reader, memory, target,
            move_type_fn=move_type_fn, predict=predict,
        )
```

`explore_grid` returns the same outcome vocabulary map_map did (`complete` | `budget_exhausted` | `left_map` | battle outcomes), so the surrounding `if result in BATTLE_OUTCOMES` / `if result in ("left_map", "budget_exhausted")` branches are unchanged (keep the `f"map:{result}"` reason labels).

- [ ] **Step 2: Update the surveyor tests**

In `tests/test_world_surveyor.py`:
- Remove `WallMap` import + construction, drop `wallmap` from `survey_world(...)` calls.
- Fakes that survey a map now flow through `explore_grid`, which calls `GridSnapshot.from_reader(reader.grid_reader, ...)` and `navigate_grid`. Give each surveying fake a `grid_reader` property returning a small all-FREE (or scenario-specific) grid, mirroring the `_AllFreeGridReader` pattern from Task 7. For a two-map survey the fake already teleports between maps via `travel_to`; ensure its `grid_reader` returns a grid sized to whatever border cell `explore_grid` routes to. Keep the portals the fake exposes so BFS still enqueues the second map.

> **Implementer:** the surveyor unit tests assert *which maps get surveyed* and *which legs fail*, not internal step counts. Adjust the fakes so `explore_grid` returns `complete`/`left_map` per scenario; run the file and fix each surfaced error.

In `tests/test_world_surveyor_rom.py` and `tests/test_battle_proof_survey_rom.py`:
- Remove `WallMap` import + construction, drop `wallmap` from `survey_world(...)`.
- Real `WorldReader` supplies `grid_reader`.
- `test_battle_proof_survey_rom.py` loads a mid-battle savestate and asserts the sweep resolves the battle rather than false-walling. Under `explore_grid`, the first `handle_battle_interruption` still fires before any border probe, so the load-bearing assertion (sweep does not hang / battle outcome surfaced) holds; keep the assertion, drop only `wallmap`.

- [ ] **Step 3: Run the migrated tests**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -q`
Expected: PASS. Fix residual `wallmap`/`grid_reader` errors until green.

Lint: `.venv/bin/ruff check env/world_surveyor.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add env/world_surveyor.py tests/test_world_surveyor.py \
  tests/test_world_surveyor_rom.py tests/test_battle_proof_survey_rom.py
git commit -m "$(cat <<'EOF'
refactor: survey_world drives explore_grid, drops WallMap

Last travel_to/map_map caller migrated: survey_world now surveys each map via
explore_grid (portal probing over the RAM grid) and imports snapshot_settled from
grid_navigator. Same SurveyReport vocabulary; wallmap parameter removed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: delete the orphaned bump-nav

**Files:**
- Delete: `env/local_navigator.py`, `env/live_navigator.py`, `env/map_explorer.py`
- Delete: `tests/test_local_navigator.py`, `tests/test_live_navigator.py`, `tests/test_live_navigator_rom.py`, `tests/test_map_explorer.py`, `tests/test_map_explorer_rom.py`

- [ ] **Step 1: Confirm nothing still imports the bump-nav**

Run: `.venv/bin/python -m pytest -q --collect-only 2>&1 | tail -20`
Then verify no source imports remain:
Run (Grep tool, not bash): search `env/` and `tests/` for `local_navigator|live_navigator|map_explorer`.
Expected: matches only inside the five files about to be deleted (and their own test files). If any *other* file still imports them, fix that import first (it should already be migrated by Tasks 7–8).

- [ ] **Step 2: Delete the modules and their tests**

```bash
git rm env/local_navigator.py env/live_navigator.py env/map_explorer.py \
  tests/test_local_navigator.py tests/test_live_navigator.py \
  tests/test_live_navigator_rom.py tests/test_map_explorer.py \
  tests/test_map_explorer_rom.py
```

- [ ] **Step 3: Run the full pure suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (green, fewer tests than before — the deleted bump-nav tests are gone, the new grid tests are in).

Lint the whole `env/`:
Run: `.venv/bin/ruff check env/ tests/`
Expected: no errors (no dangling imports).

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor: delete orphaned bump-nav (local/live navigator, map_explorer)

Nothing imports the bump-learned nav after the grid migration: WallMap,
navigate_to, plan_path, resolve_move, map_map and their tests are removed. The
RAM grid replaces bump-learning entirely.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: gated ROM smoke on route_101

**Files:**
- Create: `tests/test_ledge_aware_nav_rom.py`

**Context:** ONE gated ROM smoke. Load `states/post_starter.state` (already exists), read the grid, and prove `explore_grid` reveals it and `navigate_grid` crosses the mid-route_101 ledge northward — where the old `map_map` stalled at `budget_exhausted`. Triple-guard: skip unless the ROM env var, the savestate, and (for the battle path) a Fighter checkpoint are present.

- [ ] **Step 1: Write the gated smoke**

```python
# tests/test_ledge_aware_nav_rom.py
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
```

> **Implementer note:** if `states/post_starter.state` starts the player somewhere the northward target is trivially blocked, adjust `target` to a known-reachable northern FREE cell using the ASCII dump (`tools/dump_map_grid.py`) as ground truth. The load-bearing assertion is the `if result == "timeout"` guard: a timeout with no y-progress is the exact regression this brick kills.

- [ ] **Step 2: Run the smoke**

Run: `export POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba && .venv/bin/python -m pytest tests/test_ledge_aware_nav_rom.py -v`
Expected: PASS (both tests; if the second's target needs adjustment, refine per the note until it passes on real RAM).

- [ ] **Step 3: Commit**

```bash
git add tests/test_ledge_aware_nav_rom.py
git commit -m "$(cat <<'EOF'
test: gated ROM smoke — grid reveals route_101, nav crosses the ledge north

Loads states/post_starter.state: GridSnapshot.from_reader decodes the live map,
navigate_grid routes north around the one-way ledge. Load-bearing guard: a
timeout must still show y-progress (never the old map_map budget_exhausted
thrash). Triple-skips without ROM / savestate.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full pure suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 0 failures. Note the passed/skipped counts.

- [ ] **Step 2: Run the full ROM suite**

Run: `export POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba && .venv/bin/python -m pytest -q`
Expected: PASS, the gated ROM smokes execute (not skipped), 0 failures.

- [ ] **Step 3: Lint everything**

Run: `.venv/bin/ruff check env/ tests/ tools/`
Expected: no errors.

- [ ] **Step 4: Final grep for dead references**

Use the Grep tool over `env/` and `tests/`:
- `WallMap` → no matches.
- `navigate_to` / `map_map` / `local_navigator` / `live_navigator` / `map_explorer` → no matches.
Expected: all clear. If any remain, fix and re-run Steps 1–3.

No commit (verification only). This completes Brique 2.

---

## Self-Review

**Spec coverage:**
- Unit 1 `plan_path_grid` → Task 2 (incl. ledge descend-OK/climb-blocked, route-around-right, unreachable, blocked-detour). ✓
- Unit 2 `GridSnapshot` → Task 1 (classify_at OOB=WALL, from_reader None). ✓
- Unit 3 `MapMemory.remember_grid/grid_for` → Task 3. ✓
- Unit 4 `navigate_grid` + transient-block → Task 5 (ledge cross, refuse climb, NPC detour). ✓
- Unit 5 `explore_grid` → Task 6 (no-thrash, records border portal). ✓
- Migration cascade (WallMap drop from travel_to/execute_order/run_campaign; survey_world→explore_grid) → Tasks 7–8. ✓
- Public `WorldReader.grid_reader` → Task 4. ✓
- Bump-nav deletion → Task 9. ✓
- Gated ROM smoke route_101 → Task 10. ✓
- Coordinate invariant (snapshot.pos indexes grid) → relied on in Task 5 `plan_path_grid(snap, before.pos, target)` and asserted live in Task 10 `classify_at(*snap.pos)`. ✓

**Placeholder scan:** the only non-literal content is the `_walk`/`_cells` helper note in Task 2 (explicit replacement code given) and the ROM-target refinement note in Task 10 (grounded in `tools/dump_map_grid.py`). No TBD/TODO/"add error handling"/"similar to Task N".

**Type consistency:** `plan_path_grid(grid, start, goal, blocked=None) -> list[str] | None`, `GridSnapshot.classify_at(x,y) -> TileKind`, `GridSnapshot.from_reader(grid_reader, map_id) -> GridSnapshot | None`, `remember_grid(snap)`/`grid_for(map_id)`, `navigate_grid(emulator, reader, target, max_steps, memory, move_type_fn, predict) -> str`, `explore_grid(emulator, reader, memory, target_map, max_steps, move_type_fn, predict) -> str`, `WorldReader.grid_reader -> MapGridReader` — used identically across tasks. `DELTAS`/`DIRECTIONS` live in `grid_navigator` and are imported by `map_traveler` (DELTAS) and `grid_explorer` (DELTAS); `snapshot_settled`/`handle_battle_interruption`/`probe_step` live in `grid_navigator` and are imported by `grid_explorer` and `world_surveyor`. Consistent. ✓
