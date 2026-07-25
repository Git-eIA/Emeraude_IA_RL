# Inter-Map Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Explorer walk from one map to another over known territory by remembering where each map's door is (portals) and chaining intra-map navigation door-to-door.

**Architecture:** Three focused changes. (1) `MapMemory` gains portal storage — a directed border crossing `(from_map → to_map)` recording the exact `from_cell` + `direction`. (2) `navigate_to` gains an optional `memory` so it records a portal the moment it crosses a boundary. (3) A new `env/map_traveler.py` provides `travel_to`, a bounded loop that plans a map route (P2 `plan_route`), walks to each known door (`navigate_to`), crosses it, and repeats to the goal cell. Known territory only: an unknown door on the route yields `"unknown_route"` (no exploration; mapping mode is a later step).

**Tech Stack:** Python ≥3.12, pytest, libmgba-py (ROM-gated smoke only). Pure Python for units; no ROM, no training.

---

### Task 1: Portals in MapMemory

**Files:**
- Modify: `env/map_memory.py`
- Test: `tests/test_map_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_map_memory.py`:

```python
from env.map_memory import Portal


def test_record_and_read_portal() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16))
    p = mem.portal((0, 9), (0, 16))
    assert p == Portal(from_cell=(5, 0), direction="up", to_map=(0, 16))


def test_portal_is_none_for_unrecorded_pair() -> None:
    assert MapMemory().portal((0, 9), (0, 16)) is None


def test_record_portal_also_creates_the_edge() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16))
    assert ((0, 9), (0, 16)) in mem.edges()


def test_record_portal_last_write_wins() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16))
    mem.record_portal((0, 9), (4, 0), "up", (0, 16))
    assert mem.portal((0, 9), (0, 16)) == Portal((4, 0), "up", (0, 16))


def test_record_portal_creates_both_nodes() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16))
    assert mem.node((0, 9)) is not None
    assert mem.node((0, 16)) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_memory.py -q`
Expected: FAIL with `ImportError: cannot import name 'Portal'`.

- [ ] **Step 3: Add Portal and the two methods**

In `env/map_memory.py`, add the dataclass after `PlaceNode` (near the other dataclasses):

```python
@dataclass(frozen=True)
class Portal:
    """One directed border crossing: leave `from_cell` going `direction` to reach `to_map`."""
    from_cell: tuple[int, int]
    direction: str
    to_map: tuple[int, int]
```

In `MapMemory.__init__`, add the portal store next to `self._edges`:

```python
        # (from_map, to_map) -> the remembered crossing between them.
        self._portals: dict[
            tuple[tuple[int, int], tuple[int, int]], Portal
        ] = {}
```

Add these two methods to `MapMemory` (after `edges`):

```python
    def record_portal(
        self,
        from_map: tuple[int, int],
        from_cell: tuple[int, int],
        direction: str,
        to_map: tuple[int, int],
    ) -> None:
        """Remember the door from from_map to to_map; also ensures the edge exists."""
        self._ensure_node(from_map)
        self._ensure_node(to_map)
        self._edges.add((from_map, to_map))
        self._portals[(from_map, to_map)] = Portal(from_cell, direction, to_map)

    def portal(
        self, from_map: tuple[int, int], to_map: tuple[int, int]
    ) -> Portal | None:
        """The known crossing from from_map to to_map, or None if never recorded."""
        return self._portals.get((from_map, to_map))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_memory.py -q`
Expected: PASS (all map_memory tests, old + 5 new).

- [ ] **Step 5: Commit**

```bash
git add env/map_memory.py tests/test_map_memory.py
git commit -m "feat: portals in MapMemory (record_portal / portal)"
```

---

### Task 2: navigate_to records a portal on crossing

**Files:**
- Modify: `env/live_navigator.py`
- Test: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_navigator.py`:

```python
from env.map_memory import MapMemory, Portal


def test_left_map_records_portal_when_memory_given() -> None:
    # Stepping down from (0,0) onto the transition cell (0,1) crosses to a new map.
    world = FakeWorld(start=(0, 0), map_flips={(0, 1)})
    memory = MapMemory()
    result = navigate_to(
        world, world, WallMap(), target=(0, 3), max_steps=50, memory=memory
    )
    assert result == "left_map"
    assert memory.portal((0, 0), (0, 1)) == Portal(
        from_cell=(0, 0), direction="down", to_map=(0, 1)
    )
```

Note: the existing `test_left_map_when_stepping_onto_a_transition_cell` (which calls `navigate_to` with no `memory`) is the regression guard that the `memory=None` default path is unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py::test_left_map_records_portal_when_memory_given -q`
Expected: FAIL with `TypeError: navigate_to() got an unexpected keyword argument 'memory'`.

- [ ] **Step 3: Add the optional memory parameter and recording**

In `env/live_navigator.py`, add the import near the top:

```python
from env.map_memory import MapMemory
```

Change the `navigate_to` signature and the transition branch. Replace the current function head and the `if outcome == "transition":` block:

```python
def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
    memory: MapMemory | None = None,
) -> str:
    """Walk the player to `target` on its current map.

    When `memory` is given, a map transition is recorded as a portal
    (from_cell + direction + landed-on map). Returns
    'arrived' | 'unreachable' | 'left_map' | 'timeout'.
    """
    for _ in range(max_steps):
        before = reader.snapshot()
        if before is None:
            emulator.step(0, RELEASE_FRAMES)   # relocating; idle a beat and retry
            continue
        if before.pos == target:
            return "arrived"
        path = plan_path(wallmap, before.map_id, before.pos, target)
        if path is None:
            return "unreachable"
        direction = path[0]
        outcome = _press_until_moved(emulator, reader, before, direction)
        if outcome == "transition":
            if memory is not None:
                landed = _snapshot_settled(reader)
                if landed is not None:
                    memory.record_portal(
                        before.map_id, before.pos, direction, landed.map_id
                    )
            return "left_map"
        if outcome == "blocked":
            wallmap.block(before.map_id, before.pos, direction)
    return "timeout"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (all 8 live_navigator tests).

- [ ] **Step 5: Commit**

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "feat: navigate_to records a portal when memory is provided"
```

---

### Task 3: travel_to — same-map and single hop

**Files:**
- Create: `env/map_traveler.py`
- Test: `tests/test_map_traveler.py`

- [ ] **Step 1: Write the failing tests (with the multi-map fake)**

Create `tests/test_map_traveler.py`:

```python
"""map_traveler: door-to-door inter-map travel over a fake multi-map world (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DIRECTIONS, WallMap
from env.map_memory import MapMemory
from env.map_traveler import travel_to
from env.world_reader import WorldSnapshot

_KEY_TO_DIR: dict[int, str] = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}
_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class MultiMapWorld:
    """Several hidden grids joined by border crossings, for the traveler.

    Acts as emulator (`step`) and reader (`snapshot`).
    `borders` maps (map_id, cell, direction) -> (next_map, entry_cell): pressing
    `direction` on `cell` of `map_id` drops the player onto `entry_cell` of
    `next_map`. `walls` blocks a (map_id, cell, direction) move. Movement inside a
    map is otherwise free.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        borders: dict[tuple[tuple[int, int], tuple[int, int], str],
                       tuple[tuple[int, int], tuple[int, int]]] | None = None,
        walls: set[tuple[tuple[int, int], tuple[int, int], str]] | None = None,
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._borders = dict(borders or {})
        self._walls = set(walls or ())

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        border = self._borders.get((self.map_id, self.pos, direction))
        if border is not None:
            self.map_id, self.pos = border
            return
        if (self.map_id, self.pos, direction) in self._walls:
            return  # wall: no move
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot | None:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def test_same_map_delegates_to_navigate() -> None:
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, MapMemory(), WallMap(),
        goal_map=(0, 0), goal_cell=(2, 0),
    )
    assert result == "arrived"
    assert world.pos == (2, 0)


def test_single_hop_crosses_one_known_door() -> None:
    # Map A=(0,0): door at (2,0) pressing 'right' lands on map B=(0,1) cell (0,0).
    borders = {((0, 0), (2, 0), "right"): ((0, 1), (0, 0))}
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "arrived"
    assert world.map_id == (0, 1)
    assert world.pos == (1, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.map_traveler'`.

- [ ] **Step 3: Write env/map_traveler.py**

Create `env/map_traveler.py`:

```python
"""map_traveler: walk the Explorer across maps, door to door.

travel_to chains P2 route planning (plan_route) with P3 step-1 intra-map
navigation (navigate_to): walk to each known portal cell, cross it, repeat until
the goal cell on the goal map is reached. Known territory only — an unknown door
on the route returns "unknown_route" rather than exploring (mapping mode is a
later step). No training, no reward. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from env.live_navigator import navigate_to
from env.local_navigator import DELTAS, WallMap
from env.map_memory import MapMemory
from env.route_planner import plan_route

SETTLE_TRIES = 4   # skip SaveBlock None frames when reading where we landed


def travel_to(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    goal_map: tuple[int, int],
    goal_cell: tuple[int, int],
    max_hops: int = 20,
) -> str:
    """Walk map-by-map to goal_cell on goal_map over known territory.

    Returns 'arrived' | 'unknown_route' | 'unreachable' | 'lost' | 'timeout'.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return navigate_to(emulator, reader, wallmap, goal_cell)

        route = plan_route(memory, here.map_id, goal_map)
        if route is None or len(route) < 2:
            return "unknown_route"
        next_map = route[1]
        crossing = memory.portal(here.map_id, next_map)
        if crossing is None:
            return "unknown_route"   # door not yet discovered (mapping is deferred)

        reached = navigate_to(emulator, reader, wallmap, crossing.from_cell)
        if reached in ("unreachable", "timeout"):
            return reached
        if reached == "left_map":
            continue   # already crossed a border on the way; re-plan from new map

        # On the door cell: press the crossing direction by targeting the off-map
        # neighbour, which transitions on the first press (and records the portal).
        dx, dy = DELTAS[crossing.direction]
        neighbour = (crossing.from_cell[0] + dx, crossing.from_cell[1] + dy)
        navigate_to(emulator, reader, wallmap, neighbour, memory=memory)

        landed = _snapshot_settled(reader)
        if landed is None or landed.map_id != next_map:
            return "lost"
    return "timeout"


def _snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add env/map_traveler.py tests/test_map_traveler.py
git commit -m "feat: travel_to — same-map delegate + single known-door hop"
```

---

### Task 4: travel_to — chain and failure outcomes

**Files:**
- Test: `tests/test_map_traveler.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_map_traveler.py` (imports already present; add `WorldEvent`):

```python
from env.map_memory import WorldEvent


def test_three_map_chain() -> None:
    # A=(0,0) --right@(2,0)--> B=(0,1) --right@(2,0)--> C=(0,2)
    borders = {
        ((0, 0), (2, 0), "right"): ((0, 1), (0, 0)),
        ((0, 1), (2, 0), "right"): ((0, 2), (0, 0)),
    }
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    memory.record_portal((0, 1), (2, 0), "right", (0, 2))
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 2), goal_cell=(1, 0),
    )
    assert result == "arrived"
    assert world.map_id == (0, 2)
    assert world.pos == (1, 0)


def test_unknown_route_when_portal_missing() -> None:
    # Edge A->B exists (from observe) but no portal was ever recorded.
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 0), (0, 0), None), WorldEvent())
    memory.observe(WorldSnapshot((0, 1), (0, 0), None), WorldEvent())
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unknown_route"


def test_unknown_route_when_goal_never_visited() -> None:
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, MapMemory(), WallMap(),
        goal_map=(9, 9), goal_cell=(0, 0),
    )
    assert result == "unknown_route"


def test_unreachable_when_door_cell_is_walled_off() -> None:
    # Start is sealed on all sides: the door cell can never be reached.
    walls = {((0, 0), (0, 0), d) for d in DIRECTIONS}
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), walls=walls)
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unreachable"


def test_lost_when_crossing_lands_on_unexpected_map() -> None:
    # Portal claims the door leads to B=(0,1), but the world sends us to C=(0,5).
    borders = {((0, 0), (2, 0), "right"): ((0, 5), (0, 0))}
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "lost"
```

- [ ] **Step 2: Run tests to verify they fail then pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -q`
Expected: the 5 new tests PASS with the Task 3 implementation (no code change needed — this task is pure test coverage of the already-implemented outcomes). If any fail, fix `env/map_traveler.py` until green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_map_traveler.py
git commit -m "test: travel_to chain + unknown_route/unreachable/lost outcomes"
```

---

### Task 5: ROM smoke test

**Files:**
- Create: `tests/test_map_traveler_rom.py`

- [ ] **Step 1: Write the ROM-gated smoke test**

Create `tests/test_map_traveler_rom.py`:

```python
"""map_traveler: ROM smoke test — real emulator wiring (gated on POKEMON_EMERALD_ROM)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")

# states/ lives in the main repo, not the worktree; resolve it absolutely.
_STATE = Path(__file__).resolve().parents[2] / "Emu" / "states" / "initial.state"


@pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")
def test_travel_same_map_arrives_on_real_rom() -> None:
    from emulator.gba import GbaEmulator
    from env.game_state import EmeraldReader
    from env.local_navigator import WallMap
    from env.map_memory import MapMemory
    from env.map_traveler import travel_to
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    emu.load_state(str(_STATE))
    reader = WorldReader(EmeraldReader(emu.read_bytes))
    snap = reader.snapshot()
    assert snap is not None

    # Same-map travel to the current cell must arrive immediately (delegates to
    # navigate_to, which returns "arrived" when pos == target). This exercises
    # the plan_route([here]) + map_id == goal_map branch on the real emulator.
    result = travel_to(
        emu, reader, MapMemory(), WallMap(),
        goal_map=snap.map_id, goal_cell=snap.pos,
    )
    assert result == "arrived"
```

Note: this smoke covers the same-map delegation path on the real ROM without
needing a captured door. Capturing an open-map savestate (`states/open_map.state`)
to make a real cross-map hop load-bearing is deferred to the mapping-mode step
(it also resolves step-1's Minor M4).

- [ ] **Step 2: Run the ROM smoke test**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler_rom.py -q`
Expected: PASS (1 test; or skips cleanly if the ROM env var is unset).

- [ ] **Step 3: Run the full suite + lint**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check .`
Expected: all tests PASS (143 prior + 5 map_memory + 1 live_navigator + 7 map_traveler + 1 ROM smoke), ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_map_traveler_rom.py
git commit -m "test: ROM smoke test for travel_to wiring"
```

---

## Notes for the implementer

- Run tests from the worktree with `PYTHONPATH` pointing at it and the main-repo venv/ROM:
  `PYTHONPATH=/Users/_eloi/Projets/Emu-p3-inter-map-navigation POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
  (The `.venv/` and `roms/` and `states/` all live in the main repo `/Users/_eloi/Projets/Emu`, not the worktree.)
- Code style: `from __future__ import annotations`, `int | None`, `tuple[int, int]`, snake_case/PascalCase/UPPER_CASE, ruff line-length 100. English comments/docstrings only.
- Bounded loops (code-safety rule #2): `travel_to`'s hop loop is bounded by `max_hops`; each `navigate_to` leg by its own `max_steps`; `_snapshot_settled` by `SETTLE_TRIES`. No recursion, no `while True`.
- DRY/YAGNI: reuse `plan_route`, `navigate_to`, `DELTAS`, `WallMap` unchanged. Do not add multi-door-per-pair support, mapping mode, or `"lost"` recovery — all deferred.
```
