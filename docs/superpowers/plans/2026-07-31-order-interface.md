# Order Interface (P4 step 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the three brains a shared "order" language: an `Order` object the Strategist can emit and the Explorer can execute by walking to a named destination.

**Architecture:** One new file `env/orders.py`, nothing else touched. An `Order` frozen dataclass (destination name + mode + combat directive), a hand-written `DESTINATIONS` registry mapping names to (map_id, cell), and `execute_order` which resolves the name and delegates navigation to the existing `travel_to` brick. `grind`/`heal` modes are stubbed (`"not_implemented"`); the combat directive is stored but unused.

**Tech Stack:** Python 3.12, pytest, existing `env/map_traveler.travel_to`, `env/map_memory.MapMemory`, `env/local_navigator.WallMap`.

---

## Context for the implementer

You are working in the git worktree `/Users/_eloi/Projets/Emu-p4-order-interface`
(branch `feat/p4-order-interface`). The `.venv`, `roms/` and `states/` live in the
**main** repo `/Users/_eloi/Projets/Emu`, NOT in the worktree.

**Run tests with this exact prefix** (targeted file, no ROM needed — these tests are pure):

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```

**Full suite** (some tests are ROM-gated and skip cleanly without the ROM):

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```

**Lint:** `/Users/_eloi/Projets/Emu/.venv/bin/ruff check .` (line-length 100).

### The brick you delegate to (existing, DO NOT modify)

`env/map_traveler.py`:

```python
def travel_to(
    emulator, reader, memory, wallmap,
    goal_map: tuple[int, int], goal_cell: tuple[int, int], max_hops: int = 20,
) -> str:
    # returns "arrived" | "unknown_route" | "unreachable" | "lost" | "timeout"
```

It walks door-to-door over **known territory only**: same-map goals delegate to
`navigate_to`; multi-map goals need portals already recorded in `memory` (else
`"unknown_route"`).

### The test fake (copied from `tests/test_map_traveler.py`, self-contained)

Each test file in this repo defines its own fake that plays BOTH the emulator
(`step(keys, frames)`) and the reader (`snapshot()`). Reuse this exact pattern.

---

## Task 1: `Order` dataclass + `DESTINATIONS` registry

**Files:**
- Create: `env/orders.py`
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_orders.py`:

```python
"""orders: the shared Order language + execute_order (pure, no ROM)."""
from __future__ import annotations

import dataclasses

from env.orders import DESTINATIONS, Order


def test_order_is_a_frozen_dataclass_with_three_fields() -> None:
    order = Order(destination="route_101", mode="advance", combat="win")
    assert order.destination == "route_101"
    assert order.mode == "advance"
    assert order.combat == "win"
    # frozen: reassigning a field must raise
    try:
        order.mode = "grind"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Order must be frozen")


def test_destinations_registry_holds_known_places() -> None:
    assert DESTINATIONS["littleroot"] == ((0, 9), (3, 10))
    assert DESTINATIONS["route_101"] == ((0, 16), (5, 12))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'env.orders'`.

- [ ] **Step 3: Write minimal implementation**

Create `env/orders.py`:

```python
"""orders: the shared "order" language between the three brains.

The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. This step wires only the navigation
mode ("advance") through to travel_to; "grind"/"heal" are stubbed and the combat
directive is stored for a future Fighter hookup. No Strategist, no reward here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    """A single order: go to `destination`, in `mode`, fighting per `combat`."""

    destination: str   # named place, e.g. "route_101"
    mode: str          # "advance" | "grind" | "heal" (only "advance" acts now)
    combat: str        # "win" | "capture" | "min_loss" (stored for later)


# Hand-written name -> (map_id, cell) registry. Chosen over a map-memory lookup on
# purpose: a name means something to the chef before any exploration has happened.
# Cells are known landmarks; fix them here if one turns out wrong.
DESTINATIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "littleroot": ((0, 9), (3, 10)),   # Bourg-en-Vol, truck landing cell
    "route_101": ((0, 16), (5, 12)),   # Route 101, south entrance from Littleroot
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "feat: Order dataclass + DESTINATIONS registry"
```

---

## Task 2: `execute_order` — unknown destination + non-nav mode stub

These two branches return **before** touching the emulator, so no world fake is
needed yet.

**Files:**
- Modify: `env/orders.py`
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orders.py`:

```python
from env.local_navigator import WallMap
from env.map_memory import MapMemory
from env.orders import execute_order


def test_unknown_destination_returns_unknown_destination() -> None:
    order = Order(destination="atlantide", mode="advance", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "unknown_destination"


def test_non_nav_mode_is_not_implemented_even_for_a_known_place() -> None:
    order = Order(destination="littleroot", mode="grind", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "not_implemented"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```
Expected: FAIL with `ImportError: cannot import name 'execute_order'`.

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `env/orders.py`:

```python
from typing import Any

from env.map_traveler import travel_to
```

Append this function to `env/orders.py`:

```python
def execute_order(
    order: Order,
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
) -> str:
    """Resolve the order's destination and hand navigation to travel_to.

    Returns "unknown_destination" | "not_implemented" | one of travel_to's
    outcomes ("arrived" | "unknown_route" | "unreachable" | "lost" | "timeout").
    """
    dest = DESTINATIONS.get(order.destination)
    if dest is None:
        return "unknown_destination"
    if order.mode != "advance":
        return "not_implemented"   # grind/heal wiring is a later step
    goal_map, goal_cell = dest
    return travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "feat: execute_order resolves destination + stubs non-nav modes"
```

---

## Task 3: `execute_order` — advance path delegates to `travel_to`

Exercise the real navigation path against a fake world: same-map arrival,
multi-map arrival, and outcome pass-through when the route is unknown.

**Files:**
- Modify: `tests/test_orders.py` (add the fake + 3 tests)
- No change to `env/orders.py` (already delegates; these tests must pass as-is)

- [ ] **Step 1: Write the failing test**

Add the fake and tests to `tests/test_orders.py`. Put the imports at the top with
the others:

```python
from emulator import buttons
from env.world_reader import WorldSnapshot
```

Then append:

```python
_KEY_TO_DIR: dict[int, str] = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}
_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class NamedWorld:
    """Hidden multi-map grid that plays emulator (step) and reader (snapshot).

    `borders` maps (map_id, cell, direction) -> (next_map, entry_cell). Movement
    inside a map is free unless the edge is in `walls`.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        borders: dict[tuple[tuple[int, int], tuple[int, int], str],
                      tuple[tuple[int, int], tuple[int, int]]] | None = None,
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._borders = dict(borders or {})

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        border = self._borders.get((self.map_id, self.pos, direction))
        if border is not None:
            self.map_id, self.pos = border
            return
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def test_advance_to_same_map_destination_arrives() -> None:
    # "littleroot" is ((0, 9), (3, 10)); start on that map, walk to the cell.
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="littleroot", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "arrived"
    assert world.map_id == (0, 9)
    assert world.pos == (3, 10)


def test_advance_across_one_known_door_arrives() -> None:
    # From littleroot (0,9), a recorded door at (2,10) pressing 'right' lands on
    # route_101 (0,16) at (0,12); then walk to the destination cell (5,12).
    borders = {((0, 9), (2, 10), "right"): ((0, 16), (0, 12))}
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10), borders=borders)
    memory = MapMemory()
    memory.record_portal(
        (0, 9), (2, 10), "right", (0, 16), reversible=True, to_cell=(0, 12)
    )
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "arrived"
    assert world.map_id == (0, 16)
    assert world.pos == (5, 12)


def test_advance_passes_through_unknown_route() -> None:
    # route_101 is on a different map, but no portal was recorded and the goal
    # map was never visited -> travel_to returns "unknown_route", passed through.
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "unknown_route"
```

- [ ] **Step 2: Run test to verify it fails first, then passes**

The implementation from Task 2 already delegates, so these should PASS once the
fake compiles. If any fail, the bug is in the fake or the test data — fix the test.

Run:
```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```
Expected: PASS (7 passed).

- [ ] **Step 3: Run the full suite + lint**

Run:
```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-order-interface \
  POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
/Users/_eloi/Projets/Emu/.venv/bin/ruff check .
```
Expected: all prior tests still pass (nothing existing was modified), +7 new, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_orders.py
git commit -m "test: execute_order advance path delegates to travel_to"
```

---

## Self-review notes (already applied)

- **Spec coverage:** Order dataclass (Task 1), DESTINATIONS registry (Task 1),
  execute_order resolution + `unknown_destination` + `not_implemented` stub
  (Task 2), advance delegation + all 5 test cases from the spec (same-map,
  multi-hop, unknown name, non-nav mode, unreachable/unknown pass-through)
  across Tasks 1-3. Non-goals (Strategist, milestones.py, reward, Fighter,
  grind/heal wiring) are respected — none of those files are touched.
- **Type consistency:** `Order(destination, mode, combat)` and
  `execute_order(order, emulator, reader, memory, wallmap, max_hops)` are used
  identically in every task; `travel_to`'s signature matches
  `env/map_traveler.py`.
- **No placeholders:** every code + command block is complete.
