# Grind Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Explorer its 2nd "savoir-reconnaître" skill — recognize grass by effect (a wild battle started here), then execute `mode="grind"`: travel to a known grass cell and walk in it until a battle starts.

**Architecture:** Exact mirror of heal mode. A new `EncounterWatcher` (edge detector on `reader.in_battle()`) learns grass cells during movement (`navigate_to`) and cartography (`map_map`), storing them in a generalized label-keyed `MapMemory` store (`cells_labeled("has_grass")`). `execute_order`'s `grind` branch travels to the first known grass cell, then treads in place until the watcher fires.

**Tech Stack:** Python 3.12, pytest (no ROM for these unit tests), ruff (line-length 100). Reuses P1–P4 modules: `WorldReader`, `EmeraldReader`, `BattleReader`, `MapMemory`, `travel_to`, `navigate_to`, `map_map`.

**Spec:** `docs/superpowers/specs/2026-07-31-grind-mode-design.md`

**Test command (from the worktree root):**
```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```
**Lint:** `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env tests tools`

---

## File Structure

- `env/world_reader.py` — MODIFY: `WorldReader` takes a raw read callable, owns a `BattleReader`, exposes `in_battle()`.
- `env/encounter_detector.py` — CREATE: `EncounterWatcher` edge detector.
- `env/map_memory.py` — MODIFY: label-keyed `_labeled_cells`, `cells_labeled()`, `healing_spots()` becomes a shortcut, `has_grass` cell storage.
- `env/live_navigator.py` — MODIFY: add `EncounterWatcher` learning branch alongside the existing `HealWatcher`.
- `env/map_explorer.py` — MODIFY: learn grass during `map_map` wandering.
- `env/orders.py` — MODIFY: fill the `grind` branch (`_execute_grind` + `_walk_until_encounter`).
- Tests: `test_world_reader.py`, `test_encounter_detector.py` (new), `test_map_memory.py`, `test_live_navigator.py`, `test_map_explorer.py`, `test_orders.py`, plus the 5 ROM-test call-site updates.

---

## Task 1: WorldReader owns a BattleReader and exposes `in_battle()`

**Files:**
- Modify: `env/world_reader.py`
- Modify: `tests/test_world_reader.py`
- Modify (call sites): `tools/capture_open_map.py:39`, `tests/test_live_navigator_rom.py:30`, `tests/test_world_surveyor_rom.py:31`, `tests/test_map_traveler_rom.py:26`, `tests/test_map_explorer_rom.py:33`
- Test: `tests/test_world_reader.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_world_reader.py` (after the existing tests). This test builds a `WorldReader` directly from a battle-RAM read function (no `FakeEmulator`), so it exercises the new constructor and the `in_battle()` passthrough:

```python
from env.game_state import (
    BATTLE_MON_SIZE,
    GBATTLE_MONS_ADDR,
    GBATTLE_OUTCOME_ADDR,
    GBATTLE_TYPE_FLAGS_ADDR,
    GMOVE_RESULT_FLAGS_ADDR,
)


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


def test_in_battle_true_when_a_battle_is_active() -> None:
    reader = WorldReader(_battle_read(in_battle=True))
    assert reader.in_battle() is True


def test_in_battle_false_out_of_battle() -> None:
    reader = WorldReader(_battle_read(in_battle=False))
    assert reader.in_battle() is False
```

Also change the existing helper at the top of the file:

```python
def _reader(emu: FakeEmulator) -> WorldReader:
    return WorldReader(emu.read_bytes)
```

and remove the now-unused `from env.game_state import EmeraldReader` line (line 4).

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_world_reader.py -q`
Expected: FAIL — `WorldReader(emu.read_bytes)` passes a callable where an `EmeraldReader` is expected (AttributeError inside `snapshot`), and `in_battle` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Rewrite `env/world_reader.py`'s imports and `WorldReader` class:

```python
from env.game_state import BattleReader, EmeraldReader, ReadFn


class WorldReader:
    """Wraps the RAM reader and returns an immutable WorldSnapshot each step."""

    def __init__(self, read: ReadFn) -> None:
        self._reader = EmeraldReader(read)
        self._battle = BattleReader(read)

    def snapshot(self) -> WorldSnapshot | None:
        """Snapshot the world, or None while the save blocks relocate."""
        ps = self._reader.player_state()
        if ps is None:
            return None
        return WorldSnapshot(
            map_id=(ps.map_group, ps.map_num),
            pos=(ps.x, ps.y),
            tile_behavior=self._tile_behavior(),
        )

    def party_hp(self) -> list[tuple[int, int]]:
        """Passthrough to the RAM reader: (current, max) HP per party member."""
        return self._reader.party_hp()

    def in_battle(self) -> bool:
        """True while a wild/trainer battle is active."""
        return self._battle.battle_state().in_battle

    def _tile_behavior(self) -> int | None:
        # TODO(probe): read the metatile-behavior byte of the tile the player
        # stands on (tall grass, water, wall, door, ...). Its RAM address on
        # BPEF is not yet known; returns None until a probe session finds it.
        return None
```

Then update the 5 remaining call sites from `WorldReader(EmeraldReader(<x>))` to `WorldReader(<x>)`:
- `tools/capture_open_map.py:39`: `reader = WorldReader(env.emulator.read_bytes)`
- `tests/test_live_navigator_rom.py:30`: `reader = WorldReader(emu.read_bytes)`
- `tests/test_world_surveyor_rom.py:31`: `reader = WorldReader(emulator.read_bytes)`
- `tests/test_map_traveler_rom.py:26`: `reader = WorldReader(emu.read_bytes)`
- `tests/test_map_explorer_rom.py:33`: `reader = WorldReader(emulator.read_bytes)`

In each of those 5 files, if `EmeraldReader` is now unused, remove its import (run ruff to confirm — see Step 4).

- [ ] **Step 4: Run tests + lint to verify they pass**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_world_reader.py -q`
Expected: PASS (all 5 world_reader tests).

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env tests tools`
Expected: clean. If ruff reports `F401 EmeraldReader imported but unused` in any ROM test file or `test_world_reader.py`, delete that import line and re-run.

- [ ] **Step 5: Commit**

```bash
git add env/world_reader.py tests/test_world_reader.py tools/capture_open_map.py \
  tests/test_live_navigator_rom.py tests/test_world_surveyor_rom.py \
  tests/test_map_traveler_rom.py tests/test_map_explorer_rom.py
git commit -m "feat: WorldReader owns a BattleReader + in_battle() passthrough"
```

---

## Task 2: EncounterWatcher edge detector

**Files:**
- Create: `env/encounter_detector.py`
- Test: `tests/test_encounter_detector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_encounter_detector.py`:

```python
"""EncounterWatcher: fire once on the not-in-battle -> in-battle edge (no ROM)."""
from __future__ import annotations

from env.encounter_detector import EncounterWatcher


def test_fires_on_absent_to_present_edge() -> None:
    w = EncounterWatcher()
    assert w.observe(False) is False   # walking, no battle
    assert w.observe(True) is True     # a wild battle just started


def test_silent_when_already_in_battle_on_first_read() -> None:
    w = EncounterWatcher()
    assert w.observe(True) is False    # optimistic init: not counted as a new start


def test_silent_on_present_to_absent() -> None:
    w = EncounterWatcher()
    w.observe(False)
    w.observe(True)
    assert w.observe(False) is False   # battle ended: not an edge we care about


def test_fires_again_after_a_battle_ends_and_a_new_one_starts() -> None:
    w = EncounterWatcher()
    w.observe(False)
    assert w.observe(True) is True     # first battle
    w.observe(False)                   # battle ends
    assert w.observe(True) is True     # a second battle starts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_encounter_detector.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.encounter_detector'`.

- [ ] **Step 3: Write minimal implementation**

Create `env/encounter_detector.py`:

```python
"""encounter_detector: recognise grass by its effect (a wild battle started here).

Pure logic, no emulator: a boolean 'am I in battle now' goes in, a single
boolean 'a wild battle just started' comes out. Structural twin of
heal_detector.HealWatcher, but watches the battle flag instead of party HP.
Reused both to LEARN a grass cell (during movement/cartography) and to KNOW
when grind's walk-loop has triggered a battle.
"""
from __future__ import annotations


class EncounterWatcher:
    """Fires once on the step where a wild battle transitions from absent to present."""

    def __init__(self) -> None:
        # Start optimistic so an already-in-battle first read is not a spurious start.
        self._was_in_battle = True

    def observe(self, in_battle: bool) -> bool:
        """Feed the current battle flag; returns True only on the absent -> present edge."""
        started = in_battle and not self._was_in_battle
        self._was_in_battle = in_battle
        return started
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_encounter_detector.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add env/encounter_detector.py tests/test_encounter_detector.py
git commit -m "feat: EncounterWatcher detects the not-in-battle -> in-battle edge"
```

---

## Task 3: Generalize MapMemory to label-keyed recognition cells

**Files:**
- Modify: `env/map_memory.py`
- Test: `tests/test_map_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_map_memory.py` (it already imports `MapMemory`, `WorldEvent`, and builds `WorldSnapshot`; reuse the same style as the existing healing-cell tests):

```python
def test_cells_labeled_remembers_a_grass_cell_on_encounter() -> None:
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    assert memory.cells_labeled("has_grass") == [((0, 16), (5, 12))]


def test_cells_labeled_is_empty_without_any_encounter() -> None:
    assert MapMemory().cells_labeled("has_grass") == []


def test_grass_cell_is_last_write_wins_per_map() -> None:
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    memory.observe(WorldSnapshot((0, 16), (7, 3), None), WorldEvent(encounter_started=True))
    assert memory.cells_labeled("has_grass") == [((0, 16), (7, 3))]


def test_healing_spots_is_a_shortcut_for_cells_labeled() -> None:
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    assert memory.healing_spots() == memory.cells_labeled("healing_spot")
    assert memory.healing_spots() == [((0, 9), (3, 10))]


def test_grass_and_healing_labels_do_not_cross_contaminate() -> None:
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    assert memory.cells_labeled("healing_spot") == [((0, 9), (3, 10))]
    assert memory.cells_labeled("has_grass") == [((0, 16), (5, 12))]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_memory.py -q`
Expected: FAIL — `AttributeError: 'MapMemory' object has no attribute 'cells_labeled'`.

- [ ] **Step 3: Write minimal implementation**

In `env/map_memory.py`:

Replace the `_healing_cells` field in `__init__` (currently the last line of `__init__`, around line 56–57) with a label-keyed store:

```python
        # recognition label -> {map_id: cell} (last-write-wins per map).
        self._labeled_cells: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
```

Replace the `observe` label block (currently lines 64–68) with:

```python
        if event.healed:
            node.labels.add("healing_spot")
            self._labeled_cells.setdefault("healing_spot", {})[snapshot.map_id] = snapshot.pos
        if event.encounter_started:
            node.labels.add("has_grass")
            self._labeled_cells.setdefault("has_grass", {})[snapshot.map_id] = snapshot.pos
```

Replace the `healing_spots` method (currently lines 104–106) with a shortcut plus the new generic accessor:

```python
    def cells_labeled(
        self, label: str
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """All (map_id, cell) pairs remembered under the given recognition label."""
        return [(map_id, cell) for map_id, cell in self._labeled_cells.get(label, {}).items()]

    def healing_spots(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Known healing locations as (map_id, cell); shortcut for cells_labeled('healing_spot')."""
        return self.cells_labeled("healing_spot")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_memory.py -q`
Expected: PASS (all existing healing-cell tests still green + 5 new).

- [ ] **Step 5: Commit**

```bash
git add env/map_memory.py tests/test_map_memory.py
git commit -m "feat: MapMemory label-keyed recognition cells (has_grass + healing_spot)"
```

---

## Task 4: `navigate_to` learns grass while moving

**Files:**
- Modify: `env/live_navigator.py`
- Modify: `tests/test_live_navigator.py` (add `in_battle()` stub to `FakeWorld`, add a grass-learning fake + test)
- Test: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing test + fakes**

In `tests/test_live_navigator.py`, add an always-`False` `in_battle()` to the base `FakeWorld` (right after its `party_hp` method at line 71–73) so every fake reader answers the new call when `memory` is set:

```python
    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False
```

Then add a grass-learning fake + test at the end of the file:

```python
class EncounterFakeWorld(FakeWorld):
    """Extends FakeWorld: a wild battle starts once the player reaches grass_at."""

    def __init__(self, grass_at: tuple[int, int], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._grass_at = grass_at

    def in_battle(self) -> bool:
        return self.pos == self._grass_at


def test_learns_grass_cell_on_the_in_battle_edge() -> None:
    # Walking (0,0)->(2,0); a battle fires on (1,0), which must be tagged has_grass.
    world = EncounterFakeWorld(grass_at=(1, 0), start=(0, 0))
    memory = MapMemory()
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50, memory=memory)
    assert result == "arrived"
    assert memory.cells_labeled("has_grass") == [((0, 0), (1, 0))]


def test_navigate_without_memory_ignores_battles() -> None:
    world = EncounterFakeWorld(grass_at=(1, 0), start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50)
    assert result == "arrived"  # memory=None: no learning path taken, no crash
```

Add `MapMemory` to the imports at the top of the test file if not already present (it is imported: `from env.map_memory import MapMemory, Portal`).

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: FAIL on `test_learns_grass_cell_on_the_in_battle_edge` — `cells_labeled("has_grass")` is empty because `navigate_to` does not yet observe battles.

- [ ] **Step 3: Write minimal implementation**

In `env/live_navigator.py`:

Add the import (next to the heal import at line 13):

```python
from env.encounter_detector import EncounterWatcher
from env.heal_detector import HealWatcher
```

Rename the single watcher to `heal_watcher` and add `enc_watcher`. Replace lines 46 and 52–53 so the loop head reads:

```python
    heal_watcher = HealWatcher()
    enc_watcher = EncounterWatcher()
    for _ in range(max_steps):
        before = reader.snapshot()
        if before is None:
            emulator.step(0, RELEASE_FRAMES)   # relocating; idle a beat and retry
            continue
        if memory is not None:
            if heal_watcher.observe(reader.party_hp()):
                memory.observe(before, WorldEvent(healed=True))
            if enc_watcher.observe(reader.in_battle()):
                memory.observe(before, WorldEvent(encounter_started=True))
        if before.pos == target:
            return "arrived"
```

(Everything below the `arrived` check is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (all existing tests + 2 new).

- [ ] **Step 5: Commit**

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "feat: navigate_to learns grass cells on the in-battle edge"
```

---

## Task 5: `map_map` learns grass while wandering

**Files:**
- Modify: `env/map_explorer.py`
- Modify: `tests/test_map_explorer.py` (add `in_battle()` to `ExploreWorld`, add a grass-learning subclass + test)
- Test: `tests/test_map_explorer.py`

- [ ] **Step 1: Write the failing test + fakes**

In `tests/test_map_explorer.py`, add an always-`False` `in_battle()` to `ExploreWorld` (right after its `snapshot` method at line 61–62):

```python
    def in_battle(self) -> bool:
        return False
```

Then add a grass-learning subclass + test. This world reports a battle the first time the survey stands on `grass_at`:

```python
class EncounterExploreWorld(ExploreWorld):
    """ExploreWorld that reports a wild battle while standing on grass_at."""

    def __init__(self, grass_at: tuple[int, int], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._grass_at = grass_at

    def in_battle(self) -> bool:
        return self.pos == self._grass_at


def test_map_map_learns_grass_cell_when_a_battle_fires():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)  # cells (0,0),(1,0),(0,1),(1,1)
    world = EncounterExploreWorld(grass_at=(1, 0), map_id=target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "complete"
    assert ((3, 3), (1, 0)) in memory.cells_labeled("has_grass")
```

Note: `ExploreWorld.__init__` is `(self, map_id, start, walls, borders=None)`; the subclass passes them through as kwargs, so call it with keyword args (`map_id=`, `start=`, `walls=`) as shown.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer.py -q`
Expected: FAIL on `test_map_map_learns_grass_cell_when_a_battle_fires` — `cells_labeled("has_grass")` empty because `map_map` does not observe battles yet.

- [ ] **Step 3: Write minimal implementation**

In `env/map_explorer.py`:

Update the import (line 14) to bring in `WorldEvent`, and add the encounter import:

```python
from env.encounter_detector import EncounterWatcher
from env.map_memory import MapMemory, WorldEvent
```

Instantiate the watcher just before the survey loop (before `for _ in range(max_steps):` at line 42):

```python
    enc_watcher = EncounterWatcher()
```

Inside the loop, right after `reached.add(here.pos)` (line 49), add the learning branch:

```python
        reached.add(here.pos)
        if enc_watcher.observe(reader.in_battle()):
            memory.observe(here, WorldEvent(encounter_started=True))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer.py -q`
Expected: PASS (all existing survey tests + 1 new).

- [ ] **Step 5: Commit**

```bash
git add env/map_explorer.py tests/test_map_explorer.py
git commit -m "feat: map_map learns grass cells as it wanders"
```

---

## Task 6: Fill the `grind` branch in `execute_order`

**Files:**
- Modify: `env/orders.py`
- Modify: `tests/test_orders.py` (replace the `not_implemented` test, add `in_battle()` stubs, add a `GrassWorld` fake + grind tests)
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing tests + fakes**

In `tests/test_orders.py`:

1. Add an always-`False` `in_battle()` stub to both existing fakes. After `NamedWorld.party_hp` (line 86–88):

```python
    def in_battle(self) -> bool:
        return False
```

and after `HealWorld.party_hp` (line 147–149):

```python
    def in_battle(self) -> bool:
        return False
```

2. Delete the now-obsolete test `test_non_nav_mode_is_not_implemented_even_for_a_known_place` (lines 37–40) — grind now acts, so it no longer returns `"not_implemented"`.

3. Add a `GrassWorld` fake and grind tests at the end of the file:

```python
# ---------------------------------------------------------------------------
# Grind tests
# ---------------------------------------------------------------------------


class GrassWorld:
    """Single-map fake: treading triggers a wild battle after N steps."""

    def __init__(
        self,
        map_id: tuple[int, int],
        cell: tuple[int, int],
        steps_to_encounter: int = 3,
    ) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_enc = steps_to_encounter
        self._steps = 0

    def step(self, keys: int, frames: int) -> None:
        if _KEY_TO_DIR.get(keys) is not None:
            self._steps += 1  # count d-pad presses; releases (keys=0) do not count

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]  # full: heal watcher stays quiet

    def in_battle(self) -> bool:
        return self._steps >= self._to_enc


def test_grind_without_known_grass_returns_no_grass_spot_known() -> None:
    world = GrassWorld((0, 16), (5, 12))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "no_grass_spot_known"


def test_grind_on_known_grass_starts_an_encounter() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "encounter_started"


def test_grind_that_never_battles_returns_no_encounter() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=10_000)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "no_encounter"


def test_grind_passes_through_travel_failure() -> None:
    # Grass is remembered on a map with no known route from here -> unknown_route.
    world = GrassWorld((0, 9), (3, 10))
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 99), (1, 1), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "unknown_route"


def test_grind_ignores_the_order_destination() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="not_a_registered_place", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "encounter_started"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q`
Expected: FAIL — the grind tests get `"not_implemented"` (current stub) instead of the new outcomes.

- [ ] **Step 3: Write minimal implementation**

In `env/orders.py`:

Update the imports (lines 14–16) to add the encounter watcher:

```python
from emulator import buttons
from env.encounter_detector import EncounterWatcher
from env.heal_detector import party_is_full
from env.map_traveler import travel_to
```

Add grind constants after the heal constants (line 38):

```python
GRIND_STEP_FRAMES = 24
GRIND_RELEASE_FRAMES = 8
GRIND_MAX_STEPS = 60    # bound the walk (code-safety #2)
```

Replace the body of `execute_order` (lines 59–69) so grind is handled next to heal, before destination resolution, and the `not_implemented` line is gone:

```python
    if order.mode == "heal":
        return _execute_heal(emulator, reader, memory, wallmap, max_hops=max_hops)
    if order.mode == "grind":
        return _execute_grind(emulator, reader, memory, wallmap, max_hops=max_hops)
    dest = DESTINATIONS.get(order.destination)
    if dest is None:
        return "unknown_destination"
    goal_map, goal_cell = dest
    return travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
```

Update the `execute_order` docstring (lines 49–58) to drop `"not_implemented"` and list grind's outcomes:

```python
    """Execute an order dispatched by the Strategist.

    heal and grind are resolved first and ignore `destination` (pure intention:
    the healing spot / grass cell comes from memory, not the order's name).
    advance requires `destination` to be in DESTINATIONS.

    Returns "unknown_destination" | "no_healing_spot_known" | "no_grass_spot_known" |
    one of travel_to's outcomes ("arrived" | "unknown_route" | "unreachable" |
    "lost" | "timeout") | "healed" | "heal_failed" | "encounter_started" |
    "no_encounter".
    """
```

Add the two grind functions after `_heal_here` (end of file):

```python
def _execute_grind(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
) -> str:
    """Travel to a known grass cell, then tread in it until a wild battle starts.

    Returns "no_grass_spot_known" | a travel_to pass-through | "encounter_started" |
    "no_encounter".
    """
    spots = memory.cells_labeled("has_grass")
    if not spots:
        return "no_grass_spot_known"
    goal_map, goal_cell = spots[0]   # v1: first known spot (nearest-choice is later)
    outcome = travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    return _walk_until_encounter(emulator, reader)


def _walk_until_encounter(emulator: Any, reader: Any) -> str:
    """Cycle d-pad directions in place until the battle flag rises."""
    watcher = EncounterWatcher()
    directions = (buttons.KEY_UP, buttons.KEY_DOWN, buttons.KEY_LEFT, buttons.KEY_RIGHT)
    for i in range(GRIND_MAX_STEPS):
        if watcher.observe(reader.in_battle()):
            return "encounter_started"
        emulator.step(directions[i % len(directions)], GRIND_STEP_FRAMES)
        emulator.step(0, GRIND_RELEASE_FRAMES)   # release between presses (GBA debounce)
    return "encounter_started" if reader.in_battle() else "no_encounter"
```

Also add `from env.map_memory import ...`? No — `execute_order` receives `memory` as `Any`; no import needed. But the test file already imports `WorldEvent` from `env.map_memory`. Fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q`
Expected: PASS (advance + heal tests unchanged + 5 grind tests).

- [ ] **Step 5: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "feat: execute_order fills mode=grind (travel + tread until a battle starts)"
```

---

## Task 7: Full suite, lint, and ROM-smoke deferral note

**Files:**
- Modify: `env/orders.py` (module docstring only)

- [ ] **Step 1: Run the full test suite with the ROM**

Run:
```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```
Expected: all previous tests + the new grind tests PASS (1 skipped is the pre-existing KNOWN_PLACES skip). If a ROM smoke fails because `states/*.state` is missing, it should SKIP, not fail — confirm the count of skips is unchanged from before this branch.

- [ ] **Step 2: Lint**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env tests tools`
Expected: clean. Fix any `F401` unused imports (most likely leftover `EmeraldReader` in the ROM test files from Task 1) or `E501` line-length issues, then re-run.

- [ ] **Step 3: Update the orders module docstring**

The module docstring in `env/orders.py` (lines 1–8) still says grind "is stubbed". Update it:

```python
"""orders: the shared "order" language between the three brains.

The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. "advance" navigates via travel_to;
"heal" travels to a known healing spot and presses A until the party is full;
"grind" travels to a known grass cell and treads until a wild battle starts.
The combat directive is stored for a future Fighter hookup. No Strategist,
no reward here.

ROM smoke for grind is deferred: it needs a savestate standing on/near grass
with a party and a deterministic encounter, which we do not have yet.
"""
```

- [ ] **Step 4: Re-run the full suite once more**

Run:
```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
PYTHONPATH=$(pwd) /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```
Expected: PASS (same counts as Step 1).

- [ ] **Step 5: Commit**

```bash
git add env/orders.py
git commit -m "docs: mark grind mode wired in orders module docstring"
```

---

## Self-Review Notes

- **Spec coverage:** §1 WorldReader in_battle → Task 1. §2 EncounterWatcher → Task 2. §3 MapMemory label-keyed store → Task 3. §4 navigate_to learning → Task 4. §5 map_map learning → Task 5. §6 orders grind branch → Task 6. Outcome contract (`no_grass_spot_known` / travel pass-through / `encounter_started` / `no_encounter`) → Task 6 tests. Fake `in_battle()` stubs across all readers passed with memory → Tasks 4/5/6. ROM-smoke deferral → Task 7. No spec section is left without a task.
- **Type consistency:** `EncounterWatcher.observe(bool) -> bool` used identically in Tasks 4/5/6. `cells_labeled(str) -> list[tuple[map_id, cell]]` defined in Task 3, consumed in Task 6. `WorldReader.in_battle() -> bool` defined in Task 1, consumed by fakes' matching signature in Tasks 4/5/6. `_execute_grind` signature matches `_execute_heal` (no `order`).
- **Removed test:** Task 6 deletes `test_non_nav_mode_is_not_implemented_even_for_a_known_place` because grind now acts — this is the one existing test whose behavior the feature changes; every other existing test stays green (always-`False` `in_battle()` and always-full `party_hp()` keep both watchers silent).
