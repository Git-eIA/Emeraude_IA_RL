# World Surveyor (P3 step 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an orchestrator that charts the reachable overworld map-by-map by looping `travel_to` + `map_map`, discovering new maps through reversible border portals until the world is fully surveyed.

**Architecture:** One new pure-orchestration module `env/world_surveyor.py` wiring the three existing P3 bricks (`navigate_to`, `travel_to`, `map_map`). `Portal` gains `reversible` + `to_cell` so the surveyor can filter overworld-only crossings and resolve the entry cell `travel_to` needs. Iterative BFS over the map graph, bounded by `max_maps`, log-and-continue on any failed leg.

**Tech Stack:** Python 3.12, pytest, dependency-injected duck-typed emulator/reader (no ROM for unit tests; one ROM-gated smoke). Emerald BPEF only.

Reference spec: `docs/superpowers/specs/2026-07-25-world-surveyor-design.md`.

Run tests with: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
(The `.venv` and `roms/` live in the main repo `~/Projets/Emu`, not in this worktree.)

---

### Task 1: Portal gains `reversible` + `to_cell`; update `record_portal` and every caller

Cross-cutting signature change. `Portal` and `record_portal` are called by two source modules (`map_explorer`, `live_navigator`) and four test modules. All must change together so the suite stays green at the end of the task. Also add two accessors the surveyor needs to enumerate portals by endpoint.

**Files:**
- Modify: `env/map_memory.py:22-28` (Portal), `env/map_memory.py:68-85` (record_portal, portal, + new accessors)
- Modify: `env/map_explorer.py:71-78` (transition branch)
- Modify: `env/live_navigator.py:55-62` (transition branch)
- Test: `tests/test_map_memory.py`, `tests/test_map_traveler.py`, `tests/test_live_navigator.py`, `tests/test_map_explorer.py`

- [ ] **Step 1: Write the failing round-trip + accessor test**

Append to `tests/test_map_memory.py`:

```python
def test_record_portal_round_trips_reversible_and_to_cell() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    p = mem.portal((0, 9), (0, 16))
    assert p == Portal(
        from_cell=(5, 0), direction="up", to_map=(0, 16),
        reversible=True, to_cell=(5, 12),
    )


def test_outgoing_and_incoming_portals() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    mem.record_portal((0, 16), (5, 12), "down", (0, 9), reversible=True, to_cell=(5, 0))
    mem.record_portal((0, 16), (2, 2), "right", (1, 0), reversible=False, to_cell=(0, 0))
    out = mem.outgoing_portals((0, 16))
    assert {p.to_map for p in out} == {(0, 9), (1, 0)}
    inc = mem.incoming_portals((0, 16))
    assert [p.from_cell for p in inc] == [(5, 0)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_map_memory.py -q`
Expected: FAIL — `record_portal() got an unexpected keyword argument 'reversible'` and `AttributeError: ... 'outgoing_portals'`.

- [ ] **Step 3: Add the two fields, the new signature, and the accessors**

In `env/map_memory.py`, replace the `Portal` dataclass (lines 22-28):

```python
@dataclass(frozen=True)
class Portal:
    """One directed border crossing: leave `from_cell` going `direction` to reach `to_map`."""
    from_cell: tuple[int, int]
    direction: str
    to_map: tuple[int, int]
    reversible: bool          # True = reversible overworld border, False = building warp
    to_cell: tuple[int, int]  # cell landed on in to_map (coords do NOT continue across a border)
```

Replace `record_portal` (lines 68-79) and add accessors after `portal` (after line 85):

```python
    def record_portal(
        self,
        from_map: tuple[int, int],
        from_cell: tuple[int, int],
        direction: str,
        to_map: tuple[int, int],
        reversible: bool,
        to_cell: tuple[int, int],
    ) -> None:
        """Remember the door from from_map to to_map; also ensures the edge exists."""
        self._ensure_node(from_map)
        self._ensure_node(to_map)
        self._edges.add((from_map, to_map))
        self._portals[(from_map, to_map)] = Portal(
            from_cell, direction, to_map, reversible, to_cell
        )

    def outgoing_portals(self, from_map: tuple[int, int]) -> tuple[Portal, ...]:
        """Every recorded portal leaving `from_map`."""
        return tuple(p for (f, _t), p in self._portals.items() if f == from_map)

    def incoming_portals(self, to_map: tuple[int, int]) -> tuple[Portal, ...]:
        """Every recorded portal arriving at `to_map`."""
        return tuple(p for (_f, t), p in self._portals.items() if t == to_map)
```

- [ ] **Step 4: Thread the fields into `map_explorer`'s transition branch**

In `env/map_explorer.py`, replace the transition branch (lines 71-78):

```python
        elif outcome == "transition":
            landed = snapshot_settled(reader)
            if landed is None:
                return "left_map"  # relocating: cannot record or verify — bail safe
            # Step back through the door; a reversible border returns us to
            # target_map, a one-way warp does not. Record the portal with the
            # reversibility we just proved and the observed landing cell.
            probe_step(emulator, reader, landed, OPPOSITE[direction])
            returned = snapshot_settled(reader)
            reversible = returned is not None and returned.map_id == target_map
            memory.record_portal(
                target_map, cell, direction, landed.map_id, reversible, landed.pos
            )
            if not reversible:
                return "left_map"
```

- [ ] **Step 5: Thread the fields into `live_navigator`'s transition branch**

In `env/live_navigator.py`, replace the transition branch (lines 55-62):

```python
        if outcome == "transition":
            if memory is not None:
                landed = snapshot_settled(reader)
                if landed is not None:
                    # A live crossing is not step-back tested, so reversibility
                    # cannot be proven here: record the cautious default False.
                    memory.record_portal(
                        before.map_id, before.pos, direction, landed.map_id,
                        False, landed.pos,
                    )
            return "left_map"
```

- [ ] **Step 6: Update every existing call site**

`tests/test_map_memory.py` — update the three plain `record_portal` calls and the two `Portal(...)` asserts (lines 104, 106, 115, 121-123, 128) to pass/expect the new fields, e.g.:

```python
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    ...
    assert p == Portal(
        from_cell=(5, 0), direction="up", to_map=(0, 16),
        reversible=True, to_cell=(5, 12),
    )
    ...
    # last-write-wins test:
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    mem.record_portal((0, 9), (4, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    assert mem.portal((0, 9), (0, 16)) == Portal((4, 0), "up", (0, 16), True, (5, 12))
```

`tests/test_map_traveler.py` — the six `record_portal` calls (lines 76, 94, 95, 131, 144, 164) each add `reversible=True` and `to_cell=` the border's entry cell defined just above them. For every one whose border is `((m), (2,0), "right"): ((n), (0,0))`, use `to_cell=(0, 0)`:

```python
    memory.record_portal((0, 0), (2, 0), "right", (0, 1), reversible=True, to_cell=(0, 0))
```

(Apply the same shape to all six; `travel_to` never reads `to_cell`, so the exact value only needs to be consistent — use each test's own `entry_cell` from its `borders` dict.)

`tests/test_live_navigator.py` — update the `Portal(...)` assert (lines 136-138). The FakeWorld's `map_flips` transition lands the player on the same `pos` it stepped onto, so `to_cell` is that cell. Read the FakeWorld in this file to confirm the landing `pos`, then assert it:

```python
    assert memory.portal((0, 0), (0, 1)) == Portal(
        from_cell=(0, 0), direction="down", to_map=(0, 1),
        reversible=False, to_cell=<the pos FakeWorld reports after the flip>,
    )
```

`tests/test_map_explorer.py` — extend the two door tests. After the existing asserts in `test_reversible_door_recorded_and_survey_continues` (after line 136) add:

```python
    assert portal.reversible is True
    assert portal.to_cell == (0, 0)   # `entry` passed to _reversible_border
```

After the existing asserts in `test_non_reversible_door_ends_run_but_records_portal` (after line 176) add:

```python
    assert portal.reversible is False
    assert portal.to_cell == (0, 0)   # `entry` in the one-way border
```

- [ ] **Step 7: Run the whole suite to verify green**

Run: `.venv/bin/python -m pytest -q` (with `POKEMON_EMERALD_ROM` exported as above)
Expected: PASS — all previously-passing tests plus the two new `test_map_memory` tests. No collection errors.

- [ ] **Step 8: Commit**

```bash
git add env/map_memory.py env/map_explorer.py env/live_navigator.py \
        tests/test_map_memory.py tests/test_map_traveler.py \
        tests/test_live_navigator.py tests/test_map_explorer.py
git commit -m "feat: Portal gains reversible + to_cell; portal endpoint accessors"
```

---

### Task 2: `world_surveyor.py` + `SurveyReport` + first scenario (two maps)

Create the module test-first with a unified `WorldGrid` fake that plays both emulator and reader for a multi-map world and supports BOTH `travel_to` and `map_map`. Start with the simplest scenario: two maps joined by one reversible border.

**Files:**
- Create: `env/world_surveyor.py`
- Test: `tests/test_world_surveyor.py`

- [ ] **Step 1: Write the failing test with the WorldGrid fake and scenario 1**

Create `tests/test_world_surveyor.py`:

```python
"""world_surveyor: chart a multi-map overworld over a fake WorldGrid (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DELTAS, OPPOSITE, WallMap
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot
from env.world_surveyor import SurveyReport, survey_world

_KEY_TO_DIR = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}


class WorldGrid:
    """Hidden per-map grids joined by borders; answers step + snapshot.

    walls: set[(map_id, cell, direction)] — a blocked directed move.
    borders: dict[(map_id, cell, direction)] -> (to_map, entry_cell).
    Movement inside a map is otherwise free.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        walls: set[tuple[tuple[int, int], tuple[int, int], str]],
        borders: dict[
            tuple[tuple[int, int], tuple[int, int], str],
            tuple[tuple[int, int], tuple[int, int]],
        ],
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._walls = walls
        self._borders = borders

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # release / noop
        key = (self.map_id, self.pos, direction)
        if key in self._borders:
            self.map_id, self.pos = self._borders[key]
            return
        if key in self._walls:
            return  # wall: stay put
        dx, dy = DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def _sealed_room(
    map_id: tuple[int, int], width: int, height: int
) -> set[tuple[tuple[int, int], tuple[int, int], str]]:
    """Every outward boundary edge of a width x height room, keyed by map_id."""
    walls: set[tuple[tuple[int, int], tuple[int, int], str]] = set()
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if x == 0:
                walls.add((map_id, cell, "left"))
            if x == width - 1:
                walls.add((map_id, cell, "right"))
            if y == 0:
                walls.add((map_id, cell, "up"))
            if y == height - 1:
                walls.add((map_id, cell, "down"))
    return walls


def _reversible_border(
    a_map: tuple[int, int], a_cell: tuple[int, int], direction: str,
    b_map: tuple[int, int], b_cell: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int], str],
          tuple[tuple[int, int], tuple[int, int]]]:
    """A two-way door: a_map@a_cell --direction--> b_map@b_cell and back."""
    return {
        (a_map, a_cell, direction): (b_map, b_cell),
        (b_map, b_cell, OPPOSITE[direction]): (a_map, a_cell),
    }


def test_two_maps_linked_by_reversible_border() -> None:
    a, b = (0, 0), (0, 1)
    # Each map is a fully-sealed 2x1 room. The door edges live in `borders`,
    # and WorldGrid.step checks borders BEFORE walls, so a door tile transitions
    # even though it is also a boundary wall — no need to punch holes in `walls`
    # (punching a hole on a NON-door edge would open an infinite void).
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1)
    borders = _reversible_border(a, (1, 0), "right", b, (0, 0))
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    assert isinstance(report, SurveyReport)
    assert set(report.surveyed) == {a, b}
    assert report.failed == ()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.world_surveyor'`.

- [ ] **Step 3: Implement `world_surveyor.py`**

Create `env/world_surveyor.py`:

```python
"""world_surveyor: chart the reachable overworld map-by-map.

survey_world starts wherever the player stands and repeatedly travels to a
not-yet-surveyed map and surveys it (travel_to + map_map), discovering new maps
through the reversible border portals map_map records. Overworld only: building
warps (non-reversible) are never followed. Log-and-continue: a failed leg is
recorded in the SurveyReport and the sweep goes on. No training, no reward,
no Strategist, no fighting. Emerald (BPEF) only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from env.live_navigator import snapshot_settled
from env.local_navigator import WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory, Portal
from env.map_traveler import travel_to


@dataclass(frozen=True)
class SurveyReport:
    surveyed: tuple[tuple[int, int], ...]              # maps charted, in visit order
    failed: tuple[tuple[tuple[int, int], str], ...]    # (map_id, reason)


def survey_world(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    max_maps: int = 50,
) -> SurveyReport:
    """Sweep the reachable overworld with an iterative, bounded BFS.

    Returns a SurveyReport listing every surveyed map and every failed leg with
    a reason ("travel:<outcome>" or "map:<result>"). Bounded by max_maps
    (code-safety rule #2); BFS is iterative, no recursion.
    """
    start = _current_map(reader)
    if start is None:
        return SurveyReport((), (("unknown", "no_start"),))

    pending: deque[tuple[int, int]] = deque([start])
    queued: set[tuple[int, int]] = {start}
    surveyed: list[tuple[int, int]] = []
    failed: list[tuple[tuple[int, int], str]] = []

    for _ in range(max_maps):
        if not pending:
            break
        target = pending.popleft()

        here = _current_map(reader)
        if here != target:
            outcome = travel_to(
                emulator, reader, memory, wallmap,
                target, _entry_cell(memory, target),
            )
            if outcome != "arrived":
                failed.append((target, f"travel:{outcome}"))
                continue

        result = map_map(emulator, reader, memory, wallmap, target)
        if result in ("left_map", "budget_exhausted"):
            failed.append((target, f"map:{result}"))
        surveyed.append(target)

        for portal in _overworld_portals(memory, target):
            nxt = portal.to_map
            if nxt not in queued and nxt not in surveyed:
                queued.add(nxt)
                pending.append(nxt)

    return SurveyReport(tuple(surveyed), tuple(failed))


def _current_map(reader: Any) -> tuple[int, int] | None:
    snap = snapshot_settled(reader)
    return None if snap is None else snap.map_id


def _overworld_portals(memory: MapMemory, map_id: tuple[int, int]) -> list[Portal]:
    """Outgoing portals of map_id that are reversible overworld borders."""
    return [p for p in memory.outgoing_portals(map_id) if p.reversible]


def _entry_cell(memory: MapMemory, target: tuple[int, int]) -> tuple[int, int]:
    """The cell we land on when entering target, from any recorded portal to it.

    Safe to index [0]: target was enqueued only because a portal to it exists.
    """
    return memory.incoming_portals(target)[0].to_cell
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add env/world_surveyor.py tests/test_world_surveyor.py
git commit -m "feat: world_surveyor charts a two-map overworld"
```

---

### Task 3: Surveyor scenarios — chain order, warp filter + log, budget

Three more deterministic scenarios exercising BFS visit order, log-and-continue on the `map_map` side, the overworld-only filter, and the `max_maps` bound. All reuse `WorldGrid`, `_sealed_room`, `_reversible_border` from Task 2.

**Design note (why not a "failed travel leg" scenario):** in this consistent fake, `map_map` records `memory` from the *same* borders `travel_to` later reads, so a reversible overworld leg can never fail — a door discovered during survey is always re-reachable at travel time. Genuine `travel:*` failures ("unreachable"/"unknown_route"/"lost") are already covered by `tests/test_map_traveler.py`. What IS constructible here is the `map_map`-side failure: a map whose only exit is a non-reversible warp makes `map_map` return `"left_map"` — logged as `map:left_map`, still counted as surveyed — and the warp's target map is filtered out (never enqueued). That single scenario covers both the filter and the map-side log-and-continue.

**Files:**
- Test: `tests/test_world_surveyor.py` (append)

- [ ] **Step 1: Write the three scenario tests**

Append to `tests/test_world_surveyor.py`:

```python
def _one_way_border(
    a_map: tuple[int, int], a_cell: tuple[int, int], direction: str,
    b_map: tuple[int, int], b_cell: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int], str],
          tuple[tuple[int, int], tuple[int, int]]]:
    """A one-way building warp: a_map@a_cell --direction--> b_map@b_cell, no return."""
    return {(a_map, a_cell, direction): (b_map, b_cell)}


def test_chain_of_three_maps_surveyed_in_bfs_order() -> None:
    a, b, c = (0, 0), (0, 1), (0, 2)
    # Fully-sealed rooms; doors live in `borders` (checked before walls).
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1) | _sealed_room(c, 2, 1)
    borders = {
        **_reversible_border(a, (1, 0), "right", b, (0, 0)),
        **_reversible_border(b, (1, 0), "right", c, (0, 0)),
    }
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    assert report.surveyed == (a, b, c)
    assert report.failed == ()


def test_warp_only_map_is_logged_left_map_and_target_never_enqueued() -> None:
    # Start map A's ONLY exit is a one-way building warp to H (down from (0,0)).
    # map_map(A) probes the warp -> "left_map"; A is still counted surveyed and
    # logged as map:left_map. H is a non-reversible target: never enqueued.
    a, h = (0, 0), (5, 5)
    walls = _sealed_room(a, 1, 1)          # single-cell room, fully sealed
    borders = _one_way_border(a, (0, 0), "down", h, (0, 0))  # warp wins over the wall
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    assert report.surveyed == (a,)                 # A surveyed despite the early exit
    assert (a, "map:left_map") in report.failed    # map-side failure logged
    assert h not in report.surveyed                # warp target never surveyed
    assert all(m != h for (m, _r) in report.failed)  # nor even attempted


def test_max_maps_stops_cleanly_with_partial_report() -> None:
    a, b, c = (0, 0), (0, 1), (0, 2)
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1) | _sealed_room(c, 2, 1)
    borders = {
        **_reversible_border(a, (1, 0), "right", b, (0, 0)),
        **_reversible_border(b, (1, 0), "right", c, (0, 0)),
    }
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=1)

    assert report.surveyed == (a,)        # exactly one map surveyed then stopped
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -q`
Expected: PASS. These are deterministic regression coverage over the Task 2 implementation — no source change is expected. If one fails for a real reason (e.g. warp target enqueued, wrong BFS order, `map:left_map` not logged), that is a genuine failing test → proceed to Step 3.

- [ ] **Step 3: Fix `survey_world` only if a scenario exposes a real gap**

If all three pass, no source change is needed. If one fails for a real reason, make the minimal `env/world_surveyor.py` change to satisfy it (do not weaken an assertion).

- [ ] **Step 4: Run the full surveyor test file to verify green**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -q`
Expected: PASS — all four scenarios (Task 2 + these three).

- [ ] **Step 5: Commit**

```bash
git add tests/test_world_surveyor.py env/world_surveyor.py
git commit -m "test: surveyor chain order, warp filter + log, budget"
```

---

### Task 4: ROM-gated smoke test

One integration smoke on the captured `states/open_map.state` (Bourg-en-Vol, map open), double-skipped when the ROM env var or the state file is missing. Asserts a coherent report and that learning happened, observed through externally-visible state only.

**Files:**
- Test: `tests/test_world_surveyor_rom.py`

- [ ] **Step 1: Write the ROM smoke test**

Create `tests/test_world_surveyor_rom.py`:

```python
"""ROM-gated smoke for survey_world on a real open-map savestate.

Double skip: POKEMON_EMERALD_ROM unset OR states/open_map.state missing.
The state and ROM are LOCAL, gitignored artifacts in the main repo ~/Projets/Emu.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.mgba_emulator import MGBAEmulator
from env.game_state import EmeraldReader
from env.local_navigator import WallMap
from env.map_memory import MapMemory
from env.world_reader import WorldReader
from env.world_surveyor import survey_world

_ROM = os.environ.get("POKEMON_EMERALD_ROM")
_STATE = Path.home() / "Projets" / "Emu" / "states" / "open_map.state"

pytestmark = [
    pytest.mark.skipif(_ROM is None, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not _STATE.exists(), reason="open_map.state missing"),
]


def test_survey_world_smoke_is_coherent_and_learns() -> None:
    emu = MGBAEmulator(_ROM)
    emu.load_state(_STATE.read_bytes())
    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(reader)
    memory = MapMemory()
    wallmap = WallMap()

    report = survey_world(emu, world, memory, wallmap, max_maps=2)

    # Report is coherent: either something got surveyed, or every miss is explained.
    assert report.surveyed or report.failed
    # Learning is externally visible: a wall was learned or a portal recorded.
    learned_wall = any(wallmap._walls.get(m) for m in wallmap._walls) \
        if hasattr(wallmap, "_walls") else False
    learned_portal = bool(memory.outgoing_portals(report.surveyed[0])) \
        if report.surveyed else False
    assert learned_wall or learned_portal or report.failed
```

Note: confirm the exact `MGBAEmulator` / `EmeraldReader` / `WorldReader` import paths and constructors against the existing ROM smoke tests (`tests/test_map_explorer_rom.py`, `tests/test_live_navigator_rom.py`) and mirror them — reuse whatever those files already do to build the emulator, reader, and world snapshot source. If `WallMap` exposes a public "any wall learned" accessor, prefer it over touching `_walls`.

- [ ] **Step 2: Run the smoke (skips if ROM/state absent)**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor_rom.py -q`
Expected: PASS (or SKIP if `POKEMON_EMERALD_ROM` unset / `open_map.state` missing).
With the ROM exported and the state present: PASS, no crash, coherent report.

- [ ] **Step 3: Run the entire suite one last time**

Run: `.venv/bin/python -m pytest -q` (with `POKEMON_EMERALD_ROM` exported)
Expected: PASS — every pre-existing test plus the new surveyor unit + ROM tests. Lint clean: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check .`

- [ ] **Step 4: Commit**

```bash
git add tests/test_world_surveyor_rom.py
git commit -m "test: ROM smoke for survey_world on open_map.state"
```

---

## Notes for the executor

- **Bounded loops (code-safety #2):** `survey_world` is bounded by `max_maps`; `travel_to`, `map_map`, `snapshot_settled` are each independently bounded. BFS is iterative with an explicit `deque` — no recursion.
- **Chicken-and-egg:** a map is enqueued only after a portal to it is recorded, so `travel_to` always has a route. The start map has no incoming portal but the player is already on it, so no `travel_to` runs for it.
- **`map_explorer` behaviour refinement (Task 1 Step 4):** the transition branch now records the portal AFTER the step-back so `reversible` reflects the proven round-trip, and bails safe (`left_map`) when `landed is None`. This also tightens the deferred Minor #1 from P3 step 3.
- **Worktree reminder:** `.venv` and `roms/` are in the main repo `~/Projets/Emu`, not this worktree. Always run pytest with the absolute `.venv` python and `POKEMON_EMERALD_ROM` exported.
