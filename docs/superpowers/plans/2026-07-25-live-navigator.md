# P3 (step 1) — Live Intra-Map Navigator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the real GBA emulator so the player walks to a target cell on its current map, learning walls by collision and replanning on each bump.

**Architecture:** One new pure control module `env/live_navigator.py` wiring P1 `WorldReader.snapshot()` + P2 `plan_path`/`resolve_move`/`WallMap` to `emulator.step(keys, frames)`. Perceive → plan → press → classify → learn → replan loop. No Gym env, no training, no reward.

**Tech Stack:** Python 3.12, pytest, mGBA via `emulator/gba.py`. Reuses `env/world_reader.py`, `env/local_navigator.py`, `emulator/buttons.py` unchanged.

**Working directory:** `/Users/_eloi/Projets/Emu-p3-live-navigator` (worktree, branch `feat/p3-live-navigator`). The `.venv`, `roms/`, and `states/` live in the main repo `/Users/_eloi/Projets/Emu`; run tests with the main-repo venv + `PYTHONPATH=$PWD`.

**Unit-test command (no ROM):**
```
PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q
```
**ROM smoke-test command:**
```
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator_rom.py -q
```

---

### Task 1: navigate_to happy path (arrive + timeout)

**Files:**
- Create: `env/live_navigator.py`
- Create: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_navigator.py` with the shared `FakeWorld` helper and the first two tests. `FakeWorld` plays both the emulator (`.step`) and the reader (`.snapshot`) roles over a hidden grid.

```python
"""live_navigator: control-loop tests over a fake grid world (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.live_navigator import navigate_to
from env.local_navigator import WallMap
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


class FakeWorld:
    """A hidden grid the navigator must traverse blind.

    Acts as emulator (`step`) and reader (`snapshot`). `walls` is a set of
    (cell, direction) edges that block movement. `turn_first` models Emerald's
    turn-in-place: the first press in a new direction only rotates. `map_flips`
    are cells that, once entered, change map_id (a map transition). `none_frames`
    emits that many None snapshots first (SaveBlock relocation).
    """

    def __init__(
        self,
        start: tuple[int, int],
        walls: set[tuple[tuple[int, int], str]] | None = None,
        map_id: tuple[int, int] = (0, 0),
        turn_first: bool = False,
        map_flips: set[tuple[int, int]] | None = None,
        none_frames: int = 0,
    ) -> None:
        self.pos = start
        self.map_id = map_id
        self._walls = set(walls or ())
        self._turn_first = turn_first
        self._facing: str | None = None
        self._map_flips = set(map_flips or ())
        self._none_frames = none_frames
        self.presses = 0

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        self.presses += 1
        if self._turn_first and self._facing != direction:
            self._facing = direction
            return  # first press only turns
        self._facing = direction
        if (self.pos, direction) in self._walls:
            return  # wall: no move
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.pos in self._map_flips:
            self.map_id = (self.map_id[0], self.map_id[1] + 1)

    def snapshot(self) -> WorldSnapshot | None:
        if self._none_frames > 0:
            self._none_frames -= 1
            return None
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def test_walks_straight_corridor_to_target() -> None:
    world = FakeWorld(start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(3, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (3, 0)
    assert world.presses == 3


def test_times_out_when_budget_too_small() -> None:
    world = FakeWorld(start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(10, 0), max_steps=3)
    assert result == "timeout"
    assert world.pos != (10, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.live_navigator'`.

- [ ] **Step 3: Write the minimal implementation**

Create `env/live_navigator.py`:

```python
"""live_navigator: drive the real emulator to a target cell on the current map.

First live loop of the Explorer: read where the player is (P1 WorldReader), plan
a path over the walls learned so far (P2 plan_path), press a d-pad key, and
repeat. Later tasks add wall-learning, turn/wall disambiguation, and the
unreachable / left_map / None-tolerance branches. No training, no reward.
Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from emulator import buttons
from env.local_navigator import WallMap, plan_path

_DIRECTION_KEYS: dict[str, int] = {
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
}

STEP_FRAMES = 24      # hold a d-pad key ~0.4 s: one walking tile (matches env FRAMES_PER_ACTION)
RELEASE_FRAMES = 8    # idle after each press so the GBA doesn't fuse consecutive presses


def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
) -> str:
    """Walk the player to `target`. Returns 'arrived' or 'timeout' (for now)."""
    for _ in range(max_steps):
        before = reader.snapshot()
        if before.pos == target:
            return "arrived"
        direction = plan_path(wallmap, before.map_id, before.pos, target)[0]
        _press(emulator, direction)
    return "timeout"


def _press(emulator: Any, direction: str) -> None:
    emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
    emulator.step(0, RELEASE_FRAMES)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "feat: live navigator happy-path loop (arrive/timeout)"
```

---

### Task 2: learn a wall on collision and reroute

**Files:**
- Modify: `env/live_navigator.py`
- Modify: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_navigator.py`:

```python
def test_records_wall_and_reroutes() -> None:
    # Stepping right from (0,0) is walled; a down/right/up detour reaches (1,0).
    world = FakeWorld(start=(0, 0), walls={((0, 0), "right")})
    wallmap = WallMap()
    result = navigate_to(world, world, wallmap, target=(1, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (1, 0)
    assert wallmap.is_blocked((0, 0), (0, 0), "right")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py::test_records_wall_and_reroutes -q`
Expected: FAIL — the loop presses right forever against the wall and times out (or never records the wall), so `result != "arrived"`.

- [ ] **Step 3: Update the implementation**

Rewrite `env/live_navigator.py` to classify each press and record a wall on `blocked`. Add the `resolve_move` import; replace `_press` with `_press_until_moved`:

```python
"""live_navigator: drive the real emulator to a target cell on the current map.

Read where the player is (P1 WorldReader), plan a path over walls learned so far
(P2 plan_path), press a d-pad key, classify the result (moved / blocked /
transition), and record any wall it bumps so the next plan routes around it
(replan-on-bump). No training, no reward. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from emulator import buttons
from env.local_navigator import WallMap, plan_path, resolve_move

_DIRECTION_KEYS: dict[str, int] = {
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
}

STEP_FRAMES = 24      # hold a d-pad key ~0.4 s: one walking tile (matches env FRAMES_PER_ACTION)
RELEASE_FRAMES = 8    # idle after each press so the GBA doesn't fuse consecutive presses


def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
) -> str:
    """Walk the player to `target`. Returns 'arrived' or 'timeout' (for now)."""
    for _ in range(max_steps):
        before = reader.snapshot()
        if before.pos == target:
            return "arrived"
        direction = plan_path(wallmap, before.map_id, before.pos, target)[0]
        outcome = _press_until_moved(emulator, reader, before, direction)
        if outcome == "blocked":
            wallmap.block(before.map_id, before.pos, direction)
    return "timeout"


def _press_until_moved(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
    emulator.step(0, RELEASE_FRAMES)
    after = reader.snapshot()
    return resolve_move(before, after)
```

- [ ] **Step 4: Run the whole file to verify all pass**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "feat: live navigator learns walls and reroutes on bump"
```

---

### Task 3: tell a turn from a wall (retry the press)

**Files:**
- Modify: `env/live_navigator.py`
- Modify: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_navigator.py`:

```python
def test_first_press_turns_without_recording_wall() -> None:
    # In Emerald the first press only rotates the character; it must not be read
    # as a wall.
    world = FakeWorld(start=(0, 0), turn_first=True)
    wallmap = WallMap()
    result = navigate_to(world, world, wallmap, target=(2, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (2, 0)
    assert not wallmap.is_blocked((0, 0), (0, 0), "right")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py::test_first_press_turns_without_recording_wall -q`
Expected: FAIL — the single press reads the turn as `blocked`, records a false wall, and `wallmap.is_blocked(...)` is True (assert fails).

- [ ] **Step 3: Update the implementation**

In `env/live_navigator.py`, add the `TURN_RETRIES` and `SETTLE_TRIES` constants and replace `_press_until_moved` with the retry loop plus a `_snapshot_settled` helper.

Add after `RELEASE_FRAMES = 8`:

```python
TURN_RETRIES = 2      # a first press may only turn the character; retry to tell turn from wall
SETTLE_TRIES = 4      # re-read snapshot this many times to skip SaveBlock None frames
```

Replace the `_press_until_moved` function with:

```python
def _press_until_moved(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    """Press `direction`, retrying so a first-press turn isn't mistaken for a wall."""
    outcome = "blocked"
    for _ in range(TURN_RETRIES):
        emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
        emulator.step(0, RELEASE_FRAMES)
        after = _snapshot_settled(reader)
        if after is None:
            return "blocked"
        outcome = resolve_move(before, after)
        if outcome != "blocked":
            return outcome
    return outcome


def _snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap
```

- [ ] **Step 4: Run the whole file to verify all pass**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "feat: live navigator retries press to distinguish turn from wall"
```

---

### Task 4: unreachable, left_map, and None-tolerance branches

**Files:**
- Modify: `env/live_navigator.py`
- Modify: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_navigator.py`:

```python
def test_unreachable_when_goal_is_sealed_off() -> None:
    walls = {((0, 0), d) for d in ("up", "down", "left", "right")}
    world = FakeWorld(start=(0, 0), walls=walls)
    result = navigate_to(world, world, WallMap(), target=(5, 5), max_steps=50)
    assert result == "unreachable"


def test_left_map_when_stepping_onto_a_transition_cell() -> None:
    world = FakeWorld(start=(0, 0), map_flips={(0, 1)})
    result = navigate_to(world, world, WallMap(), target=(0, 3), max_steps=50)
    assert result == "left_map"


def test_tolerates_none_snapshots_at_loop_top() -> None:
    world = FakeWorld(start=(0, 0), none_frames=2)
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (2, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: FAIL — `test_unreachable...` times out instead of returning "unreachable"; `test_left_map...` returns "timeout"/"arrived" not "left_map"; `test_tolerates_none...` raises `AttributeError: 'NoneType' object has no attribute 'pos'`.

- [ ] **Step 3: Update the implementation**

Rewrite `navigate_to` in `env/live_navigator.py` to its final form: tolerate a `None` snapshot at the loop top, return `"unreachable"` when `plan_path` gives `None`, and return `"left_map"` on a `transition`. The helpers `_press_until_moved` and `_snapshot_settled` from Task 3 stay unchanged.

```python
def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
) -> str:
    """Walk the player to `target` on its current map.

    Returns 'arrived' | 'unreachable' | 'left_map' | 'timeout'.
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
            return "left_map"
        if outcome == "blocked":
            wallmap.block(before.map_id, before.pos, direction)
    return "timeout"
```

- [ ] **Step 4: Run the whole file to verify all pass**

Run: `PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Lint and commit**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/live_navigator.py tests/test_live_navigator.py`
Expected: no errors.

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "feat: live navigator unreachable/left_map/None-tolerance branches"
```

---

### Task 5: ROM smoke test (gated)

**Files:**
- Create: `tests/test_live_navigator_rom.py`

- [ ] **Step 1: Write the test**

Create `tests/test_live_navigator_rom.py`. It runs only when `POKEMON_EMERALD_ROM` is set (same gating as the other ROM tests) and proves the loop is wired to the real emulator: it reads the live position, arrives instantly when already on target, and returns a known outcome (without crashing) when told to walk toward a nearby same-map cell.

```python
"""ROM-gated smoke test: navigate_to wired to the real mGBA emulator."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.live_navigator import navigate_to
from env.local_navigator import WallMap
from env.world_reader import WorldReader

ROM = os.environ.get("POKEMON_EMERALD_ROM")


@pytest.mark.skipif(not ROM, reason="requires POKEMON_EMERALD_ROM")
def test_navigate_to_wires_to_real_emulator() -> None:
    emu = GbaEmulator(ROM)
    emu.load_state(Path("states/initial.state").read_bytes())
    reader = WorldReader(EmeraldReader(emu.read_bytes))

    start = reader.snapshot()
    assert start is not None  # live position read works

    # Already on target -> arrives immediately.
    assert navigate_to(emu, reader, WallMap(), target=start.pos, max_steps=5) == "arrived"

    # Nearby same-map target: the loop runs live and returns a known outcome.
    target = (start.pos[0], start.pos[1] + 2)
    result = navigate_to(emu, reader, WallMap(), target=target, max_steps=40)
    assert result in {"arrived", "unreachable", "left_map", "timeout"}
```

- [ ] **Step 2: Run the smoke test**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator_rom.py -q`
Expected: PASS (1 passed). If the constants `STEP_FRAMES` / `TURN_RETRIES` / `RELEASE_FRAMES` need tuning for a clean one-tile step, adjust them in `env/live_navigator.py` and re-run.

- [ ] **Step 3: Run the full suite to confirm no regressions**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: all prior tests still pass plus the 8 new ones (7 unit + 1 ROM).

- [ ] **Step 4: Commit**

```bash
git add tests/test_live_navigator_rom.py
git commit -m "test: ROM smoke test for live navigator wiring"
```
