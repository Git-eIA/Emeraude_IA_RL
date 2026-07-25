# P3 Step 3 — Mapping Mode (`map_map`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `map_map`, a frontier-based survey that wanders one Emerald map to exhaustion — learning every wall into a `WallMap` and recording every door as a portal in `MapMemory` — plus a tool that auto-captures an open-map savestate so the ROM smoke becomes load-bearing.

**Architecture:** Promote two internal `live_navigator` helpers to public names (`probe_step`, `snapshot_settled`) so both `navigate_to` and the new `map_map` share the GBA button-debounce choreography without duplication. `map_map` (new `env/map_explorer.py`) keeps a `reached`/`tried` frontier, repositions only over already-walked cells (BFS over `reached`, never `plan_path`) so the only step into the unknown is a deliberate probe, then classifies each probe as moved/blocked/transition. Doors are recorded and stepped back through the reversible border; a non-reversible warp ends the run as `"left_map"`.

**Tech Stack:** Python 3.12, pytest (fake-world unit tests + one ROM-gated smoke), stable-baselines3 PPO (savestate capture tool only), libmgba-py.

---

## File Structure

- **`env/live_navigator.py`** (modify): rename `_press_until_moved` → `probe_step` (public) and `_snapshot_settled` → `snapshot_settled` (public); update the two internal call sites in `navigate_to`. No behavior change.
- **`env/map_explorer.py`** (new): `map_map` + `_nearest_frontier` + `_follow_route`. The single-map frontier survey.
- **`tools/capture_open_map.py`** (new): drive the trained Explorer from `states/initial.state` until it stands on an overworld map, then save `states/open_map.state`. One-time artifact generation.
- **`tests/test_map_explorer.py`** (new): unit tests with an `ExploreWorld` fake (plays emulator + reader).
- **`tests/test_map_explorer_rom.py`** (new): ROM-gated smoke on `states/open_map.state`.
- **`tests/test_live_navigator.py`** (modify): tiny regression proving the public `probe_step`/`snapshot_settled` names exist and behave.

Test-run command (from the worktree, because `.venv`/`roms`/`states` live in the main repo):

```
cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```

---

### Task 1: Promote shared primitives to public names in `live_navigator`

**Files:**
- Modify: `env/live_navigator.py`
- Test: `tests/test_live_navigator.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_live_navigator.py`)

```python
def test_probe_step_and_snapshot_settled_are_public():
    """map_map imports these two by their public names; guard the rename."""
    from env.live_navigator import probe_step, snapshot_settled

    world = FakeWorld(walls=set())  # open one-cell world, player at (0, 0)
    before = snapshot_settled(world)
    assert before is not None
    assert before.pos == (0, 0)

    outcome = probe_step(world, world, before, "right")
    assert outcome == "moved"
    assert snapshot_settled(world).pos == (1, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py::test_probe_step_and_snapshot_settled_are_public -q`
Expected: FAIL with `ImportError: cannot import name 'probe_step'`.

- [ ] **Step 3: Rename the two helpers and their call sites**

In `env/live_navigator.py`, rename the function definitions:
- `def _press_until_moved(` → `def probe_step(`
- `def _snapshot_settled(` → `def snapshot_settled(`

Update the docstrings to drop the leading-underscore framing (optional cosmetic), then fix the two call sites inside `navigate_to`:

```python
        outcome = probe_step(emulator, reader, before, direction)
        if outcome == "transition":
            if memory is not None:
                landed = snapshot_settled(reader)
```

and inside `probe_step` itself, its internal call:

```python
        after = snapshot_settled(reader)
```

- [ ] **Step 4: Run the test to verify it passes, then the whole live_navigator suite**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (all previous tests + the new one). No behavior change to `navigate_to`.

- [ ] **Step 5: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p3-map-explorer && git add env/live_navigator.py tests/test_live_navigator.py && git commit -m "refactor: promote probe_step/snapshot_settled to public names"
```

---

### Task 2: `_nearest_frontier` + `map_map` sealed-room complete survey

**Files:**
- Create: `env/map_explorer.py`
- Test: `tests/test_map_explorer.py`

- [ ] **Step 1: Write the `ExploreWorld` fake + the sealed-room test** (create `tests/test_map_explorer.py`)

```python
"""Unit tests for map_map with an ExploreWorld fake (plays emulator + reader).

ExploreWorld models one hidden grid keyed by map_id, with `walls` (blocked
directed edges the survey must learn by bumping) and optional `borders`
(reversible/non-reversible map crossings). It exposes both the emulator API
(`step(keys, frames)` decodes the d-pad bit and moves on the hidden grid) and
the reader API (`snapshot()` -> WorldSnapshot). No ROM, no emulator.
"""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DELTAS, OPPOSITE, WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot

_KEY_TO_DIR = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}


class ExploreWorld:
    """Hidden grid that answers both emulator.step and reader.snapshot."""

    def __init__(
        self,
        map_id: tuple[int, int],
        start: tuple[int, int],
        walls: set[tuple[tuple[int, int], str]],
        borders: dict[
            tuple[tuple[int, int], str], tuple[tuple[int, int], tuple[int, int]]
        ]
        | None = None,
    ) -> None:
        self.map_id = map_id
        self.pos = start
        self._walls = walls
        self._borders = borders or {}
        self.presses = 0

    # --- emulator side -------------------------------------------------
    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # release frame: no movement
        self.presses += 1
        key = (self.map_id, self.pos, direction)
        if key in self._borders:
            to_map, entry = self._borders[key]
            self.map_id = to_map
            self.pos = entry
            return
        if (self.pos, direction) in self._walls:
            return  # wall: stay put
        dx, dy = DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    # --- reader side ---------------------------------------------------
    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def _sealed_room_walls(
    width: int, height: int
) -> set[tuple[tuple[int, int], str]]:
    """All outward edges on the boundary of a width x height room at origin."""
    walls: set[tuple[tuple[int, int], str]] = set()
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if x == 0:
                walls.add((cell, "left"))
            if x == width - 1:
                walls.add((cell, "right"))
            if y == 0:
                walls.add((cell, "up"))
            if y == height - 1:
                walls.add((cell, "down"))
    return walls


def test_sealed_room_complete_survey():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)  # cells (0,0),(1,0),(0,1),(1,1)
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "complete"
    # every boundary wall was learned (block is bidirectional, so check the
    # outward edges we planted)
    for (cell, direction) in walls:
        assert wallmap.is_blocked(target, cell, direction)
    # no door was invented on a wall-only room
    assert memory.edges() == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.map_explorer'`.

- [ ] **Step 3: Write `env/map_explorer.py`**

```python
"""map_explorer: survey one Emerald map to exhaustion by frontier search.

map_map stands on a cell (now `reached`), looks at its four directions, and
treats any direction that is neither already `tried` nor a known wall as a
`frontier`. It repositions to the nearest frontier cell over KNOWN-walkable
cells only (BFS over `reached`, never plan_path — plan_path's optimistic grid
could route through an unknown door), then probes the one unknown edge:
moved -> new walkable cell, blocked -> a wall (recorded), transition -> a door
(recorded as a portal, then stepped back through the reversible border).
Reuses P2 (WallMap, DELTAS, OPPOSITE, DIRECTIONS) and P3 step-1 primitives
(probe_step, snapshot_settled, timing constants). Emerald (BPEF) only.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from env.live_navigator import RELEASE_FRAMES, probe_step, snapshot_settled
from env.local_navigator import DELTAS, DIRECTIONS, OPPOSITE, WallMap
from env.map_memory import MapMemory


def map_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    target_map: tuple[int, int],
    max_steps: int = 2000,
) -> str:
    """Survey `target_map` by frontier search: learn every wall and record
    every door as a portal. Known-walkable repositioning only.

    Returns:
      "complete"          — frontier exhausted; the map is fully known
      "budget_exhausted"  — hit max_steps before the frontier emptied
      "left_map"          — crossed a non-reversible door and could not return
    """
    reached: set[tuple[int, int]] = set()
    tried: set[tuple[tuple[int, int], str]] = set()

    for _ in range(max_steps):
        here = snapshot_settled(reader)
        if here is None:
            emulator.step(0, RELEASE_FRAMES)  # relocating; idle a beat and retry
            continue
        if here.map_id != target_map:
            return "left_map"
        reached.add(here.pos)

        plan = _nearest_frontier(reached, tried, wallmap, target_map, here.pos)
        if plan is None:
            return "complete"
        route, cell, direction = plan

        if not _follow_route(emulator, reader, route):
            continue  # world surprised us; next snapshot re-grounds the survey

        before = snapshot_settled(reader)
        if before is None or before.pos != cell:
            continue  # repositioning landed off the frontier cell; re-loop

        outcome = probe_step(emulator, reader, before, direction)
        tried.add((cell, direction))
        if outcome == "moved":
            # mirror the edge so the neighbour never re-probes back at us
            dx, dy = DELTAS[direction]
            neighbour = (cell[0] + dx, cell[1] + dy)
            tried.add((neighbour, OPPOSITE[direction]))
        elif outcome == "blocked":
            wallmap.block(target_map, cell, direction)
        elif outcome == "transition":
            landed = snapshot_settled(reader)
            if landed is not None:
                memory.record_portal(target_map, cell, direction, landed.map_id)
                probe_step(emulator, reader, landed, OPPOSITE[direction])
            returned = snapshot_settled(reader)
            if returned is None or returned.map_id != target_map:
                return "left_map"

    return "budget_exhausted"


def _follow_route(emulator: Any, reader: Any, route: list[str]) -> bool:
    """Press each known-walkable direction; True if every press moved us."""
    for direction in route:
        before = snapshot_settled(reader)
        if before is None:
            return False
        if probe_step(emulator, reader, before, direction) != "moved":
            return False
    return True


def _nearest_frontier(
    reached: set[tuple[int, int]],
    tried: set[tuple[tuple[int, int], str]],
    wallmap: WallMap,
    target_map: tuple[int, int],
    start: tuple[int, int],
) -> tuple[list[str], tuple[int, int], str] | None:
    """BFS over reached cells from `start`; return (route, cell, direction) for
    the nearest reached cell that still has an unexplored, non-walled direction,
    or None if the frontier is empty. Ties break by DIRECTIONS order."""
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    seen: set[tuple[int, int]] = {start}
    while queue:
        cell, route = queue.popleft()
        for direction in DIRECTIONS:
            if (cell, direction) in tried:
                continue
            if wallmap.is_blocked(target_map, cell, direction):
                continue
            return route, cell, direction
        for direction in DIRECTIONS:
            if wallmap.is_blocked(target_map, cell, direction):
                continue
            dx, dy = DELTAS[direction]
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in reached and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, route + [direction]))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p3-map-explorer && git add env/map_explorer.py tests/test_map_explorer.py && git commit -m "feat: map_map frontier survey completes a sealed room"
```

---

### Task 3: Door discovered, recorded, stepped back — survey continues

**Files:**
- Modify: `tests/test_map_explorer.py`

- [ ] **Step 1: Write the reversible-door test + a no-re-probe guard** (append to `tests/test_map_explorer.py`)

```python
def _reversible_border(
    from_map: tuple[int, int],
    from_cell: tuple[int, int],
    direction: str,
    to_map: tuple[int, int],
    entry: tuple[int, int],
) -> dict[
    tuple[tuple[int, int], str], tuple[tuple[int, int], tuple[int, int]]
]:
    """A two-way border: crossing `direction` lands on to_map@entry, and the
    opposite press from entry returns to from_map@from_cell."""
    return {
        (from_map, from_cell, direction): (to_map, entry),
        (to_map, entry, OPPOSITE[direction]): (from_map, from_cell),
    }


def test_reversible_door_recorded_and_survey_continues():
    target = (3, 3)
    other = (7, 7)
    # 2x1 room: cells (0,0),(1,0). Seal every boundary EXCEPT (1,0)->right,
    # which is a reversible door to `other`.
    walls = _sealed_room_walls(2, 1)
    walls.discard(((1, 0), "right"))
    borders = _reversible_border(target, (1, 0), "right", other, (0, 0))
    world = ExploreWorld(target, start=(0, 0), walls=walls, borders=borders)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "complete"
    portal = memory.portal(target, other)
    assert portal is not None
    assert portal.from_cell == (1, 0)
    assert portal.direction == "right"
    assert portal.to_map == other
    # we returned and finished mapping the room's remaining walls
    assert wallmap.is_blocked(target, (0, 0), "left")


def test_no_edge_is_reprobed():
    """After 'complete', every (cell, direction) was tried at most once, so the
    press count cannot exceed the total edge count of the reached region."""
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=500)

    assert result == "complete"
    # 4 cells x 4 directions = 16 directed edges is the hard ceiling on probes;
    # repositioning presses are bounded by the same walked region. A runaway
    # re-probe loop would blow past this immediately.
    assert world.presses <= 64
```

- [ ] **Step 2: Run test to verify it passes** (Task 2's `map_map` already implements door handling)

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer.py -q`
Expected: PASS. If `test_reversible_door_recorded_and_survey_continues` fails, the step-back or portal recording in `map_map`'s `transition` branch is wrong — fix there, not in the test.

- [ ] **Step 3: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p3-map-explorer && git add tests/test_map_explorer.py && git commit -m "test: reversible door recorded, stepped back, survey completes"
```

---

### Task 4: Non-reversible door ends the run; budget exhaustion

**Files:**
- Modify: `tests/test_map_explorer.py`

- [ ] **Step 1: Write the non-reversible + budget tests** (append to `tests/test_map_explorer.py`)

```python
def test_non_reversible_door_ends_run_but_records_portal():
    target = (3, 3)
    other = (7, 7)
    walls = _sealed_room_walls(2, 1)
    walls.discard(((1, 0), "right"))
    # one-way warp: crossing right lands on `other`, but the opposite press
    # from the entry cell does NOT return (no reverse border) — a building warp.
    borders = {(target, (1, 0), "right"): (other, (0, 0))}
    world = ExploreWorld(target, start=(0, 0), walls=walls, borders=borders)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "left_map"
    # the portal is still recorded before the run ends
    portal = memory.portal(target, other)
    assert portal is not None
    assert portal.from_cell == (1, 0)
    assert portal.direction == "right"


def test_budget_exhausted_on_large_room_with_tiny_budget():
    target = (3, 3)
    walls = _sealed_room_walls(6, 6)  # 36 cells, far more than the budget
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=5)

    assert result == "budget_exhausted"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer.py -q`
Expected: PASS. If `test_non_reversible_door...` fails, the `returned.map_id != target_map` branch in `map_map` is not returning `"left_map"` — fix there.

- [ ] **Step 3: Run the full non-ROM suite to catch regressions**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q -k "not rom"`
Expected: PASS (all prior tests + the new map_explorer tests).

- [ ] **Step 4: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p3-map-explorer && git add tests/test_map_explorer.py && git commit -m "test: non-reversible door -> left_map, budget exhaustion"
```

---

### Task 5: `capture_open_map.py` tool + ROM smoke test

**Files:**
- Create: `tools/capture_open_map.py`
- Test: `tests/test_map_explorer_rom.py`

- [ ] **Step 1: Write `tools/capture_open_map.py`**

Model the PPO + `PokemonEmeraldEnv` + `save_state()` pattern on `tools/auto_capture_battles.py`. Drive the trained Explorer until `WorldReader.snapshot()` reports a map different from the truck interior, then save.

```python
"""Capture an open-map savestate by letting the trained Explorer walk out.

Loads the Explorer PPO policy, plays from states/initial.state, and the moment
WorldReader reports the player has left the truck interior onto a different map
it writes states/open_map.state. One-time artifact generation: run once locally
where checkpoints/ppo_emerald_final.zip exists; the output is committed so the
map-explorer ROM smoke does not need the checkpoint afterwards.

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_open_map.py \
      --max-steps 4000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/open_map.state")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), initial, max_steps=10_000_000)
    reader = WorldReader(EmeraldReader(env.emulator.read_bytes))
    model = PPO.load(args.model, device="cpu")

    obs, _ = env.reset()
    start = reader.snapshot()
    start_map = start.map_id if start is not None else None
    print(f"start map: {start_map}", flush=True)

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, _ = env.step(int(action))
        snap = reader.snapshot()
        if snap is not None and snap.map_id != start_map:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            print(
                f"OPEN MAP at step {step}: map {snap.map_id} pos {snap.pos} "
                f"-> {OUT_PATH}",
                flush=True,
            )
            return

    print(
        f"no open map reached in {args.max_steps} steps "
        f"(still on {start_map}); try more steps or a different seed",
        flush=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tool once to produce `states/open_map.state`**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python tools/capture_open_map.py --max-steps 6000`
Expected: prints `OPEN MAP at step N: map (g, n) pos (x, y) -> states/open_map.state`. Note the printed `map (g, n)` — that open map id is what the ROM smoke targets. The file is written under the main repo's `states/` (shared with the worktree via absolute path resolution below). If it prints "no open map reached", re-run with a larger `--max-steps`.

- [ ] **Step 3: Write the ROM smoke test** (create `tests/test_map_explorer_rom.py`)

```python
"""ROM-gated smoke for map_map on a real open map.

Loads states/open_map.state (produced by tools/capture_open_map.py), runs a
short survey, and asserts a load-bearing result: a legal outcome without
crashing AND that learning actually happened, checked only through externally
visible state (reached is private to map_map): the player moved, or a wall was
learned, or a portal was recorded.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.local_navigator import DIRECTIONS, WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory
from env.world_reader import WorldReader

POKEMON_EMERALD_ROM = os.environ.get("POKEMON_EMERALD_ROM")
# states/ lives in the main repo, not the worktree — resolve absolutely.
_STATE = Path.home() / "Projets" / "Emu" / "states" / "open_map.state"


@pytest.mark.skipif(not POKEMON_EMERALD_ROM, reason="requires POKEMON_EMERALD_ROM")
@pytest.mark.skipif(not _STATE.exists(), reason="requires states/open_map.state")
def test_map_map_learns_something_on_a_real_open_map():
    emulator = GbaEmulator(POKEMON_EMERALD_ROM)
    emulator.load_state(_STATE.read_bytes())
    reader = WorldReader(EmeraldReader(emulator.read_bytes))

    start = reader.snapshot()
    assert start is not None, "open_map.state should sit on a readable map"
    target_map = start.map_id
    start_cell = start.pos

    memory = MapMemory()
    wallmap = WallMap()
    result = map_map(emulator, reader, memory, wallmap, target_map, max_steps=40)

    assert result in ("complete", "budget_exhausted", "left_map")

    moved = reader.snapshot() is not None and reader.snapshot().pos != start_cell
    learned_wall = any(
        wallmap.is_blocked(target_map, start_cell, d) for d in DIRECTIONS
    )
    recorded_portal = len(memory.edges()) > 0
    assert moved or learned_wall or recorded_portal, (
        "survey should move, learn a wall, or record a portal on a real map"
    )
```

- [ ] **Step 4: Run the ROM smoke**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_explorer_rom.py -q`
Expected: PASS (not skipped, since the ROM env var and `states/open_map.state` both exist).

- [ ] **Step 5: Run the entire suite (unit + ROM) to confirm green end-to-end**

Run: `cd /Users/_eloi/Projets/Emu-p3-map-explorer && PYTHONPATH=/Users/_eloi/Projets/Emu-p3-map-explorer POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: all pass (new: 1 live_navigator rename guard + 5 map_explorer unit + 1 ROM smoke).

- [ ] **Step 6: Commit** (the `open_map.state` artifact lives in the main repo's `states/`, so commit only code here; the state file is captured/committed separately in the main repo when merging)

```bash
cd /Users/_eloi/Projets/Emu-p3-map-explorer && git add tools/capture_open_map.py tests/test_map_explorer_rom.py && git commit -m "feat: capture_open_map tool + load-bearing map_map ROM smoke"
```

---

## Self-Review

**Spec coverage:**
- `map_map` single-map frontier survey (walls + portals) → Tasks 2–4. ✓
- Reuse P2 + step-1 primitives (shared-primitive extraction) → Task 1. ✓
- `tools/capture_open_map.py` auto-capture → Task 5. ✓
- Unit tests: sealed-room complete, no-re-probe, reversible door, non-reversible → left_map, budget → Tasks 2–4. ✓
- ROM smoke, load-bearing via externally-visible state → Task 5. ✓

**Placeholder scan:** none — every step has full code or an exact command.

**Type consistency:** `map_map(emulator, reader, memory, wallmap, target_map, max_steps)` used identically in tests and impl. `_nearest_frontier` returns `(route, cell, direction) | None` and `map_map` unpacks `route, cell, direction`. `probe_step`/`snapshot_settled` public names introduced in Task 1 and imported in Task 2. `memory.portal(...)` / `memory.record_portal(...)` / `memory.edges()` match the P3-step-2 `MapMemory` API. `WallMap.block`/`is_blocked` and `DELTAS`/`OPPOSITE`/`DIRECTIONS` match `local_navigator`.

**Bounded loops (code-safety #2):** `map_map` main loop bounded by `max_steps`; `_nearest_frontier` BFS over the finite `reached` set; `_follow_route` presses `len(route)` ≤ `|reached|` times; `probe_step` bounded by `TURN_RETRIES`; `snapshot_settled` by `SETTLE_TRIES`. No recursion, no `while True`.
