# Phase 2 Return Descent (B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Physically return the player from route_101 to Birch's lab so the merged Phase 2 story campaign can deliver the Pokédex + 5 Poké Balls, driven by a durable generic `reach_map` greedy descent.

**Architecture:** Add `reach_map` beside `travel_to` in `env/map_traveler.py`: a generic hop loop that crosses each map in a caller-supplied direction until a goal map is reached. Two crossing kinds — a **directional border crossing** DOWN (navigate to the southmost reachable border cell, press DOWN to cross, record the reversible portal) and an interior door-warp UP (walk onto the `MB_WARP` tile, settle, record). `run_campaign` dispatches a single reach-home milestone; the anchor state moves to `post_starter.state` (already on route_101) and the Oldale/route_103 return legs are dropped entirely.

> **Task 1 de-risk finding (2026-08-12, PROVEN live):** the generic `explore_grid` sweep is the WRONG DOWN primitive — from post_starter (route_101 (10,17)) it greedily leaves via route_101's NORTH border into Oldale (0,10), not south to Littleroot (0,9), dumping the player onto the un-traversable Oldale map. The directional `_column_scan` (edge-first border cells + `probe_step` DOWN) crosses cleanly: `route_101 (10,19) --down--> Littleroot (0,9)@(10,1)`, then `_warp_scan_up @ (7,17) --> lab (1,4)@(6,12)`. So `_cross_border` below is a directional column-scan (ported from `tools/probe_return_portals.py:_column_scan`), NOT an `explore_grid` sweep.

**Tech Stack:** Python 3.12, pytest, ruff (line-length 100). Pokémon Emerald (BPEF) via the mgba `GbaEmulator`; PPO Fighter via stable-baselines3 (ROM smoke only).

**Spec:** `docs/superpowers/specs/2026-08-12-phase2-return-descent-b2-design.md`

---

## File Structure

- `env/map_traveler.py` — add `reach_map` (loop) + `_cross_in_direction` (dispatch) + `_cross_border` (directional DOWN column-scan) + `_cross_up_warp` (UP door warp) + precision-walk helpers. Reuses the file's existing `_snapshot_settled`, `BATTLE_OUTCOMES`.
- `env/campaign.py` — `Milestone.reach` field, `_RETURN_DIRECTIONS`, `run_campaign` reach dispatch, `PHASE2_CAMPAIGN` reach-home + shoes→lab edit, delete `_PortalSeed`/`_RETURN_PORTALS`/`seed_return_portals` and the now-dead `ROUTE_103`/`OLDALE` constants.
- `tools/probe_phase2_b2.py` — throwaway de-risk probe (not committed as durable; left untracked or deleted after the chain is frozen).
- `tests/test_map_traveler.py` — reach_map loop + crossing unit tests.
- `tests/test_campaign.py` — reach-dispatch tests; remove the `seed_return_portals` import/test and `OLDALE`/`ROUTE_103` imports.
- `tests/test_phase2_rom.py` — rewrite to anchor on `post_starter.state`, drop the seed call.

---

## Task 1: De-risk probe (throwaway, gating)

**Files:**
- Create: `tools/probe_phase2_b2.py`

This task settles the crux empirically before any durable code: from `post_starter.state` (player on route_101), a continuous descent route_101 → Littleroot → lab reaches the lab with NO reload, and lab A-spam raises `has_pokedex()`. It reuses the proven helpers already in `tools/probe_return_portals.py`.

- [ ] **Step 1: Write the probe**

```python
"""Throwaway de-risk probe: continuous route_101 -> Littleroot -> lab from post_starter.

Proves the two facts B2's durable code leans on, with NO reload mid-descent:
  1. route_101 --down(explore_grid sweep)--> Littleroot --up(door warp)--> lab reaches the lab.
  2. From the lab landing, bounded A-spam raises has_pokedex() (and holds 5 Poke Balls).

Output is stdout only (REACHED lab = .../ pokedex raised = ...); nothing is committed.
Run:
  cd /Users/_eloi/Projets/Emu-phase2-return-reload
  POKEMON_EMERALD_ROM="/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba" \\
      /Users/_eloi/Projets/Emu/.venv/bin/python tools/probe_phase2_b2.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator import buttons
from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.grid_navigator import snapshot_settled
from env.map_memory import MapMemory
from env.world_reader import WorldReader
from tools.probe_return_portals import (
    LAB,
    LITTLEROOT,
    ROUTE_101,
    _hop_via_explore_then_scan,
    _load_state,
    _warp_scan_up,
)

_A_PRESS_FRAMES = 6
_A_RELEASE_FRAMES = 10
_A_MAX_PRESSES = 2000


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    fighter_ckpt = "checkpoints/fighter/ppo_fighter_final.zip"

    emu = GbaEmulator(rom)
    _load_state(emu, "states/post_starter.state")
    world = WorldReader(emu.read_bytes)
    game = EmeraldReader(emu.read_bytes)

    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn

    model = PPO.load(fighter_ckpt, device="cpu")
    mtf = make_move_type_fn(emu)

    def predict(obs: object) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    memory = MapMemory()
    here = snapshot_settled(world)
    print(f"start: {here.map_id if here else None} @ {here.pos if here else None}")

    note = _hop_via_explore_then_scan(emu, world, memory, ROUTE_101, LITTLEROOT, "down", mtf, predict)
    here = snapshot_settled(world)
    print(f"route_101 -> Littleroot: note={note!r} now={here.map_id if here else None}")

    note = _warp_scan_up(emu, world, memory, LITTLEROOT, LAB, mtf, predict)
    here = snapshot_settled(world)
    reached = here is not None and here.map_id == LAB
    print(f"Littleroot -> lab: note={note!r} now={here.map_id if here else None}")
    print(f"REACHED lab = {reached}")

    before = game.has_pokedex()
    for _ in range(_A_MAX_PRESSES):
        if game.has_pokedex():
            break
        emu.step(buttons.KEY_A, _A_PRESS_FRAMES)
        emu.step(0, _A_RELEASE_FRAMES)
    print(f"pokedex before={before} raised={game.has_pokedex()} balls5={game.has_item(0x4, 5)}")
    sys.exit(0 if reached and game.has_pokedex() else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && POKEMON_EMERALD_ROM="/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba" /Users/_eloi/Projets/Emu/.venv/bin/python tools/probe_phase2_b2.py 2>&1 | grep -Ev "mgba|GBA|core|WARN|INFO|\.so|libmgba"`

Expected: prints `REACHED lab = True` and `pokedex before=False raised=True balls5=True`; exit 0.

**GATE:** If `REACHED lab = False` or `raised=False`, STOP. A crossing primitive is still wrong — do not proceed to Task 2. Report which fact failed.

- [ ] **Step 3: Commit the probe (throwaway, kept for provenance)**

```bash
cd /Users/_eloi/Projets/Emu-phase2-return-reload
git add tools/probe_phase2_b2.py
git commit -m "probe: de-risk B2 — continuous route_101 -> lab from post_starter + pokedex A-spam

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Crossing helpers (`_cross_in_direction`, `_cross_border`, `_cross_up_warp`)

**Files:**
- Modify: `env/map_traveler.py` (add helpers after `travel_to`)
- Test: `tests/test_map_traveler.py` (append)

The two crossing kinds and their dispatch. `_cross_border` (DOWN) is a **directional column-scan** (ported from the proven `tools/probe_return_portals.py:_column_scan`): enumerate standable border cells whose `direction`-neighbour is off-map, edge-first (southmost for DOWN), `navigate_grid` to each, then `probe_step` DOWN up to `_SCAN_HOLD_PRESSES` times; the first cell that flips the map wins and records the reversible portal. `_cross_up_warp` (UP) walks onto the interior `MB_WARP` tile and settles. `_cross_in_direction` picks by direction. Unit tests monkeypatch `navigate_grid`/`probe_step`/`GridSnapshot`/`plan_path_grid`/`handle_battle_interruption`/`_warp_cells` so the ROM-empirical behaviour stays in Task 1's probe and Task 5's smoke; here we test control flow and portal bookkeeping.

- [ ] **Step 1: Write failing tests for the crossing helpers**

Append to `tests/test_map_traveler.py`:

```python
from env import map_traveler
from env.map_traveler import _cross_border, _cross_in_direction, _cross_up_warp


from env.map_grid_reader import TileKind


class _OneMapWorld:
    """Single-map fake that can be flipped to another map by a helper stub."""

    def __init__(self, map_id=(0, 16), pos=(10, 17)) -> None:
        self.map_id = map_id
        self.pos = pos
        self.grid_reader = object()   # GridSnapshot.from_reader is monkeypatched, so opaque

    def step(self, keys: int, frames: int) -> None:
        pass

    def snapshot(self):
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def in_battle(self) -> bool:
        return False

    def battle_starting(self) -> bool:
        return False


class _FakeSnap:
    """Fake GridSnapshot: FREE only at the given cells, WALL elsewhere."""

    def __init__(self, free, w=20, h=20) -> None:
        self._free = set(free)
        self.width, self.height = w, h

    def classify_at(self, x, y):
        return TileKind.FREE if (x, y) in self._free else TileKind.WALL


def test_cross_in_direction_dispatches_up_to_warp(monkeypatch):
    seen = []
    monkeypatch.setattr(map_traveler, "_cross_up_warp",
                        lambda *a, **k: seen.append("up") or "crossed")
    monkeypatch.setattr(map_traveler, "_cross_border",
                        lambda *a, **k: seen.append("border") or "crossed")
    world = _OneMapWorld()
    assert _cross_in_direction(world, world, MapMemory(), (0, 9), "up") == "crossed"
    assert seen == ["up"]


def test_cross_in_direction_dispatches_down_to_border(monkeypatch):
    seen = []
    monkeypatch.setattr(map_traveler, "_cross_up_warp",
                        lambda *a, **k: seen.append("up") or "crossed")
    monkeypatch.setattr(map_traveler, "_cross_border",
                        lambda *a, **k: seen.append("border") or "crossed")
    world = _OneMapWorld()
    assert _cross_in_direction(world, world, MapMemory(), (0, 16), "down") == "crossed"
    assert seen == ["border"]


def _stub_border_env(monkeypatch, world):
    """Wire the directional-scan boundary: opaque snap, reachable cells, no battle."""
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19), (10, 17)})))
    monkeypatch.setattr(map_traveler, "plan_path_grid", lambda snap, a, b: ["down"])
    monkeypatch.setattr(map_traveler, "handle_battle_interruption", lambda *a, **k: None)

    def fake_nav(emu, rdr, cell, **kw):
        world.pos = cell
        return "arrived"

    monkeypatch.setattr(map_traveler, "navigate_grid", fake_nav)


def test_cross_border_crosses_the_south_border_and_records(monkeypatch):
    world = _OneMapWorld(map_id=(0, 16), pos=(10, 17))
    memory = MapMemory()
    _stub_border_env(monkeypatch, world)

    def fake_probe(emu, rdr, before, direction):
        world.map_id = (0, 9)   # pressing DOWN on the border cell flips to Littleroot
        return "transition"

    monkeypatch.setattr(map_traveler, "probe_step", fake_probe)
    assert _cross_border(world, world, memory, (0, 16), "down", None, None) == "crossed"
    assert world.map_id == (0, 9)
    assert memory.portal((0, 16), (0, 9)) is not None


def test_cross_border_reports_no_crossing_when_border_never_flips(monkeypatch):
    world = _OneMapWorld(map_id=(0, 16), pos=(10, 17))
    _stub_border_env(monkeypatch, world)
    monkeypatch.setattr(map_traveler, "probe_step", lambda *a: "blocked")   # never crosses
    assert _cross_border(world, world, MapMemory(), (0, 16), "down", None, None) == "no_crossing"


def test_cross_border_passes_through_a_battle_outcome(monkeypatch):
    world = _OneMapWorld(map_id=(0, 16), pos=(10, 17))
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19)})))
    monkeypatch.setattr(map_traveler, "plan_path_grid", lambda snap, a, b: ["down"])
    monkeypatch.setattr(map_traveler, "handle_battle_interruption",
                        lambda *a, **k: "battle_lost")
    assert _cross_border(world, world, MapMemory(), (0, 16), "down", None, None) == "battle_lost"


def test_cross_up_warp_walks_onto_a_warp_tile_and_records(monkeypatch):
    world = _OneMapWorld(map_id=(0, 9), pos=(8, 18))
    memory = MapMemory()
    monkeypatch.setattr(map_traveler, "_warp_cells", lambda rdr, snap, m: [(8, 17)])
    monkeypatch.setattr(map_traveler, "GridSnapshot",
                        type("_GS", (), {"from_reader": staticmethod(lambda *a: object())}))

    def fake_walk(emu, rdr, snap, cell, from_map):
        world.pos = cell
        world.map_id = (1, 4)  # walking onto the warp tile triggers the transition
        return True

    monkeypatch.setattr(map_traveler, "_precision_walk_to", fake_walk)
    assert _cross_up_warp(world, world, memory, (0, 9), None, None) == "crossed"
    assert memory.portal((0, 9), (1, 4)) is not None


def test_cross_up_warp_reports_no_crossing_without_a_warp_tile(monkeypatch):
    world = _OneMapWorld(map_id=(0, 9))
    monkeypatch.setattr(map_traveler, "_warp_cells", lambda rdr, snap, m: [])
    monkeypatch.setattr(map_traveler, "GridSnapshot",
                        type("_GS", (), {"from_reader": staticmethod(lambda *a: object())}))
    assert _cross_up_warp(world, world, MapMemory(), (0, 9), None, None) == "no_crossing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k "cross_" -q`
Expected: FAIL with `ImportError: cannot import name '_cross_border'` (and siblings).

- [ ] **Step 3: Implement the crossing helpers**

Add to the imports at the top of `env/map_traveler.py`:

```python
from emulator import buttons
from env.grid_navigator import (
    DELTAS,
    handle_battle_interruption,
    navigate_grid,
    plan_path_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
```

(Replace the existing `from env.grid_navigator import DELTAS, navigate_grid` with the single import above so `DELTAS`, `navigate_grid`, `plan_path_grid`, `probe_step`, `handle_battle_interruption`, `snapshot_settled` are all imported once. `explore_grid`/`travel_to` are NOT used by the directional crossing — do not import `explore_grid`.)

Append after `travel_to`:

```python
_MB_WARP = 0x60          # pokeemerald metatile_behaviors.h MB_WARP (interior door tile)
_WARP_SETTLE_FRAMES = 64  # let the warp engine complete after stepping onto the tile
_UP_HOLD_FRAMES = 24
_PRECISION_STEP_FRAMES = 4
_PRECISION_RELEASE_FRAMES = 32
_PRECISION_MAX_STEPS = 600
_SCAN_MAX_CANDIDATES = 12   # cap border cells tried per crossing (matches _column_scan)
_SCAN_HOLD_PRESSES = 20     # direction presses per candidate before giving up on it
_STANDABLE = frozenset({TileKind.FREE, TileKind.GRASS})
_KEY_FOR = {
    "up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT,
}
# Try the most promising border exits first: southmost for DOWN, northmost for UP,
# westmost for LEFT, eastmost for RIGHT.
_BORDER_SORT = {
    "down": lambda c: -c[1], "up": lambda c: c[1],
    "left": lambda c: c[0], "right": lambda c: -c[0],
}


def _precision_step(emulator: Any, key: int) -> None:
    """Press a direction for 4 frames (1-tile precision) then release."""
    emulator.step(key, _PRECISION_STEP_FRAMES)
    emulator.step(0, _PRECISION_RELEASE_FRAMES)


def _precision_walk_to(
    emulator: Any, reader: Any, snap: Any, target: tuple[int, int], from_map: tuple[int, int]
) -> bool:
    """Re-plan each step and take 4-frame precision steps toward target on from_map.

    navigate_grid's 24-frame steps overshoot on Littleroot's dense topology; precision
    steps land exactly. Returns True on arrival, False if it leaves the map or stalls.
    """
    for _ in range(_PRECISION_MAX_STEPS):
        here = snapshot_settled(reader)
        if here is None or here.map_id != from_map:
            return False
        if here.pos == target:
            return True
        path = plan_path_grid(snap, here.pos, target)
        if not path:
            return False
        _precision_step(emulator, _KEY_FOR[path[0]])
    return False


def _warp_cells(reader: Any, snap: Any, map_id: tuple[int, int]) -> list[tuple[int, int]]:
    """FREE cells whose tile behavior is MB_WARP (0x60): interior door tiles."""
    out: list[tuple[int, int]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) != TileKind.FREE:
                continue
            if reader.grid_reader.tile_behavior_at(x, y) == _MB_WARP:
                out.append((x, y))
    return out


def _cross_in_direction(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int], direction: str,
    move_type_fn: Any = None, predict: Any = None,
) -> str:
    """Cross from_map's border going `direction`. Two kinds, chosen by direction.

    Returns 'crossed' | 'no_crossing' | a battle outcome.
    """
    if direction == "up":
        return _cross_up_warp(emulator, reader, memory, from_map, move_type_fn, predict)
    return _cross_border(emulator, reader, memory, from_map, direction, move_type_fn, predict)


def _border_cells(snap: Any, here_pos: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """Standable cells whose `direction`-neighbour is off-map and that are reachable
    from here_pos, sorted edge-first (southmost for DOWN, etc.)."""
    dx, dy = DELTAS[direction]
    out: list[tuple[int, int]] = []
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < snap.width and 0 <= ny < snap.height:
                continue   # neighbour is on-map: not a border cell for this direction
            if (x, y) == here_pos or plan_path_grid(snap, here_pos, (x, y)) is not None:
                out.append((x, y))
    out.sort(key=_BORDER_SORT[direction])
    return out[:_SCAN_MAX_CANDIDATES]


def _cross_border(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int], direction: str,
    move_type_fn: Any, predict: Any,
) -> str:
    """Directional border crossing (ported from probe_return_portals._column_scan).

    Navigate to each edge-first border cell and press `direction`; the first cell that
    flips the map wins and records the reversible portal. A DIRECTED descent, unlike the
    greedy explore_grid sweep which leaves via the first non-reversible border in ANY
    direction (route_101's north exit to Oldale instead of its south exit to Littleroot).
    Returns 'crossed' | 'no_crossing' | a battle outcome.
    """
    here = _snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return "no_crossing"
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return "no_crossing"
    for cell in _border_cells(snap, here.pos, direction):
        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle
        arrived = navigate_grid(
            emulator, reader, cell, memory=memory, move_type_fn=move_type_fn, predict=predict
        )
        if arrived in BATTLE_OUTCOMES:
            return arrived
        if arrived != "arrived":
            continue
        for _ in range(_SCAN_HOLD_PRESSES):
            battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
            if battle is not None:
                return battle
            before = _snapshot_settled(reader)
            if before is None or before.map_id != from_map:
                break
            probe_step(emulator, reader, before, direction)
            after = _snapshot_settled(reader)
            if after is not None and after.map_id != from_map:
                memory.record_portal(
                    from_map, before.pos, direction, after.map_id, True, after.pos
                )
                return "crossed"
    return "no_crossing"


def _cross_up_warp(
    emulator: Any, reader: Any, memory: MapMemory, from_map: tuple[int, int],
    move_type_fn: Any, predict: Any,
) -> str:
    """UP interior door: walk onto an MB_WARP tile (which triggers the warp), settle,
    record the portal. explore_grid only tests map-edge cells, so it misses interior doors."""
    battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
    if battle is not None:
        return battle
    here = _snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return "no_crossing"
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return "no_crossing"
    for cell in _warp_cells(reader, snap, from_map):
        cur = _snapshot_settled(reader)
        if cur is None or cur.map_id != from_map:
            return "no_crossing"
        _precision_walk_to(emulator, reader, snap, cell, from_map)
        landed = _snapshot_settled(reader)
        if landed is not None and landed.map_id != from_map:
            memory.record_portal(from_map, cell, "up", landed.map_id, True, landed.pos)
            return "crossed"
        emulator.step(_KEY_FOR["up"], _UP_HOLD_FRAMES)
        emulator.step(0, _WARP_SETTLE_FRAMES)
        settled = _snapshot_settled(reader)
        if settled is not None and settled.map_id != from_map:
            memory.record_portal(from_map, cell, "up", settled.map_id, True, settled.pos)
            return "crossed"
    return "no_crossing"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k "cross_" -q`
Expected: 7 passed.

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/_eloi/Projets/Emu-phase2-return-reload
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/map_traveler.py tests/test_map_traveler.py
git add env/map_traveler.py tests/test_map_traveler.py
git commit -m "feat: reach_map crossing helpers — border DOWN sweep + interior door warp UP

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `reach_map` greedy-descent loop

**Files:**
- Modify: `env/map_traveler.py` (add `reach_map` after the crossing helpers)
- Test: `tests/test_map_traveler.py` (append)

The generic hop loop. Snapshot the current map; if it is `goal_map`, `arrived`. Else look up the direction (absent → `stall`), cross, re-evaluate. Battle outcomes pass through. Loop tests monkeypatch `_cross_in_direction` so they exercise the loop, not the ROM crossing.

- [ ] **Step 1: Write failing tests for reach_map**

Append to `tests/test_map_traveler.py`:

```python
from env.map_traveler import reach_map


class _ScriptedDescentWorld:
    """Fake whose map flips through a scripted sequence each time a crossing fires."""

    def __init__(self, sequence: list[tuple[int, int]]) -> None:
        self._sequence = sequence
        self.map_id = sequence[0]
        self.pos = (10, 17)
        self._i = 0

    def advance(self) -> None:
        if self._i < len(self._sequence) - 1:
            self._i += 1
            self.map_id = self._sequence[self._i]

    def step(self, keys: int, frames: int) -> None:
        pass

    def snapshot(self):
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def in_battle(self) -> bool:
        return False

    def battle_starting(self) -> bool:
        return False


def test_reach_map_arrives_over_two_hops(monkeypatch):
    world = _ScriptedDescentWorld([(0, 16), (0, 9), (1, 4)])
    directions = {(0, 16): "down", (0, 9): "up"}

    def fake_cross(emu, rdr, mem, from_map, direction, **kw):
        world.advance()
        return "crossed"

    monkeypatch.setattr(map_traveler, "_cross_in_direction", fake_cross)
    assert reach_map(world, world, MapMemory(), (1, 4), directions) == "arrived"


def test_reach_map_already_on_goal_returns_arrived():
    world = _ScriptedDescentWorld([(1, 4)])
    assert reach_map(world, world, MapMemory(), (1, 4), {}) == "arrived"


def test_reach_map_stalls_on_an_unexpected_map():
    world = _ScriptedDescentWorld([(0, 16)])
    assert reach_map(world, world, MapMemory(), (1, 4), {(0, 9): "up"}) == "stall"


def test_reach_map_stalls_when_no_crossing_fires(monkeypatch):
    world = _ScriptedDescentWorld([(0, 16)])
    monkeypatch.setattr(map_traveler, "_cross_in_direction", lambda *a, **k: "no_crossing")
    assert reach_map(world, world, MapMemory(), (1, 4), {(0, 16): "down"}) == "stall"


def test_reach_map_passes_through_a_battle_outcome(monkeypatch):
    world = _ScriptedDescentWorld([(0, 16)])
    monkeypatch.setattr(map_traveler, "_cross_in_direction", lambda *a, **k: "battle_lost")
    assert reach_map(world, world, MapMemory(), (1, 4), {(0, 16): "down"}) == "battle_lost"


def test_reach_map_times_out_when_hops_exhaust(monkeypatch):
    world = _ScriptedDescentWorld([(0, 16)])
    monkeypatch.setattr(map_traveler, "_cross_in_direction", lambda *a, **k: "crossed")
    assert reach_map(world, world, MapMemory(), (1, 4), {(0, 16): "down"}, max_hops=3) == "timeout"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k "reach_map" -q`
Expected: FAIL with `ImportError: cannot import name 'reach_map'`.

- [ ] **Step 3: Implement reach_map**

Add after `_cross_up_warp` in `env/map_traveler.py`:

```python
def reach_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    goal_map: tuple[int, int],
    direction_by_map: dict[tuple[int, int], str],
    *,
    move_type_fn: Any = None,
    predict: Any = None,
    max_hops: int = 12,
) -> str:
    """Greedy descent: cross each map in its prescribed direction until goal_map.

    direction_by_map maps a map_id to the crossing direction to take FROM it. goal_map
    is the stop condition only — reach_map never routes 'toward' it geometrically.
    Portals are recorded live as they are discovered (no pre-seeding needed).

    Returns 'arrived' | 'stall' | 'timeout' | 'battle_lost' | 'battle_timeout' |
    'battle_interrupted'. 'stall' = current map not in direction_by_map, or no crossing
    fired. 'timeout' = hop budget exhausted.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating between maps; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return "arrived"
        direction = direction_by_map.get(here.map_id)
        if direction is None:
            return "stall"
        crossed = _cross_in_direction(
            emulator, reader, memory, here.map_id, direction,
            move_type_fn=move_type_fn, predict=predict,
        )
        if crossed in BATTLE_OUTCOMES:
            return crossed
        if crossed != "crossed":
            return "stall"
    return "timeout"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -q`
Expected: all pass (existing travel_to tests + the 7 crossing + 6 reach_map tests).

- [ ] **Step 5: Lint + commit**

```bash
cd /Users/_eloi/Projets/Emu-phase2-return-reload
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/map_traveler.py tests/test_map_traveler.py
git add env/map_traveler.py tests/test_map_traveler.py
git commit -m "feat: reach_map generic greedy-descent loop (arrived/stall/timeout/battle)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Campaign wiring

**Files:**
- Modify: `env/campaign.py`
- Test: `tests/test_campaign.py`

Add `Milestone.reach`, dispatch it at the head of `run_campaign`, define `_RETURN_DIRECTIONS`, rebuild `PHASE2_CAMPAIGN` (reach-home + shoes→lab), and delete the dead placeholder portal seeding.

- [ ] **Step 1: Write failing tests for the reach dispatch**

In `tests/test_campaign.py`, remove `OLDALE`, `ROUTE_103`, `seed_return_portals` from the import block (lines 3-14) and delete any test that calls `seed_return_portals` (search the file for `seed_return_portals`). Then append:

```python
from env import campaign
from env.campaign import _RETURN_DIRECTIONS


def test_reach_milestone_dispatches_reach_map(monkeypatch):
    calls = []

    def fake_reach(emu, rdr, mem, goal, directions, **kw):
        calls.append((goal, directions))
        return "arrived"

    monkeypatch.setattr(campaign, "reach_map", fake_reach)
    result = run_campaign(
        None, FakeReader([5]), MapMemory(),
        curriculum=(Milestone("lab", 0, reach=LAB),),
    )
    assert result == "campaign_complete"
    assert calls == [(LAB, _RETURN_DIRECTIONS)]


def test_reach_milestone_aborts_on_non_arrived(monkeypatch):
    monkeypatch.setattr(campaign, "reach_map", lambda *a, **k: "stall")
    result = run_campaign(
        None, FakeReader([5]), MapMemory(),
        curriculum=(Milestone("lab", 0, reach=LAB),),
    )
    assert result == "stall"


def test_return_directions_are_route101_down_littleroot_up():
    assert _RETURN_DIRECTIONS == {ROUTE_101: "down", LITTLEROOT: "up"}


def test_phase2_campaign_opens_with_a_reach_home_milestone():
    assert PHASE2_CAMPAIGN[0].reach == LAB
    # Every story milestone targets the lab (travel-first mode; 0-step arrival).
    assert all(m.destination == "lab" for m in PHASE2_CAMPAIGN if m.story_target is not None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -q`
Expected: FAIL — `ImportError: cannot import name '_RETURN_DIRECTIONS'` / `Milestone.__init__() got an unexpected keyword argument 'reach'`.

- [ ] **Step 3: Edit `env/campaign.py`**

Delete the comment block + `_PortalSeed` class + `_RETURN_PORTALS` + `seed_return_portals` (current lines 28-59). Delete the now-unused `ROUTE_103` and `OLDALE` constants (lines 22-23). Change the import `from typing import Any, NamedTuple` to `from typing import Any` (NamedTuple was only used by `_PortalSeed`). Add `from env.map_traveler import reach_map` beside the existing imports.

The map-id constants block becomes:

```python
# Map-group ids on the southbound return path.
ROUTE_101 = (0, 16)
LITTLEROOT = (0, 9)
LAB = (1, 4)

# Return chain for reach_map: cross route_101 south (down) into Littleroot, then the
# lab door warp (up). Oldale/route_103 are dropped — the north-entry Oldale hop is not
# crossable in a continuous descent (see the B2 spec), and B2 starts on route_101.
_RETURN_DIRECTIONS: dict[tuple[int, int], str] = {ROUTE_101: "down", LITTLEROOT: "up"}
```

Add the `reach` field to `Milestone` (as the last field so defaults stay valid):

```python
@dataclass(frozen=True)
class Milestone:
    """One curriculum step: reach `destination` once the mean party level is at
    least `target_level`; if `trainer`, fight the trainer there on arrival; if
    `reach` is set, greedy-descend to that goal map via reach_map instead."""

    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # mean, not max — one powerhouse shouldn't unlock advance
    trainer: bool = False   # end the milestone with a battle_trainer Order
    story_target: Callable[[Any], bool] | None = None   # story mode: A-spam until this holds
    reach: tuple[int, int] | None = None   # reach-home mode: reach_map(goal=reach)
```

Rebuild `PHASE2_CAMPAIGN`:

```python
# Phase 2 curriculum (B2). Start on route_101 (post_starter.state): greedy-descend home
# to the lab, then A-spam the Pokédex + 5 Poké Balls cutscene. The shoes flag is already
# set at post_starter, so its milestone is an idempotent post-assert AT THE LAB (the story
# mode is travel-first, so it must not name a cell away from where the player stands).
PHASE2_CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("lab", 0, reach=LAB),
    Milestone("lab", 0, story_target=lambda r: r.has_pokedex()),
    Milestone("lab", 0, story_target=lambda r: r.has_item(0x4, 5)),
    Milestone("lab", 0, story_target=lambda r: r.has_running_shoes()),
)
```

Add the reach branch at the head of the `run_campaign` loop, before the `story_target` branch:

```python
    for milestone in curriculum:
        if milestone.reach is not None:
            arrived = reach_map(
                emulator, reader, memory, milestone.reach, _RETURN_DIRECTIONS,
                move_type_fn=move_type_fn, predict=predict,
            )
            if arrived != "arrived":
                return arrived
            continue
        if milestone.story_target is not None:
            ...
```

(Leave the rest of the loop body unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -q`
Expected: all pass.

- [ ] **Step 5: Full suite + lint + commit**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/campaign.py tests/test_campaign.py`
Expected: whole suite green (ROM smokes skip without ROM), ruff clean.

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "feat: PHASE2_CAMPAIGN reach-home milestone via reach_map; drop placeholder portals

Add Milestone.reach + a head-of-loop reach dispatch; _RETURN_DIRECTIONS = {route_101:
down, littleroot: up}. Retarget the shoes milestone to 'lab' (story mode is travel-first;
the flag is already set so it is a 0-step idempotent post-assert). Delete the never-
verified _RETURN_PORTALS/seed_return_portals and the dead route_103/Oldale constants —
reach_map discovers portals live.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: ROM smoke from `post_starter.state`

**Files:**
- Modify: `tests/test_phase2_rom.py`

Rewrite the gated smoke to anchor on `post_starter.state` and drop the seed call: `reach_map` discovers the portals live. Assert the deliverables and dump `post_phase2.state`.

- [ ] **Step 1: Rewrite the smoke**

Replace the whole body of `tests/test_phase2_rom.py`:

```python
"""Gated ROM smoke: run PHASE2_CAMPAIGN (B2) from post_starter.state end to end.

reach_map greedy-descends route_101 -> Littleroot -> lab (portals discovered live, no
seed), then the story A-spam delivers the Pokédex + 5 Poké Balls. Asserts the
deliverables landed and dumps states/post_phase2.state. Triple-skips without ROM /
Fighter checkpoint / post_starter.state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"
START_STATE = "states/post_starter.state"

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(FIGHTER_CKPT).exists(), reason="Fighter checkpoint missing"),
    pytest.mark.skipif(not Path(START_STATE).exists(), reason="post_starter.state missing"),
]


def test_phase2_campaign_delivers_pokedex_and_balls() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.campaign import PHASE2_CAMPAIGN, run_campaign
    from env.game_state import EmeraldReader
    from env.map_memory import MapMemory
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    with open(START_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)
    memory = MapMemory()

    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs):
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)

    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_campaign(
        emu, _Reader(), memory,
        curriculum=PHASE2_CAMPAIGN,
        move_type_fn=move_type_fn, predict=predict,
    )

    assert result == "campaign_complete", result
    assert reader.has_pokedex()
    assert reader.has_item(0x4, 5)

    Path("states/post_phase2.state").write_bytes(emu.save_state())
```

- [ ] **Step 2: Run the gated smoke (local, with ROM)**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && POKEMON_EMERALD_ROM="/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba" /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_phase2_rom.py -q`
Expected: 1 passed (state file `states/post_phase2.state` written). If the campaign returns a non-`campaign_complete` string, the assertion prints it — feed that back into Task 2/3 (a crossing returned `stall`/battle).

- [ ] **Step 3: Run the whole suite without ROM (skip-guard sanity) + lint**

Run: `cd /Users/_eloi/Projets/Emu-phase2-return-reload && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check tests/test_phase2_rom.py`
Expected: full suite green with the Phase 2 ROM smoke skipped (no ROM in CI), ruff clean.

- [ ] **Step 4: Commit**

```bash
cd /Users/_eloi/Projets/Emu-phase2-return-reload
git add tests/test_phase2_rom.py
git commit -m "test: Phase 2 ROM smoke anchors on post_starter.state, reach_map discovers portals

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Unit 1 (de-risk probe) → Task 1 (gate: REACHED lab + pokedex raised). ✓
- Unit 2 (reach_map, two crossing kinds) → Task 2 (crossing helpers) + Task 3 (loop). ✓ Both kinds have distinct code paths and distinct tests (spec G2). Door-warp settle included (spec G3, `_WARP_SETTLE_FRAMES`). Fighter threaded through `move_type_fn`/`predict` (spec G4). Sweep starts from an arbitrary cell (spec G5, explore_grid handles it). Portals discovered live (spec G6, no seed).
- Unit 3 (campaign wiring) → Task 4: `_RETURN_DIRECTIONS`, `Milestone.reach`, head-of-loop dispatch, `PHASE2_CAMPAIGN` reach-home, shoes→lab (spec G7), delete `_RETURN_PORTALS`/`seed_return_portals` + its test (spec G8: the only other caller, `test_phase2_rom.py`, is rewritten in Task 5). ✓
- Unit 4 (ROM smoke from post_starter) → Task 5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test asserts concrete values. The de-risk probe reuses named helpers from `probe_return_portals.py` (`_hop_via_explore_then_scan`, `_warp_scan_up`, `_load_state`) confirmed present in that file. ✓

**Type consistency:** `reach_map`/`_cross_in_direction`/`_cross_border`/`_cross_up_warp` signatures and the `"crossed"|"no_crossing"|<battle>` contract are consistent across Tasks 2-3. `Milestone.reach: tuple[int,int] | None`; `_RETURN_DIRECTIONS: dict[tuple[int,int], str]`; `run_campaign` compares `reach_map(...) != "arrived"`. `MapMemory.outgoing_portals`/`record_portal`/`portal` match the real API (`env/map_memory.py:81-110`). `has_item(0x4, 5)` matches `EmeraldReader.has_item`. ✓

---

## Execution Handoff

After Task 5, run `superpowers:finishing-a-development-branch` (verify tests → merge `feat/phase2-return-reload-per-hop` to `main` via `--no-ff` → clean branch + worktree), then delete `tools/probe_phase2_b2.py` if you prefer not to keep the throwaway probe, and update the crux memory to record B2 landed.
