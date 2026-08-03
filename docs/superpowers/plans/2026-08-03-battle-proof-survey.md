# Battle-proof survey (Brique 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `map_map` (survey) survive a mid-survey wild battle by handing it to the Fighter — the same fix Brique 1 gave `navigate_to` — thread the Fighter through `map_map` + `survey_world`, and prove it against the ROM with a deterministic in-battle savestate.

**Architecture:** Promote `live_navigator._handle_battle_interruption` to public `handle_battle_interruption`, call it at the top of the `map_map` loop (after the encounter-recording block, before frontier selection); propagate its battle outcomes as new `map_map` return values. `survey_world` forwards the Fighter deps to both `travel_to` and `map_map` and aborts the sweep on the first battle outcome (reusing `map_traveler.BATTLE_OUTCOMES`). A disposable capture tool produces a mid-battle route_101 savestate so a gated ROM smoke exercises the fix every run.

**Tech Stack:** Python 3.12, pytest (no network), Stable-Baselines3 (Fighter, imported inside ROM test body only), Emerald BPEF ROM.

**Working directory:** `/Users/_eloi/Projets/Emu-p4-battle-proof-survey` (worktree, branch `feat/p4-battle-proof-survey`). Run tests from the main repo `/Users/_eloi/Projets/Emu` where ROM/checkpoints/states live; pure tests run in either.

---

### Task 1: Promote `_handle_battle_interruption` to public

**Files:**
- Modify: `env/live_navigator.py:32` (rename + docstring) and its call site `env/live_navigator.py:90`
- Test: `tests/test_live_navigator.py`

The helper is used only inside `live_navigator` today; Brique 2 is the first cross-module import. A public name avoids importing a `_private` across modules (same fix that promoted `_reached` → `reached` in étape 6).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_navigator.py`:

```python
def test_handle_battle_interruption_is_public_and_quiet_off_battle() -> None:
    from env.live_navigator import handle_battle_interruption

    class _NoBattle:
        def in_battle(self) -> bool:
            return False

    # No battle -> None, and no Fighter needed to reach that branch.
    assert handle_battle_interruption(None, _NoBattle(), None, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_live_navigator.py::test_handle_battle_interruption_is_public_and_quiet_off_battle -v`
Expected: FAIL with `ImportError: cannot import name 'handle_battle_interruption'`.

- [ ] **Step 3: Rename the helper to public**

In `env/live_navigator.py`, rename the definition (currently line 32):

```python
def handle_battle_interruption(
    emulator: Any, reader: Any, move_type_fn: Any, predict: Any
) -> str | None:
    """If a wild battle is in progress, hand it to the Fighter and report.

    Public (imported by map_explorer as well as used here). Returns None when
    there is no battle (or the battle was won) so the caller resumes; returns a
    terminal outcome when the caller must abort: "battle_interrupted" (no Fighter
    supplied), "battle_lost", "battle_timeout".
    """
    if not reader.in_battle():
        return None
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result == "won":
        return None
    return "battle_lost" if result == "lost" else "battle_timeout"
```

And update the internal call site (currently line 90) inside `navigate_to`:

```python
        interruption = handle_battle_interruption(
            emulator, reader, move_type_fn, predict
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_navigator.py -v`
Expected: PASS (new test + all existing `navigate_to` tests, including the Brique 1 win/loss interruption tests, still green).

- [ ] **Step 5: Commit**

```bash
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "refactor: promote handle_battle_interruption to public for cross-module reuse"
```

---

### Task 2: `map_map` wires the helper + threads the Fighter

**Files:**
- Modify: `env/map_explorer.py` (import, signature, loop body, docstring return list)
- Test: `tests/test_map_explorer.py`

`map_map` currently observes `in_battle()` only to *learn* grass, then presses the d-pad during the frozen battle → false wall → spin. Wiring the helper right after the recording block fights the battle (or aborts) before any press.

- [ ] **Step 1: Write the failing tests**

In `tests/test_map_explorer.py`, first update the existing grass test to reflect the new no-Fighter abort (walking onto grass now aborts after learning it):

```python
def test_map_map_learns_grass_then_aborts_without_a_fighter():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)  # cells (0,0),(1,0),(0,1),(1,1)
    world = EncounterExploreWorld(grass_at=(1, 0), map_id=target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    # No Fighter deps: stepping onto grass learns it, then aborts on that frame.
    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "battle_interrupted"
    assert ((3, 3), (1, 0)) in memory.cells_labeled("has_grass")
```

(Delete the old `test_map_map_learns_grass_cell_when_a_battle_fires`, which asserted `"complete"` — no longer true once the battle is honoured.)

Then add a battle-capable fake and two Fighter tests. Add near the top (after the `_KEY_TO_DIR` block):

```python
def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class BattleExploreWorld(ExploreWorld):
    """ExploreWorld pre-armed in a wild battle at the start cell. A supplied
    Fighter plays it via play_battle; on a win in_battle drops so the survey
    resumes. can_win=False makes the Fighter lose. Serves battle-reader bytes
    exactly like BattleNavWorld in test_live_navigator."""

    _RESOLVE_PRESSES = 2

    def __init__(self, can_win: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._can_win = can_win
        self._battle = True
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        if self._battle:
            self._battle_step(keys)
            return
        super().step(keys, frames)

    def _battle_step(self, keys: int) -> None:
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            if not self._can_win:
                self._outcome = 2   # terminal loss -> play_battle returns "lost"
                self._battle = False
                return
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
                self._battle = False   # won: resume surveying
            self._phase = "resolving"
            self._resolve_left = self._RESOLVE_PRESSES
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def in_battle(self) -> bool:
        return self._battle

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            ACTION_MENU_VALUE,
            BATTLE_MON_SIZE,
            GBATTLE_ACTION_MENU_ADDR,
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_ACTION_MENU_ADDR:
            return bytes([ACTION_MENU_VALUE if self._phase == "menu" else 0])
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16b(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16b(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        for base, hp, mx in ((pbase, self._my_hp, 19), (obase, self._opp_hp, 18)):
            if base <= addr < base + BATTLE_MON_SIZE:
                buf = bytearray(BATTLE_MON_SIZE)
                buf[0x00:0x02] = _u16b(1)
                buf[0x0C:0x0E] = _u16b(1)
                buf[0x24] = 10
                buf[0x21], buf[0x22] = 12, 12
                buf[0x28:0x2A] = _u16b(hp)
                buf[0x2A] = 5
                buf[0x2C:0x2E] = _u16b(mx)
                off = addr - base
                return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def test_map_map_fighter_wins_the_battle_and_survey_completes():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)
    world = BattleExploreWorld(map_id=target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(
        world, world, memory, wallmap, target, max_steps=200,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )

    assert result == "complete"
    assert not world._battle  # battle was actually resolved before the survey finished


def test_map_map_fighter_loss_aborts_the_survey():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)
    world = BattleExploreWorld(map_id=target, start=(0, 0), walls=walls, can_win=False)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(
        world, world, memory, wallmap, target, max_steps=200,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )

    assert result == "battle_lost"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_map_explorer.py -v`
Expected: FAIL — `map_map()` has no `move_type_fn`/`predict` kwargs (`TypeError`), and the no-Fighter grass test still returns `"complete"`.

- [ ] **Step 3: Wire the helper into `map_map`**

In `env/map_explorer.py`, extend the import (currently `from env.live_navigator import RELEASE_FRAMES, probe_step, snapshot_settled`):

```python
from env.live_navigator import (
    RELEASE_FRAMES,
    handle_battle_interruption,
    probe_step,
    snapshot_settled,
)
```

Extend the signature:

```python
def map_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    target_map: tuple[int, int],
    max_steps: int = 2000,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
```

Insert the handler in the loop, immediately after the encounter-recording block and before `_nearest_frontier`:

```python
        reached.add(here.pos)
        if enc_watcher.observe(reader.in_battle()):
            memory.observe(here, WorldEvent(encounter_started=True))

        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle  # "battle_lost" | "battle_timeout" | "battle_interrupted"

        plan = _nearest_frontier(reached, tried, wallmap, target_map, here.pos)
```

Update the module docstring's returns note to add: `"battle_lost" / "battle_timeout" / "battle_interrupted"` when a wild battle interrupts the survey (lost, timed out, or no Fighter supplied).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_map_explorer.py -v`
Expected: PASS (all map_explorer tests, including the migrated grass test and the two new Fighter tests).

- [ ] **Step 5: Commit**

```bash
git add env/map_explorer.py tests/test_map_explorer.py
git commit -m "feat: map_map hands a mid-survey wild battle to the Fighter and resumes"
```

---

### Task 3: `survey_world` threads deps + aborts on a battle outcome

**Files:**
- Modify: `env/world_surveyor.py` (import, signature, leg handling)
- Test: `tests/test_world_surveyor.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_world_surveyor.py`, add a battle-firing grid and two tests. Add after the `WorldGrid` class:

```python
def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class BattleWorldGrid(WorldGrid):
    """WorldGrid pre-armed in a wild battle at the start cell. With a Fighter it
    plays out via play_battle (win clears in_battle so the survey resumes);
    without one, map_map returns "battle_interrupted" and the sweep aborts.
    Serves battle-reader bytes like BattleNavWorld in test_live_navigator."""

    _RESOLVE_PRESSES = 2

    def __init__(self, can_win: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._can_win = can_win
        self._battle = True
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        if self._battle:
            self._battle_step(keys)
            return
        super().step(keys, frames)

    def _battle_step(self, keys: int) -> None:
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            if not self._can_win:
                self._outcome = 2
                self._battle = False
                return
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
                self._battle = False
            self._phase = "resolving"
            self._resolve_left = self._RESOLVE_PRESSES
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def in_battle(self) -> bool:
        return self._battle

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            ACTION_MENU_VALUE,
            BATTLE_MON_SIZE,
            GBATTLE_ACTION_MENU_ADDR,
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_ACTION_MENU_ADDR:
            return bytes([ACTION_MENU_VALUE if self._phase == "menu" else 0])
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16b(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16b(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        for base, hp, mx in ((pbase, self._my_hp, 19), (obase, self._opp_hp, 18)):
            if base <= addr < base + BATTLE_MON_SIZE:
                buf = bytearray(BATTLE_MON_SIZE)
                buf[0x00:0x02] = _u16b(1)
                buf[0x0C:0x0E] = _u16b(1)
                buf[0x24] = 10
                buf[0x21], buf[0x22] = 12, 12
                buf[0x28:0x2A] = _u16b(hp)
                buf[0x2A] = 5
                buf[0x2C:0x2E] = _u16b(mx)
                off = addr - base
                return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def test_map_battle_without_a_fighter_aborts_the_sweep() -> None:
    a = (0, 0)
    walls = _sealed_room(a, 1, 1)  # single-cell sealed room
    world = BattleWorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders={})

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    # No Fighter: map_map(a) returns battle_interrupted -> sweep aborts before
    # a is counted surveyed, logging the leg.
    assert report.surveyed == ()
    assert report.failed == ((a, "map:battle_interrupted"),)


def test_fighter_win_lets_the_sweep_complete_the_map() -> None:
    a = (0, 0)
    walls = _sealed_room(a, 1, 1)
    world = BattleWorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders={})

    report = survey_world(
        world, world, MapMemory(), WallMap(), max_maps=10,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )

    # Fighter deps reach map_map: the battle is won, in_battle clears, and the
    # sealed single-cell map completes normally.
    assert report.surveyed == (a,)
    assert report.failed == ()
    assert not world._battle
```

Add the `buttons` import at the top if not present: `from emulator import buttons` (already imported).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -v`
Expected: FAIL — `survey_world()` has no `move_type_fn`/`predict` kwargs (`TypeError`), and without threading the no-Fighter battle would spin `map_map` to `budget_exhausted` rather than aborting `battle_interrupted`.

- [ ] **Step 3: Thread deps + propagate battle outcomes**

In `env/world_surveyor.py`, extend the import (currently `from env.map_traveler import travel_to`):

```python
from env.map_traveler import BATTLE_OUTCOMES, travel_to
```

Extend the signature:

```python
def survey_world(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    max_maps: int = 50,
    move_type_fn: Any = None,
    predict: Any = None,
) -> SurveyReport:
```

Replace the leg body (the `if here != target ...` travel block through `surveyed.append(target)`) with:

```python
        here = _current_map(reader)
        if here != target and target != start:
            outcome = travel_to(
                emulator, reader, memory, wallmap,
                target, _entry_cell(memory, target),
                move_type_fn=move_type_fn, predict=predict,
            )
            if outcome in BATTLE_OUTCOMES:
                failed.append((target, f"travel:{outcome}"))
                return SurveyReport(tuple(surveyed), tuple(failed))
            if outcome != "arrived":
                failed.append((target, f"travel:{outcome}"))
                continue

        result = map_map(
            emulator, reader, memory, wallmap, target,
            move_type_fn=move_type_fn, predict=predict,
        )
        if result in BATTLE_OUTCOMES:
            failed.append((target, f"map:{result}"))
            return SurveyReport(tuple(surveyed), tuple(failed))
        if result in ("left_map", "budget_exhausted"):
            failed.append((target, f"map:{result}"))
        surveyed.append(target)
```

Refresh the module docstring: a battle the Fighter loses/times out (or with no Fighter) aborts the sweep early, recorded in `failed` as `travel:<outcome>` / `map:<outcome>`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_world_surveyor.py -v`
Expected: PASS (new battle tests + all existing sweep tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add env/world_surveyor.py tests/test_world_surveyor.py
git commit -m "feat: survey_world threads the Fighter and aborts the sweep on a battle outcome"
```

---

### Task 4: Deterministic live grass smoke (capture tool + gated ROM test)

**Files:**
- Create: `tools/capture_route101_in_battle.py`
- Create: `tests/test_battle_proof_survey_rom.py`

The capture tool is disposable scaffolding (run once locally where ROM + `states/post_starter.state` exist). The gated smoke double-skips without the ROM or the produced state, and is load-bearing once the state exists.

- [ ] **Step 1: Write the capture tool**

Create `tools/capture_route101_in_battle.py`:

```python
"""Walk post_starter into route_101 grass until a wild battle fires, then cache
states/route101_in_battle.state.

Loads states/post_starter.state (a level-5 party free-roaming on route_101) and
cycles the four d-pad directions with a release-frame debounce (grind's
_walk_until_encounter pattern) until reader.in_battle() flips true, then saves.
A fixed heading would risk hitting a wall or leaving the grass patch before the
stochastic encounter roll fires, so we wander.

One-shot scaffolding: run once locally where the ROM + post_starter.state exist.
The output feeds tests/test_battle_proof_survey_rom.py (deterministic mid-battle
smoke). Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_route101_in_battle.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from emulator.buttons import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP
from emulator.gba import GbaEmulator
from env.orders import GRIND_RELEASE_FRAMES, GRIND_STEP_FRAMES
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/route101_in_battle.state")
_DIRECTIONS = (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="states/post_starter.state")
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    start = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [start], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()

    for i in range(args.max_steps):
        env.emulator.step(_DIRECTIONS[i % len(_DIRECTIONS)], GRIND_STEP_FRAMES)
        env.emulator.step(0, GRIND_RELEASE_FRAMES)  # release (GBA debounce)
        if reader.in_battle():
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            snap = reader.snapshot()
            print(
                f"IN-BATTLE state saved after {i} steps "
                f"(map {None if snap is None else snap.map_id}) -> {OUT_PATH.resolve()}",
                flush=True,
            )
            return

    print(f"no wild battle in {args.max_steps} steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the tool imports cleanly (no ROM run in CI)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('tools/capture_route101_in_battle.py').read())"`
Expected: no output (parses). Confirm `GRIND_STEP_FRAMES`, `GRIND_RELEASE_FRAMES` are exported from `env/orders.py` (they are, at `env/orders.py:47-48`).

- [ ] **Step 3: Write the gated ROM smoke**

Create `tests/test_battle_proof_survey_rom.py`:

```python
"""Deterministic live smoke: map_map survives a real route_101 wild battle.

Double-skips without the ROM or the states/route101_in_battle.state artifact
(produced by tools/capture_route101_in_battle.py). Loading that mid-battle state
means map_map's first iteration always sees an in-progress battle: the real
Fighter must win it and the survey must resume rather than abort.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from env.map_traveler import BATTLE_OUTCOMES

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = Path("states/route101_in_battle.state")
ROUTE_101 = (0, 16)


@pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not STATE.exists(), reason="states/route101_in_battle.state missing")
def test_map_map_survives_a_real_route101_battle() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.local_navigator import WallMap
    from env.map_explorer import map_map
    from env.map_memory import MapMemory
    from env.pokemon_env import PokemonEmeraldEnv
    from env.world_reader import WorldReader

    state = STATE.read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(ROM), [state], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()
    assert reader.in_battle()  # precondition: the artifact really is mid-battle

    fighter = PPO.load("checkpoints/fighter/ppo_fighter_final.zip", device="cpu")

    def predict(obs) -> int:
        return int(fighter.predict(obs, deterministic=True)[0])

    result = map_map(
        env.emulator, reader, MapMemory(), WallMap(), ROUTE_101,
        max_steps=300,
        move_type_fn=make_move_type_fn(env.emulator), predict=predict,
    )

    # The Fighter won and the survey resumed: not a battle outcome, and the
    # battle is actually resolved (assertion above is not vacuous).
    assert result not in BATTLE_OUTCOMES
    assert not reader.in_battle()
```

- [ ] **Step 4: Run the gated smoke (skips without the artifact)**

Run: `.venv/bin/python -m pytest tests/test_battle_proof_survey_rom.py -v`
Expected: SKIPPED (`states/route101_in_battle.state missing`) in a clean checkout. To make it load-bearing, run the capture tool once locally first:
`POKEMON_EMERALD_ROM=<rom> .venv/bin/python tools/capture_route101_in_battle.py`
then re-run — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/capture_route101_in_battle.py tests/test_battle_proof_survey_rom.py
git commit -m "test: deterministic live smoke — map_map survives a real route_101 battle"
```

---

## Final verification

- [ ] **Run the full suite (with ROM if available)**

Run: `POKEMON_EMERALD_ROM=<rom> .venv/bin/python -m pytest -q`
Expected: all pass (the new ROM smoke is load-bearing only if the artifact exists; otherwise skipped). Without the ROM, everything except gated ROM tests passes.

- [ ] **ruff clean**

Run: `.venv/bin/ruff check env/ tools/ tests/`
Expected: no findings.

## Self-review notes (for the executor)

- **DRY caveat:** the battle-RAM fake (`BattleExploreWorld`, `BattleWorldGrid`) is
  duplicated from `BattleNavWorld` in `tests/test_live_navigator.py`, matching the
  project's existing precedent (`GrassBattleWorld`, `BattleNavWorld` are each
  local copies). Do **not** hoist a shared cross-file test fixture unless a
  reviewer asks — cross-file test coupling has been rejected here before.
- **Type consistency:** `handle_battle_interruption(emulator, reader, move_type_fn, predict)`
  is called positionally in both `navigate_to` and `map_map`. `survey_world`
  forwards `move_type_fn`/`predict` as kwargs to both `travel_to` and `map_map`,
  matching their kwarg signatures.
- **Return-value contract:** `map_map` → `complete | budget_exhausted | left_map |
  battle_lost | battle_timeout | battle_interrupted`. `survey_world` records a
  battle outcome in `failed` as `travel:<outcome>` / `map:<outcome>` and returns
  early.
