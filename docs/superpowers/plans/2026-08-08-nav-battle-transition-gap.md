# Nav battle-transition detection gap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `navigate_grid` from recording false walls when a wild battle's intro or end-fade window freezes movement, which currently poisons the blocked-set and returns a false `unreachable` on grass-dense route_101.

**Architecture:** Add a raw `battle_starting()` signal (`gBattleTypeFlags != 0 && gBattleOutcome == 0`) to the reader; make `handle_battle_interruption` gate on it, wait out the intro until the opponent populates, play the battle, and wait out the end-fade so it always returns in overworld control; add an anti-poison guard in `navigate_grid` so a press that fails while a battle is starting is never recorded as a wall.

**Tech Stack:** Python 3.12, pytest, stable-baselines3 (Fighter, ROM smoke only), mGBA-backed `GbaEmulator`. Emerald (BPEF) RAM.

Spec: `docs/superpowers/specs/2026-08-08-nav-battle-transition-gap-design.md`

---

## File Structure

- `env/game_state.py` — `BattleReader.battle_starting()` (new method, 2 small RAM reads).
- `env/world_reader.py` — `WorldReader.battle_starting()` (delegation).
- `env/grid_navigator.py` — rewrite `handle_battle_interruption`; add `BATTLE_TRANSITION_SETTLE`; anti-poison guard in `navigate_grid`.
- Tests: `tests/test_battle_reader.py`, `tests/test_world_reader.py`, `tests/test_grid_navigator.py` (new battle-interruption + nav cases), `tests/test_nav_battle_gap_rom.py` (new, gated).
- Fake migration (additive stub only): `tests/test_grid_navigator.py`, `tests/test_grid_explorer.py`, `tests/test_orders.py`, `tests/test_map_traveler.py`, `tests/test_world_surveyor.py`.

---

## Task 1: `battle_starting()` on the reader

**Files:**
- Modify: `env/game_state.py` (add method to `BattleReader`, near `battle_state`)
- Modify: `env/world_reader.py` (add delegation after `in_battle`)
- Test: `tests/test_battle_reader.py`, `tests/test_world_reader.py`

- [ ] **Step 1: Write the failing test for `BattleReader.battle_starting()`**

Add to `tests/test_battle_reader.py`:

```python
from env.game_state import (
    BattleReader,
    GBATTLE_OUTCOME_ADDR,
    GBATTLE_TYPE_FLAGS_ADDR,
)


def _reader(flags: int, outcome: int) -> BattleReader:
    def read(addr: int, size: int) -> bytes:
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return flags.to_bytes(2, "little")
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([outcome])
        return bytes(size)
    return BattleReader(read)


def test_battle_starting_true_on_active_or_intro() -> None:
    # flags set, no terminal outcome yet: intro (opp not populated) or active.
    assert _reader(flags=0x0004, outcome=0).battle_starting() is True


def test_battle_starting_false_on_residual_flags() -> None:
    # Loaded post-battle savestate: flags linger but outcome is terminal.
    assert _reader(flags=0x0004, outcome=1).battle_starting() is False


def test_battle_starting_false_on_overworld() -> None:
    assert _reader(flags=0x0000, outcome=0).battle_starting() is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_battle_reader.py -k battle_starting -v`
Expected: FAIL with `AttributeError: 'BattleReader' object has no attribute 'battle_starting'`

- [ ] **Step 3: Implement `BattleReader.battle_starting()`**

In `env/game_state.py`, add inside `class BattleReader` (right after `battle_state`):

```python
    def battle_starting(self) -> bool:
        """True while battle flags are set with no terminal outcome yet.

        Covers the intro window — flags set before gBattleMons populates, so
        battle_state().in_battle is still False — and the active battle. Residual
        flags from a loaded post-battle savestate carry outcome != 0 and read
        False, so a freshly loaded savestate does not hang the navigator.
        """
        flags = self._u16(GBATTLE_TYPE_FLAGS_ADDR)
        outcome = self._u8(GBATTLE_OUTCOME_ADDR)
        return flags != 0 and outcome == 0
```

(`self._u16` / `self._u8` already exist on `BattleReader`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_battle_reader.py -k battle_starting -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing test for `WorldReader.battle_starting()`**

Add to `tests/test_world_reader.py`:

```python
def test_world_reader_battle_starting_delegates() -> None:
    from env.game_state import GBATTLE_OUTCOME_ADDR, GBATTLE_TYPE_FLAGS_ADDR

    def read(addr: int, size: int) -> bytes:
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return (0x0004).to_bytes(2, "little")
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([0])
        return bytes(size)

    assert WorldReader(read).battle_starting() is True
```

Ensure `WorldReader` is imported at the top of `tests/test_world_reader.py` (it already is if other tests build one; otherwise add `from env.world_reader import WorldReader`).

- [ ] **Step 6: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_world_reader.py -k battle_starting -v`
Expected: FAIL with `AttributeError: 'WorldReader' object has no attribute 'battle_starting'`

- [ ] **Step 7: Implement `WorldReader.battle_starting()`**

In `env/world_reader.py`, add right after `in_battle`:

```python
    def battle_starting(self) -> bool:
        """True during a battle's intro/active window (flags set, no outcome yet).

        Distinct from in_battle(): it is already True during the intro, before the
        opponent's HP populates. The navigator uses it to avoid mistaking a
        battle-frozen press for a wall.
        """
        return self._battle.battle_starting()
```

- [ ] **Step 8: Run both reader tests**

Run: `.venv/bin/pytest tests/test_battle_reader.py tests/test_world_reader.py -v`
Expected: PASS (all green)

- [ ] **Step 9: Commit**

```bash
git add env/game_state.py env/world_reader.py tests/test_battle_reader.py tests/test_world_reader.py
git commit -m "feat: battle_starting() raw signal for intro/active battle windows"
```

---

## Task 2: Migrate test fakes to expose `battle_starting()`

This is a pure additive change so the full suite stays green the moment Task 3 flips
`handle_battle_interruption` to call `reader.battle_starting()`. Every reader fake that
reaches `navigate_grid` (directly or via `travel_to`/`execute_order`/`map_map`/
`survey_world`/`explore_grid`) needs the method.

**Files (add one method to each fake class that already defines `in_battle`):**
- Modify: `tests/test_grid_navigator.py`, `tests/test_grid_explorer.py`,
  `tests/test_orders.py`, `tests/test_map_traveler.py`, `tests/test_world_surveyor.py`

**Rule:** for every fake class with a `def in_battle(self)`, add directly below it:

```python
    def battle_starting(self) -> bool:
        return <the exact same expression this fake's in_battle returns>
```

Concretely that is `return False` for fakes whose `in_battle` returns `False`, and
`return self._battle` (or `return self._in_battle`, matching the fake's field) for
fakes that model a battle. These fakes have no separate intro window, so mirroring
`in_battle` is faithful for them; the intro window is exercised only by the dedicated
fakes in Task 4.

- [ ] **Step 1: Locate every fake to migrate**

Run: `.venv/bin/python - <<'PY'
import subprocess
print(subprocess.run(["grep","-rn","def in_battle","tests"],capture_output=True,text=True).stdout)
PY`
Expected: the list of `def in_battle` sites in `tests/test_grid_navigator.py`,
`tests/test_grid_explorer.py`, `tests/test_orders.py` (6), `tests/test_map_traveler.py` (2),
`tests/test_world_surveyor.py` (2). For each, read its body and add the mirrored
`battle_starting`.

- [ ] **Step 2: Add the mirrored `battle_starting` under each `in_battle`**

Edit each fake per the Rule above. Example for `_LedgeWorld` in
`tests/test_grid_navigator.py` (its `in_battle` returns `False`):

```python
    def in_battle(self):
        return False

    def battle_starting(self):
        return False
```

Example for `FarmWorld` in `tests/test_orders.py` (its `in_battle` returns `self._battle`):

```python
    def in_battle(self) -> bool:
        return self._battle

    def battle_starting(self) -> bool:
        return self._battle
```

- [ ] **Step 3: Run the full suite (still on old handle behavior)**

Run: `.venv/bin/pytest -q`
Expected: PASS (same pass count as before + Task 1's new tests; no failures). The new
methods are unused for now.

- [ ] **Step 4: Commit**

```bash
git add tests/test_grid_navigator.py tests/test_grid_explorer.py tests/test_orders.py tests/test_map_traveler.py tests/test_world_surveyor.py
git commit -m "test: add battle_starting() stub to reader fakes reaching navigate_grid"
```

---

## Task 3: Rewrite `handle_battle_interruption` to guarantee overworld control

**Files:**
- Modify: `env/grid_navigator.py` (add `BATTLE_TRANSITION_SETTLE`; rewrite the function)
- Test: `tests/test_grid_navigator.py` (new unit tests, monkeypatched `play_battle`)

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/test_grid_navigator.py` (top-level, near the other helpers). These use a
tiny scripted reader/emulator double and monkeypatch `grid_navigator.play_battle` so the
test is pure:

```python
import env.grid_navigator as gn


class _ScriptedBattleReader:
    """Emulator+reader double scripting the intro/active/overworld sequence.

    battle_starting() is True from the start; in_battle() flips True only after
    `intro_steps` idle steps (models the opponent populating). A monkeypatched
    play_battle calls end() to clear both, so the fade-wait terminates.
    """

    def __init__(self, intro_steps: int = 2) -> None:
        self._intro_left = intro_steps
        self._battle = True
        self.steps = 0

    def step(self, _key: int, _frames: int) -> None:
        self.steps += 1
        if self._intro_left > 0:
            self._intro_left -= 1

    def battle_starting(self) -> bool:
        return self._battle

    def in_battle(self) -> bool:
        return self._battle and self._intro_left == 0

    def end(self) -> str:
        self._battle = False
        return "won"


def test_handle_waits_out_intro_then_wins(monkeypatch) -> None:
    r = _ScriptedBattleReader(intro_steps=2)
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: r.end())
    out = gn.handle_battle_interruption(r, r, move_type_fn=lambda m: 0, predict=lambda o: 0)
    assert out is None                       # won -> resume
    assert r.in_battle() is False            # overworld control on return
    assert r.battle_starting() is False


def test_handle_returns_none_when_no_battle() -> None:
    class _NoBattle:
        def battle_starting(self) -> bool:
            return False
        def in_battle(self) -> bool:
            return False
    r = _NoBattle()
    assert gn.handle_battle_interruption(r, r, None, None) is None


def test_handle_reports_interrupted_without_fighter() -> None:
    r = _ScriptedBattleReader(intro_steps=0)
    assert gn.handle_battle_interruption(r, r, None, None) == "battle_interrupted"


def test_handle_reports_loss(monkeypatch) -> None:
    r = _ScriptedBattleReader(intro_steps=0)
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: "lost")
    out = gn.handle_battle_interruption(r, r, move_type_fn=lambda m: 0, predict=lambda o: 0)
    assert out == "battle_lost"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_grid_navigator.py -k "handle_" -v`
Expected: FAIL — `test_handle_waits_out_intro_then_wins` fails because the current
function gates on `in_battle()` (False during intro) and returns None without playing;
`test_handle_reports_interrupted_without_fighter` fails for the same reason.

- [ ] **Step 3: Add the constant and rewrite the function**

In `env/grid_navigator.py`, add near the other timing constants (after `SETTLE_TRIES = 4`):

```python
BATTLE_TRANSITION_SETTLE = 8   # bounded idle steps to wait out a battle intro/fade
```

Replace the whole `handle_battle_interruption` body with:

```python
def handle_battle_interruption(
    emulator: Any, reader: Any, move_type_fn: Any, predict: Any
) -> str | None:
    """Play any live wild battle and return only in overworld control.

    Gating on battle_starting() (flags set, no terminal outcome) catches the intro
    window where in_battle() is still False because the opponent has not populated.
    We idle until in_battle() confirms, play the battle, then idle out the end-fade
    so the caller never presses into a frozen game.

    None when there is no battle (or it was won) so the caller resumes; a terminal
    outcome otherwise: "battle_interrupted" (no Fighter), "battle_lost",
    "battle_timeout".
    """
    if not reader.battle_starting():
        return None
    for _ in range(BATTLE_TRANSITION_SETTLE):
        if reader.in_battle():
            break
        emulator.step(0, RELEASE_FRAMES)
    if not reader.in_battle():
        return None   # flags set but never became a real battle: not ours
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result != "won":
        return "battle_lost" if result == "lost" else "battle_timeout"
    for _ in range(BATTLE_TRANSITION_SETTLE):
        if not reader.battle_starting() and not reader.in_battle():
            break
        emulator.step(0, RELEASE_FRAMES)
    return None
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/pytest tests/test_grid_navigator.py -k "handle_" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite (regression on all navigate_grid callers)**

Run: `.venv/bin/pytest -q`
Expected: PASS. Fakes migrated in Task 2 supply `battle_starting()`, so the top-of-loop
call resolves everywhere. Battle-modelling fakes (FarmWorld etc.) mirror in_battle so
the intro/fade waits are no-ops (0 iterations) for them.

- [ ] **Step 6: Commit**

```bash
git add env/grid_navigator.py tests/test_grid_navigator.py
git commit -m "fix: handle_battle_interruption waits out intro+fade, gates on battle_starting"
```

---

## Task 4: Anti-poison guard in `navigate_grid`

**Files:**
- Modify: `env/grid_navigator.py` (the `if outcome == "blocked":` branch in `navigate_grid`)
- Test: `tests/test_grid_navigator.py` (intro-race, fade, genuine-wall)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_grid_navigator.py`. `_TrapGrassWorld` models the real bug: at the
trap cell the north-going press is frozen because a battle starts; once the battle is
played (monkeypatched to `won`, which clears the battle), the same press moves. A
genuine wall (no battle context) must still poison and reroute.

```python
class _TrapGrassWorld:
    """3x1 vertical corridor: start (0,2) -> target (0,0), all FREE.

    Stepping from (0,1) to (0,0) is 'frozen' by a wild battle: the first press at
    (0,1) does not move and battle_starting is True. A monkeypatched play_battle
    calls clear() so the retry moves. Tracks pressed directions to prove no poison.
    """

    def __init__(self) -> None:
        self._pos = (0, 2)
        self._battle = False
        self._armed = True          # next northward press at (0,1) triggers a battle
        self._grid = _FakeGridReader([[_TK.FREE], [_TK.FREE], [_TK.FREE]])

    def snapshot(self):
        return WorldSnapshot(map_id=(0, 16), pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return self._battle

    def battle_starting(self):
        return self._battle

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    def clear(self) -> str:
        self._battle = False
        return "won"

    def step(self, key, _frames):
        from emulator import buttons
        if key != buttons.KEY_UP:
            return
        if self._battle:
            return                              # frozen mid-battle
        if self._pos == (0, 1) and self._armed:
            self._battle = True                 # encounter fires, no move
            self._armed = False
            return
        self._pos = (self._pos[0], self._pos[1] - 1)


def test_navigate_grid_recovers_from_battle_frozen_press(monkeypatch) -> None:
    w = _TrapGrassWorld()
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: w.clear())
    result = gn.navigate_grid(
        w, w, target=(0, 0), max_steps=50,
        move_type_fn=lambda m: 0, predict=lambda o: 0,
    )
    assert result == "arrived"
    assert w._pos == (0, 0)


class _WalledWorld:
    """Target (0,0) is unreachable: (0,1)->(0,0) is a WALL, no battle ever."""

    def __init__(self) -> None:
        self._pos = (0, 2)
        self._grid = _FakeGridReader([[_TK.WALL], [_TK.FREE], [_TK.FREE]])

    def snapshot(self):
        return WorldSnapshot(map_id=(0, 16), pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return False

    def battle_starting(self):
        return False

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    def step(self, key, _frames):
        from emulator import buttons
        if key == buttons.KEY_UP and self._pos == (0, 2):
            self._pos = (0, 1)      # can reach (0,1); (0,0) is a wall


def test_navigate_grid_still_reports_genuine_wall() -> None:
    w = _WalledWorld()
    result = gn.navigate_grid(w, w, target=(0, 0), max_steps=30)
    assert result == "unreachable"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_grid_navigator.py -k "recovers_from_battle or genuine_wall" -v`
Expected: `test_navigate_grid_recovers_from_battle_frozen_press` FAILS — the current
`blocked` branch adds `((0,1),"up")` to the blocked set, `plan_path_grid` returns None,
`navigate_grid` returns `unreachable` (not `arrived`). `test_...genuine_wall` PASSES
already (documents preserved behavior).

- [ ] **Step 3: Add the anti-poison guard**

In `env/grid_navigator.py` `navigate_grid`, replace the final blocked branch:

```python
        if outcome == "blocked":
            blocked.add((before.pos, direction))   # transient: NPC / surprise
```

with:

```python
        if outcome == "blocked":
            # A press can fail because a wild battle just started on this step
            # (grass), not because of a wall. Consume the battle and re-plan
            # instead of poisoning the tile as unreachable.
            if reader.battle_starting() or reader.in_battle():
                battle = handle_battle_interruption(
                    emulator, reader, move_type_fn, predict
                )
                if battle is not None:
                    return battle
                continue
            blocked.add((before.pos, direction))   # genuine wall / NPC
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_grid_navigator.py -k "recovers_from_battle or genuine_wall" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all green).

- [ ] **Step 6: Run ruff**

Run: `.venv/bin/ruff check env/ tests/`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add env/grid_navigator.py tests/test_grid_navigator.py
git commit -m "fix: navigate_grid consumes battle-frozen presses instead of poisoning the blocked set"
```

---

## Task 5: Load-bearing ROM smoke

**Files:**
- Create: `tests/test_nav_battle_gap_rom.py`

- [ ] **Step 1: Write the gated ROM smoke**

Create `tests/test_nav_battle_gap_rom.py`:

```python
"""Load-bearing ROM smoke: the battle-transition fix un-poisons the north nav.

Before the fix, navigate_grid to (11,0) from post_starter.state returns
'unreachable' because wild-battle intro/fade windows at (11,10) get recorded as
walls. After the fix it must NOT return 'unreachable' and the player must cross
north of the trap band (y < 10). Reaching (11,0) exactly is not required — the
Oldale north-exit geometry is a separate, out-of-scope gap.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = Path("states/post_starter.state")
CKPT = Path("checkpoints/fighter/ppo_fighter_final.zip")


@pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not STATE.exists(), reason="post_starter.state missing")
@pytest.mark.skipif(not CKPT.exists(), reason="Fighter checkpoint missing")
def test_north_nav_not_poisoned_by_battle_transitions() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.grid_navigator import navigate_grid
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)

    reader = WorldReader(emu.read_bytes)
    model = PPO.load(str(CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    result = navigate_grid(
        emu, reader, target=(11, 0), max_steps=300,
        move_type_fn=make_move_type_fn(emu), predict=predict,
    )
    end = reader.snapshot()
    assert result != "unreachable", f"still poisoned: {result} at {end and end.pos}"
    assert end is not None and end.pos[1] < 10, f"did not cross north: {end.pos}"
```

- [ ] **Step 2: Run the smoke with the ROM env set**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest tests/test_nav_battle_gap_rom.py -v`
Expected: PASS (branch executed — result != "unreachable" and pos.y < 10). If it fails,
STOP and diagnose: print the `result` and final pos; the fix or the target may need
adjustment (do not weaken the assertion to force a pass).

- [ ] **Step 3: Confirm it skips cleanly without the ROM**

Run: `.venv/bin/pytest tests/test_nav_battle_gap_rom.py -v`
Expected: SKIPPED (POKEMON_EMERALD_ROM not set).

- [ ] **Step 4: Commit**

```bash
git add tests/test_nav_battle_gap_rom.py
git commit -m "test: load-bearing ROM smoke — north nav no longer poisoned by battle windows"
```

---

## Task 6: Cleanup + verification

**Files:**
- Delete: throwaway probes under `tools/` created during the investigation.
- Modify: `/Users/_eloi/.claude/projects/-Users--eloi-Projets-Emu/memory/rival-reach-findings.md` (mark FIXED)

- [ ] **Step 1: Remove the throwaway investigation probes**

```bash
git rm -f tools/probe_stuck_cell.py tools/probe_frozen_at_waypoint.py \
         tools/probe_nav_trace.py tools/probe_regrid_north.py \
         tools/probe_astar_static.py tools/probe_direct_crossing.py \
         tools/probe_battle_attrition.py 2>/dev/null || true
```

(Only remove files that exist; some may be untracked — delete those with plain `rm`.
Keep `tools/dump_map_grid.py` and any non-throwaway tooling.)

- [ ] **Step 2: Full suite + ruff, final gate**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check env/ tests/`
Expected: all green, no ruff errors.

- [ ] **Step 3: Run the ROM smoke once more end-to-end**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest tests/test_nav_battle_gap_rom.py -v`
Expected: PASS.

- [ ] **Step 4: Update the findings memory to FIXED**

Edit `rival-reach-findings.md`: under the "ROOT CAUSE CONFIRMED" section, change the
"Proposed fix (PENDING)" note to "FIXED on branch fix/nav-battle-transition-gap" with
the commit range, and note the ROM smoke is load-bearing and green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove investigation probes; mark nav battle-gap fixed"
```

---

## Self-Review

**Spec coverage:**
- `battle_starting()` on BattleReader + WorldReader → Task 1. ✓
- `handle_battle_interruption` intro-wait + fade-wait + gate → Task 3. ✓
- Anti-poison guard in `navigate_grid` → Task 4. ✓
- Fake migration (all readers reaching navigate_grid) → Task 2. ✓
- Pure tests (battle_starting truth table, delegation, intro race, fade, genuine wall,
  handle units) → Tasks 1, 3, 4. ✓
- Load-bearing ROM smoke → Task 5. ✓
- Known minor race (stale-outcome/new-flags) → documented in spec; no task needed
  (self-heals within one loop iteration; guard only poisons with no battle context). ✓
- Non-goals (no play_battle/trainer change, no in_battle() change, no grid/A* change) →
  respected; Tasks touch only the listed methods. ✓

**Placeholder scan:** every code step shows full code; no TBD/TODO. Task 2 uses a
mechanical mirror rule with both concrete forms shown (`return False` /
`return self._battle`) plus the exact file list — not a "similar to" placeholder. ✓

**Type consistency:** `battle_starting()` signature identical across `BattleReader`,
`WorldReader`, and every fake. `handle_battle_interruption` signature unchanged
`(emulator, reader, move_type_fn, predict) -> str | None`. `BATTLE_TRANSITION_SETTLE`
defined once (Task 3) and only referenced there. `navigate_grid` guard reuses the
existing `handle_battle_interruption` and `blocked` set. ✓
