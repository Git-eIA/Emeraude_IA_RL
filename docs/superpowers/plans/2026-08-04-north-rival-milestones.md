# North-rival milestones (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Explorer a north-only reward chain (`reach_oldale` → `reach_route_103` → `beat_rival`) that resets from `states/post_starter.state`, hooks the Fighter into the env to resolve wild + trainer battles mid-episode, and detects the rival win from a won trainer battle on route_103 — delivering machinery + a gated ROM smoke + a manual runbook (the multi-hour run is a documented human follow-up).

**Architecture:** Three additive changes, each behind a default that preserves the existing intro path: (1) `env/milestones.py` gains `EnvContext` + `Milestone.env_condition` + `route103_milestones()`; (2) `env/game_state.py` exposes `BattleReader.is_trainer_battle()` via the `BATTLE_TYPE_TRAINER = 0x0008` bit; (3) `env/pokemon_env.py` gains injectable `milestones`/`move_type_fn`/`predict`, a `BattleReader`, a `_rival_beaten` latch, and a battle hook in `step` that dispatches `play_battle`/`play_trainer_battle` before the frame append.

**Tech Stack:** Python 3.12, Gymnasium, SB3 PPO (injected via `move_type_fn`/`predict` so the env stays SB3-free), pytest, ruff (line-length 100). Pokémon Emerald FR ROM (BPEF).

---

## File Structure

- **Modify** `env/milestones.py` — add `EnvContext` dataclass, `Milestone.env_condition` field, `OLDALE`/`ROUTE_103` constants, `route103_milestones()`, and the `ctx` argument on `MilestoneTracker.update`. Responsibility unchanged: define named one-time rewards + track which have fired.
- **Modify** `env/game_state.py` — add `BATTLE_TYPE_TRAINER` constant + `BattleReader.is_trainer_battle()`. Responsibility unchanged: read game RAM into typed snapshots.
- **Modify** `env/pokemon_env.py` — constructor params + `_battle_reader` + `_rival_beaten` + battle hook + ctx-passing milestone update. Responsibility unchanged: the Gymnasium env.
- **Test** `tests/test_milestones.py`, `tests/test_game_state.py`, `tests/test_env.py` (pure) + **create** `tests/test_north_rival_milestones_rom.py` (gated).

Each task is one file's change + its tests, self-contained.

---

### Task 1: `env/milestones.py` — EnvContext + env_condition + route103 table

**Files:**
- Modify: `env/milestones.py`
- Test: `tests/test_milestones.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_milestones.py`. NOTE: `make_state`'s default is `map_num=10` = OLDALE `(0, 10)` — so `reach_oldale` fires on a default state. route103 tests that must NOT be on Oldale pass an explicit `map_num`.

```python
from env.milestones import EnvContext, route103_milestones


def test_env_condition_blocks_fire_when_ctx_false():
    m = Milestone(
        "beat_rival",
        lambda s: (s.map_group, s.map_num) == (0, 18),
        100.0,
        terminal=True,
        env_condition=lambda ctx: ctx.rival_beaten,
    )
    tracker = MilestoneTracker((m,))
    on_route103 = make_state(map_group=0, map_num=18)
    # Position holds but ctx says the rival is not beaten yet.
    assert tracker.update(on_route103, EnvContext(rival_beaten=False)) == (0.0, False)
    assert tracker.fired == set()


def test_env_condition_fires_when_both_hold():
    m = Milestone(
        "beat_rival",
        lambda s: (s.map_group, s.map_num) == (0, 18),
        100.0,
        terminal=True,
        env_condition=lambda ctx: ctx.rival_beaten,
    )
    tracker = MilestoneTracker((m,))
    on_route103 = make_state(map_group=0, map_num=18)
    assert tracker.update(on_route103, EnvContext(rival_beaten=True)) == (100.0, True)
    assert tracker.fired == {"beat_rival"}


def test_starter_milestones_unaffected_by_passed_ctx():
    tracker = MilestoneTracker(starter_milestones())
    # A ctx argument must not change the behavior of pure-PlayerState milestones.
    state = make_state(map_group=0, map_num=9, x=5, y=0, clock_set=True)
    without = MilestoneTracker(starter_milestones()).update(state)
    with_ctx = tracker.update(state, EnvContext(rival_beaten=True))
    assert with_ctx == without


def test_route103_milestones_shape():
    table = route103_milestones()
    names = [m.name for m in table]
    assert names == ["reach_oldale", "reach_route_103", "beat_rival"]
    points = {m.name: m.points for m in table}
    assert points == {"reach_oldale": 30.0, "reach_route_103": 40.0, "beat_rival": 100.0}
    beat = next(m for m in table if m.name == "beat_rival")
    assert beat.terminal is True
    assert beat.env_condition is not None
    assert all(m.env_condition is None for m in table if m.name != "beat_rival")


def test_reach_oldale_fires_on_oldale_only():
    tracker = MilestoneTracker(route103_milestones())
    # route_101, not Oldale: nothing fires.
    assert tracker.update(make_state(map_group=0, map_num=16)) == (0.0, False)
    assert tracker.fired == set()
    # Oldale: reach_oldale fires.
    assert tracker.update(make_state(map_group=0, map_num=10)) == (30.0, False)
    assert tracker.fired == {"reach_oldale"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_milestones.py -q`
Expected: FAIL with `ImportError: cannot import name 'EnvContext'` (and `route103_milestones`).

- [ ] **Step 3: Implement the additive changes**

In `env/milestones.py`, add the `EnvContext` dataclass above `Milestone`, add the `env_condition` field to `Milestone`, add the two constants near the existing map constants, add `route103_milestones()`, and extend `MilestoneTracker.update` with the `ctx` argument.

```python
@dataclass(frozen=True)
class EnvContext:
    """Env-side signals not present in PlayerState RAM."""

    rival_beaten: bool = False


@dataclass(frozen=True)
class Milestone:
    name: str
    condition: Callable[[PlayerState], bool]
    points: float
    terminal: bool = False
    # Additive: when set, BOTH condition(state) AND env_condition(ctx) must hold.
    # Default None keeps the intro lambdas untouched (pure PlayerState).
    env_condition: Callable[[EnvContext], bool] | None = None
```

Add constants (near `LITTLEROOT`/`ROUTE_101`):

```python
OLDALE = (0, 10)  # pret ordinal; pin live on first north transition
ROUTE_103 = (0, 18)  # corroborated by orders.py DESTINATIONS; pin live
```

Add the table:

```python
def route103_milestones() -> tuple[Milestone, ...]:
    """Phase 1 north push: leave route_101, reach route_103, beat the rival.

    Reset from states/post_starter.state (free-roam route_101, lv5). Does NOT
    reuse starter_milestones(): starter_obtained (party_count>=1, terminal)
    would fire at t=0 and end the episode instantly.
    """
    return (
        Milestone(
            "reach_oldale",
            lambda s: (s.map_group, s.map_num) == OLDALE,
            30.0,
        ),
        Milestone(
            "reach_route_103",
            lambda s: (s.map_group, s.map_num) == ROUTE_103,
            40.0,
        ),
        Milestone(
            "beat_rival",
            lambda s: (s.map_group, s.map_num) == ROUTE_103,
            100.0,
            terminal=True,
            env_condition=lambda ctx: ctx.rival_beaten,
        ),
    )
```

Extend `update` (add the `ctx` param + the env_condition gate):

```python
def update(
    self, state: PlayerState | None, ctx: EnvContext | None = None,
) -> tuple[float, bool]:
    if state is None:
        return 0.0, False
    ctx = ctx or EnvContext()
    reward = 0.0
    terminated = False
    for m in self._milestones:
        if m.name in self._fired or not m.condition(state):
            continue
        if m.env_condition is not None and not m.env_condition(ctx):
            continue
        self._fired.add(m.name)
        reward += m.points
        terminated = terminated or m.terminal
    return reward, terminated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_milestones.py -q`
Expected: PASS (all existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add env/milestones.py tests/test_milestones.py
git commit -m "$(cat <<'EOF'
feat: EnvContext + Milestone.env_condition + route103_milestones table

Additive north-only reward chain (reach_oldale/reach_route_103/beat_rival).
env_condition gates a milestone on env-side signals (rival_beaten) not present
in PlayerState RAM; MilestoneTracker.update gains a defaulted ctx arg so every
starter_milestones() caller is byte-identical.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `env/game_state.py` — expose the trainer bit

**Files:**
- Modify: `env/game_state.py`
- Test: `tests/test_battle_reader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_battle_reader.py`. The `_make_reader` helper builds crafted RAM; it accepts a `flags` word. Confirm the helper's flags parameter name by reading the file head first — if it only sets a default trainer/wild flags word, extend it to accept `flags=` (a keyword defaulting to the current wild value). The tests below assume `_make_reader(..., flags=...)` writes `GBATTLE_TYPE_FLAGS_ADDR`.

```python
from env.game_state import BATTLE_TYPE_TRAINER


def test_is_trainer_battle_true_when_trainer_bit_set():
    reader = _make_reader(in_battle=True, flags=BATTLE_TYPE_TRAINER | 0x0001)
    assert reader.is_trainer_battle() is True


def test_is_trainer_battle_false_for_wild_flags():
    # A wild battle: any nonzero flags word WITHOUT the 0x0008 trainer bit.
    reader = _make_reader(in_battle=True, flags=0x0001)
    assert reader.is_trainer_battle() is False
```

If `_make_reader` does not already thread a flags word into `GBATTLE_TYPE_FLAGS_ADDR`, add that plumbing (a `flags: int = 0x0001` kwarg that the crafted-bytes builder writes at `GBATTLE_TYPE_FLAGS_ADDR` as u16 LE). Keep the default equal to whatever the existing `in_battle=True` path already produced so existing tests are unaffected.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_battle_reader.py -q`
Expected: FAIL with `ImportError: cannot import name 'BATTLE_TYPE_TRAINER'`.

- [ ] **Step 3: Implement**

In `env/game_state.py`, add the constant near the other `GBATTLE_*` battle constants:

```python
BATTLE_TYPE_TRAINER = 0x0008  # gBattleTypeFlags bit distinguishing trainer battles
```

Add the method to `BattleReader`:

```python
def is_trainer_battle(self) -> bool:
    """True when the current battle is against a trainer (not wild)."""
    return bool(self._u16(GBATTLE_TYPE_FLAGS_ADDR) & BATTLE_TYPE_TRAINER)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_battle_reader.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add env/game_state.py tests/test_battle_reader.py
git commit -m "$(cat <<'EOF'
feat: BattleReader.is_trainer_battle via BATTLE_TYPE_TRAINER (0x0008)

The env needs to dispatch play_battle (wild) vs play_trainer_battle (trainer);
gBattleTypeFlags already read for in_battle, now the 0x0008 bit is surfaced.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `env/pokemon_env.py` — Fighter hook + injectable milestones + rival latch

**Files:**
- Modify: `env/pokemon_env.py`
- Test: `tests/test_env.py`, `tests/conftest.py` (extend `FakeEmulator` if needed)

- [ ] **Step 1: Write the failing tests**

First read `tests/conftest.py` to see what `FakeEmulator` exposes (party_count, party_levels, read_bytes, step). The tests need the fake emulator to (a) report `in_battle`/`is_trainer_battle` via `read_bytes`, or (b) be simple enough that we drive the battle path with injected `move_type_fn`/`predict` and a `_battle_reader` stub. Since `_battle_reader = BattleReader(emulator.read_bytes)`, the cleanest fake drives `read_bytes` to return an in-trainer-battle RAM snapshot for one step, then an out-of-battle snapshot.

Given that plumbing crafted battle RAM into `FakeEmulator` is heavy, use a **seam**: the tests below monkeypatch the env's `_battle_reader` and the two battle-player functions. Append to `tests/test_env.py`.

```python
from env import pokemon_env as pe
from env.milestones import route103_milestones


class _StubBattleReader:
    """Reports a scripted battle sequence for the env hook."""

    def __init__(self, script):
        # script: list of (in_battle, is_trainer) tuples consumed per call.
        self._script = list(script)
        self._i = 0

    def battle_state(self):
        in_battle, _ = self._script[min(self._i, len(self._script) - 1)]

        class _BS:
            pass

        bs = _BS()
        bs.in_battle = in_battle
        return bs

    def is_trainer_battle(self):
        _, is_trainer = self._script[min(self._i, len(self._script) - 1)]
        return is_trainer

    def advance(self):
        self._i += 1


def test_injected_milestones_swaps_the_table():
    env = pe.PokemonEmeraldEnv(
        FakeEmulator(), initial_states=[b"fake"], max_steps=50,
        milestones=route103_milestones(),
    )
    env.reset()
    assert [m.name for m in env._milestones._milestones] == [
        "reach_oldale", "reach_route_103", "beat_rival",
    ]


def test_default_ctor_uses_starter_milestones():
    env = pe.PokemonEmeraldEnv(FakeEmulator(), initial_states=[b"fake"], max_steps=50)
    env.reset()
    names = {m.name for m in env._milestones._milestones}
    assert "starter_obtained" in names


def test_trainer_win_on_route103_latches_rival_beaten(monkeypatch):
    emu = FakeEmulator()
    emu.map_group, emu.map_num = 0, 18  # route_103

    def fake_trainer_battle(*a, **k):
        return "won"

    monkeypatch.setattr(pe, "play_trainer_battle", fake_trainer_battle)
    monkeypatch.setattr(pe, "play_battle", lambda *a, **k: "won")

    env = pe.PokemonEmeraldEnv(
        emu, initial_states=[b"fake"], max_steps=50,
        milestones=route103_milestones(),
        move_type_fn=lambda t: 0, predict=lambda obs: 0,
    )
    env.reset()
    # in trainer battle this step, out next step.
    env._battle_reader = _StubBattleReader([(True, True), (False, False)])
    env.step(0)
    assert env._rival_beaten is True


def test_wild_battle_does_not_latch_rival_beaten(monkeypatch):
    emu = FakeEmulator()
    emu.map_group, emu.map_num = 0, 18  # route_103, but wild -> no latch

    monkeypatch.setattr(pe, "play_trainer_battle", lambda *a, **k: "won")
    monkeypatch.setattr(pe, "play_battle", lambda *a, **k: "won")

    env = pe.PokemonEmeraldEnv(
        emu, initial_states=[b"fake"], max_steps=50,
        milestones=route103_milestones(),
        move_type_fn=lambda t: 0, predict=lambda obs: 0,
    )
    env.reset()
    env._battle_reader = _StubBattleReader([(True, False), (False, False)])
    env.step(0)
    assert env._rival_beaten is False
```

The env must read the map at battle start via `self._reader.player_state()`. `FakeEmulator` must let `_reader.player_state()` return a state whose `(map_group, map_num)` reflects `emu.map_group`/`emu.map_num`. Read conftest; if `FakeEmulator.read_bytes` already encodes map from settable attributes, use those attribute names. If not, extend `FakeEmulator` minimally so `player_state()` returns the set map (this is the same mechanism existing map/pos env tests rely on — reuse it; do not invent a parallel path).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_env.py -q`
Expected: FAIL — `PokemonEmeraldEnv.__init__` got an unexpected keyword argument `milestones` (and `_rival_beaten` missing).

- [ ] **Step 3: Implement**

In `env/pokemon_env.py`:

Update imports:

```python
from env.battle_player import play_battle, play_trainer_battle
from env.game_state import BattleReader, EmeraldReader, PlayerState
from env.milestones import ROUTE_103, EnvContext, MilestoneTracker, starter_milestones
```

Constructor — add the three params, the battle reader, and the latch:

```python
def __init__(
    self,
    emulator,
    initial_states: list[bytes],
    max_steps: int = 2048,
    milestones=None,
    move_type_fn=None,
    predict=None,
) -> None:
    ...
    self._reader = EmeraldReader(emulator.read_bytes)
    self._battle_reader = BattleReader(emulator.read_bytes)
    self._milestones = MilestoneTracker(milestones or starter_milestones())
    self._move_type_fn = move_type_fn
    self._predict = predict
    self._rival_beaten: bool = False
    ...
```

`reset()` — reset the latch:

```python
self._rival_beaten = False
```

`step()` — insert the hook right after `self.emulator.step(keys, FRAMES_PER_ACTION)` and BEFORE `self._frames.append(...)`:

```python
self.emulator.step(keys, FRAMES_PER_ACTION)
# battle hook: resolve any in-progress battle before the frame/state read, so
# this step's frame + player_state reflect the post-battle overworld.
if self._move_type_fn is not None and self._predict is not None:
    bs = self._battle_reader.battle_state()
    if bs.in_battle:
        # G3: read the map at battle START (battles never warp the player).
        pre = self._reader.player_state()
        on_route103 = pre is not None and (pre.map_group, pre.map_num) == ROUTE_103
        if self._battle_reader.is_trainer_battle():
            result = play_trainer_battle(self.emulator, self._move_type_fn, self._predict)
            if result == "won" and on_route103:
                self._rival_beaten = True
        else:
            play_battle(self.emulator, self._move_type_fn, self._predict)
self._frames.append(self._current_frame())
```

Milestone update — pass the context:

```python
milestone_reward, terminated = self._milestones.update(
    state, EnvContext(rival_beaten=self._rival_beaten),
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_env.py -q`
Expected: PASS (existing + 4 new).

- [ ] **Step 5: Run the full pure suite (no ROM)**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (ROM smokes skip without `POKEMON_EMERALD_ROM`).

- [ ] **Step 6: Commit**

```bash
git add env/pokemon_env.py tests/test_env.py tests/conftest.py
git commit -m "$(cat <<'EOF'
feat: Fighter battle hook + injectable milestones + rival-beaten latch in env

step() resolves any in-progress battle before the frame append: is_trainer_battle
dispatches play_trainer_battle (latching _rival_beaten on a won trainer fight on
route_103) vs play_battle for wilds. milestones/move_type_fn/predict are additive
ctor params (defaults preserve every existing caller: starter_milestones, no hook).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Gated ROM smoke + runbook

**Files:**
- Create: `tests/test_north_rival_milestones_rom.py`
- Modify: docstring in `env/milestones.py` (point to the runbook) — optional, keep minimal.

- [ ] **Step 1: Write the gated smoke**

Triple-skip: ROM unset, Fighter checkpoint missing, `states/trainer_battle.state` missing. SB3/torch imported inside the test body (not at collect). Read `tests/test_battle_player_rom.py` first to copy the exact Fighter-wrapping pattern (`make_move_type_fn`, `PPO.load`, the deterministic `predict`).

```python
"""Gated ROM smoke: the env dispatches the Fighter on a real trainer battle and
(when the captured state is on route_103) latches beat_rival end-to-end.

Triple-skip: POKEMON_EMERALD_ROM unset | Fighter checkpoint missing |
states/trainer_battle.state missing (the route_103 rival capture is deferred).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.gba import GbaEmulator
from env.game_state import BattleReader, EmeraldReader
from env.milestones import ROUTE_103, route103_milestones
from env.pokemon_env import PokemonEmeraldEnv

_ROM = os.environ.get("POKEMON_EMERALD_ROM")
_CKPT = Path("checkpoints/fighter/ppo_fighter_final.zip")
_STATE = Path("states/trainer_battle.state")

pytestmark = pytest.mark.skipif(
    _ROM is None or not _CKPT.exists() or not _STATE.exists(),
    reason="requires ROM + Fighter checkpoint + states/trainer_battle.state",
)


def _fighter(emu):
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn

    model = PPO.load(str(_CKPT), device="cpu")

    def predict(obs):
        return int(model.predict(obs, deterministic=True)[0])

    return make_move_type_fn(emu), predict


def test_env_dispatches_fighter_on_real_trainer_battle():
    state = _STATE.read_bytes()
    emu = GbaEmulator(_ROM)
    move_type_fn, predict = _fighter(emu)
    env = PokemonEmeraldEnv(
        emu, initial_states=[state], max_steps=64,
        milestones=route103_milestones(),
        move_type_fn=move_type_fn, predict=predict,
    )
    env.reset()

    reader = BattleReader(emu.read_bytes)
    assert reader.battle_state().in_battle, "precondition: state must be mid-battle"
    assert reader.is_trainer_battle(), "precondition: must be a TRAINER battle"

    world = EmeraldReader(emu.read_bytes)
    pre = world.player_state()
    on_route103 = pre is not None and (pre.map_group, pre.map_num) == ROUTE_103

    env.step(0)

    # Load-bearing regardless of capture location: the battle was dispatched and
    # resolved (play_trainer_battle ran, not play_battle which would mis-handle
    # send-outs). in_battle is False after the step.
    assert not reader.battle_state().in_battle

    if on_route103:
        assert env._rival_beaten is True
        assert "beat_rival" in env._milestones.fired
```

- [ ] **Step 2: Run the smoke (skips without artifact)**

Run: `export POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba"; .venv/bin/python -m pytest tests/test_north_rival_milestones_rom.py -q`
Expected: SKIP (`states/trainer_battle.state` absent — route_103 capture deferred). The test is load-bearing the moment that artifact exists.

- [ ] **Step 3: Verify collection is clean**

Run: `.venv/bin/python -m pytest tests/test_north_rival_milestones_rom.py --collect-only -q`
Expected: 1 test collected, no import errors (SB3 imports are inside the body).

- [ ] **Step 4: Append the runbook to the spec**

The runbook already lives in the spec (`docs/superpowers/specs/2026-08-04-north-rival-milestones-design.md`, "Manual runbook" section, item G8 = dump savestate at first `beat_rival`). No new file. If any runbook detail drifted during implementation (e.g. the actual injected param names), reconcile the spec's runbook with the shipped constructor signature.

- [ ] **Step 5: Commit**

```bash
git add tests/test_north_rival_milestones_rom.py
git commit -m "$(cat <<'EOF'
test: gated ROM smoke for the north-rival env hook

Triple-skip (ROM|Fighter ckpt|states/trainer_battle.state). Load-bearing half:
the env dispatches play_trainer_battle on a real trainer battle (in_battle False
after one step). Conditional-on-route_103 half asserts the _rival_beaten latch +
beat_rival fired; skipped when the capture is elsewhere (route_103 still deferred).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Full-suite verification + ruff

**Files:** none (verification only)

- [ ] **Step 1: Full pure suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, ROM smokes skip.

- [ ] **Step 2: Full suite with ROM**

Run: `export POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba"; .venv/bin/python -m pytest -q`
Expected: all pass; the new smoke skips (artifact absent), every pre-existing ROM smoke still passes.

- [ ] **Step 3: Lint**

Run: `.venv/bin/ruff check env/milestones.py env/game_state.py env/pokemon_env.py tests/test_milestones.py tests/test_battle_reader.py tests/test_env.py tests/test_north_rival_milestones_rom.py`
Expected: `All checks passed!`

- [ ] **Step 4: Report which tests ran and which were skipped**

State explicitly: pure suite green, ROM suite green with the new smoke + `trainer_battle.state`-dependent smokes skipped (documented deferral). No commit (verification task).

---

## Self-Review Notes

- **Spec coverage:** Task 1 = Change 1 (milestones); Task 2 = Change 2 (game_state trainer bit); Task 3 = Change 3 (env hook + injectable milestones + latch); Task 4 = smoke + runbook; Task 5 = verification. All spec sections mapped.
- **Type consistency:** `route103_milestones()` name used identically in Tasks 1/3/4. `EnvContext(rival_beaten=...)` field name consistent. `is_trainer_battle()` consistent. `_rival_beaten` / `_battle_reader` / `_move_type_fn` / `_predict` attribute names consistent across Task 3 and Task 4.
- **Known implementation dependency (flagged, not a placeholder):** Task 2 Step 1 and Task 3 Step 1 require reading `tests/test_battle_reader.py` / `tests/conftest.py` first to reuse the existing crafted-RAM / FakeEmulator map mechanism rather than inventing a parallel one. This is real integration work, not a gap — the exact seam depends on the helper's current signature.
- **make_state Oldale trap:** explicitly handled in Task 1 tests (non-Oldale states use explicit `map_num`; `reach_oldale` test asserts both the non-fire on route_101 and the fire on Oldale).
