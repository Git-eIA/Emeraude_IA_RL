# Go-Explore Palier 0 (reset-state distribution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Explorer start training episodes from a random savestate drawn from `states/initial.state` + captured frontier states, instead of always the truck, to test whether reset-from-frontier cures the detachment regression.

**Architecture:** Mirror the proven multi-state pattern in `env/battle_env.py`: `PokemonEmeraldEnv` takes `initial_states: list[bytes]` and `reset()` picks one uniformly via `self.np_random`. `agent/train.py` loads the truck plus every `states/explorer/*.state`. A new one-shot tool `tools/capture_frontier.py` drives the trained checkpoint and dumps a savestate per frontier milestone. No reward changes.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3 (PPO), pytest. Emerald BPEF only. Unit tests use `tests.conftest.FakeEmulator` (no ROM); one ROM-gated smoke.

Reference spec: `docs/superpowers/specs/2026-07-31-go-explore-palier0-design.md`.

Run tests (venv + roms live in the MAIN repo, not this worktree):
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore \
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```

---

## File Structure

- Modify `env/pokemon_env.py` — constructor takes `initial_states: list[bytes]`; `reset()` draws one at random.
- Modify `tests/test_env.py` — add multi-state + empty-list tests; migrate existing call sites to `initial_states=[...]`.
- Modify `tests/test_train_smoke.py` — migrate its call site.
- Modify `tools/watch.py`, `tools/capture_open_map.py`, `tools/auto_capture_battles.py` — wrap their single state in a list.
- Modify `agent/train.py` — add `load_initial_states()` helper; feed the list to every env.
- Create `tests/test_train_states.py` — unit-test `load_initial_states()`.
- Create `tools/capture_frontier.py` — one-shot frontier-state capture (no unit test, matches `capture_open_map.py`).

Do NOT touch `env/world_reader.py`, `env/map_memory.py`, `env/orders.py`, or any reward file — those overlap with the in-flight `feat/p4-heal-mode` worktree and are out of scope here.

---

### Task 1: PokemonEmeraldEnv accepts a list of initial states

**Files:**
- Modify: `env/pokemon_env.py:40-63`
- Test: `tests/test_env.py`
- Migrate call sites: `tests/test_env.py`, `tests/test_train_smoke.py:13`, `tools/watch.py:37-39`, `tools/capture_open_map.py:38`, `tools/auto_capture_battles.py:69`

- [ ] **Step 1: Write the failing tests**

Add these two tests to `tests/test_env.py` (append at end of file):

```python
def test_reset_draws_one_of_the_initial_states():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_states=[b"a", b"b", b"c"], max_steps=50)
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        seen.add(emu.loaded_states[-1])
    # Over 20 seeds the uniform draw should hit more than one state.
    assert seen.issubset({b"a", b"b", b"c"})
    assert len(seen) > 1


def test_empty_initial_states_raises():
    import pytest

    with pytest.raises(ValueError):
        PokemonEmeraldEnv(FakeEmulator(), initial_states=[], max_steps=50)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_env.py::test_reset_draws_one_of_the_initial_states tests/test_env.py::test_empty_initial_states_raises -q
```
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'initial_states'`.

- [ ] **Step 3: Change the constructor and reset()**

In `env/pokemon_env.py`, change the constructor signature and body:

```python
    def __init__(
        self,
        emulator: Any,
        initial_states: list[bytes],
        max_steps: int = 2048,
    ) -> None:
        super().__init__()
        if not initial_states:
            raise ValueError("initial_states must be non-empty")
        self.emulator = emulator
        self._initial_states = initial_states
        self._max_steps = max_steps
```
(leave the rest of `__init__` — `self._reader = ...` onward — unchanged.)

And change the first two lines of `reset()`'s body (the `super().reset(...)` line stays):

```python
        super().reset(seed=seed)
        idx = int(self.np_random.integers(len(self._initial_states)))
        self.emulator.load_state(self._initial_states[idx])
```

Update the class docstring line 35 to:
```python
    """Pixels in, exploration reward out. Episodes start from a random savestate."""
```

- [ ] **Step 4: Migrate every existing call site to the list form**

In `tests/test_env.py`:
- Line 12: `return PokemonEmeraldEnv(FakeEmulator(), initial_states=[b"fake"], max_steps=max_steps)`
- Lines 61, 75, 86, 96: replace `initial_state=b"state"` with `initial_states=[b"state"]`.
- The assertion `assert env.emulator.loaded_states == [b"fake"]` (line 22) stays correct: a single-element list always loads `b"fake"`.

In `tests/test_train_smoke.py:13`:
```python
        [lambda: PokemonEmeraldEnv(FakeEmulator(), initial_states=[b"fake"], max_steps=64)]
```

In `tools/watch.py:37-39`:
```python
    env = PokemonEmeraldEnv(
        GbaEmulator(rom), [Path(args.state).read_bytes()], max_steps=args.max_steps
    )
```

In `tools/capture_open_map.py:38`:
```python
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
```

In `tools/auto_capture_battles.py:69`:
```python
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
```

- [ ] **Step 5: Run the full env test file to verify everything passes**

Run:
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_env.py -q
```
Expected: PASS (all prior tests + the 2 new ones).

- [ ] **Step 6: Commit**

```bash
git add env/pokemon_env.py tests/test_env.py tests/test_train_smoke.py tools/watch.py tools/capture_open_map.py tools/auto_capture_battles.py
git commit -m "feat: PokemonEmeraldEnv draws a random initial state per reset"
```

---

### Task 2: train.py loads truck + frontier states as a list

**Files:**
- Modify: `agent/train.py:20-56`
- Create/Test: `tests/test_train_states.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_states.py`:

```python
from __future__ import annotations

import pytest

from agent.train import load_initial_states


def test_load_returns_truck_plus_frontier(tmp_path):
    truck = tmp_path / "initial.state"
    truck.write_bytes(b"truck")
    frontier = tmp_path / "explorer"
    frontier.mkdir()
    (frontier / "reach_route_101.state").write_bytes(b"r101")
    (frontier / "meet_rival.state").write_bytes(b"rival")

    states = load_initial_states(truck, frontier)

    assert states[0] == b"truck"  # truck always first
    assert set(states) == {b"truck", b"r101", b"rival"}


def test_load_truck_only_when_no_frontier_dir(tmp_path):
    truck = tmp_path / "initial.state"
    truck.write_bytes(b"truck")
    states = load_initial_states(truck, tmp_path / "explorer")  # dir absent
    assert states == [b"truck"]


def test_load_raises_when_truck_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_initial_states(tmp_path / "initial.state", tmp_path / "explorer")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_train_states.py -q
```
Expected: FAIL with `ImportError: cannot import name 'load_initial_states'`.

- [ ] **Step 3: Add the helper and wire it into main()**

In `agent/train.py`, add the helper after the `STATE_PATH` constant (line 20) and add an `EXPLORER_DIR` constant:

```python
STATE_PATH = Path("states/initial.state")
EXPLORER_DIR = Path("states/explorer")


def load_initial_states(truck: Path, frontier_dir: Path) -> list[bytes]:
    """Truck state first, then every frontier savestate (Go-Explore Palier 0)."""
    if not truck.is_file():
        raise FileNotFoundError(truck)
    states = [truck.read_bytes()]
    if frontier_dir.is_dir():
        states += [p.read_bytes() for p in sorted(frontier_dir.glob("*.state"))]
    return states
```

Change `make_env` to take the list:

```python
def make_env(rom_path: str, initial_states: list[bytes], max_steps: int):
    def _init() -> Monitor:
        # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean.
        env = PokemonEmeraldEnv(GbaEmulator(rom_path), initial_states, max_steps=max_steps)
        return Monitor(env)

    return _init
```

In `main()`, replace the state-loading block (current lines 49-56) with:

```python
    if not STATE_PATH.is_file():
        log.error("Missing %s — create it with tools/play_interactive.py", STATE_PATH)
        return 1

    initial_states = load_initial_states(STATE_PATH, EXPLORER_DIR)
    log.info("Reset pool: %d state(s) (truck + %d frontier)", len(initial_states), len(initial_states) - 1)
    vec = SubprocVecEnv(
        [make_env(rom, initial_states, args.max_steps) for _ in range(args.envs)]
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_train_states.py -q
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/train.py tests/test_train_states.py
git commit -m "feat: train.py builds the reset pool from truck + states/explorer"
```

---

### Task 3: capture_frontier.py — dump a savestate per frontier milestone

**Files:**
- Create: `tools/capture_frontier.py`

No unit test: this is a one-shot ROM+checkpoint tool, consistent with `tools/capture_open_map.py` (also untested). It reuses `PokemonEmeraldEnv` (covered by Task 1) and reads milestones from the env's own `info` dict.

- [ ] **Step 1: Create the tool**

```python
"""Capture frontier savestates by letting the trained Explorer play.

Drives the Explorer PPO policy from states/initial.state and, the first time a
frontier milestone appears in the env info, writes states/explorer/<name>.state.
These become extra reset points for Go-Explore Palier 0 training. One-shot,
run locally where checkpoints/ppo_emerald_final.zip exists (outputs gitignored).

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_frontier.py \
      --episodes 5 --max-steps 4096
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv

# Milestones in the detachment zone worth restarting from. exit_truck/enter_house/
# clock_set are too early to matter; starter_obtained is terminal.
FRONTIER_MILESTONES = frozenset(
    {"enter_rival_house", "rival_upstairs", "meet_rival", "north_littleroot", "reach_route_101"}
)
OUT_DIR = Path("states/explorer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=args.max_steps)
    model = PPO.load(args.model, device="cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captured: set[str] = set()

    for episode in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(int(action))
            for name in info["milestones"]:
                if name in FRONTIER_MILESTONES and name not in captured:
                    out = OUT_DIR / f"{name}.state"
                    out.write_bytes(env.emulator.save_state())
                    captured.add(name)
                    print(f"captured {name} -> {out.resolve()}", flush=True)
            if terminated or truncated:
                break
        if captured >= FRONTIER_MILESTONES:
            break

    missing = sorted(FRONTIER_MILESTONES - captured)
    print(f"done: {len(captured)} captured, missing={missing}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax/import check (no ROM needed)**

Run:
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore /Users/_eloi/Projets/Emu/.venv/bin/python -c "import ast; ast.parse(open('tools/capture_frontier.py').read()); print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add tools/capture_frontier.py
git commit -m "feat: capture_frontier tool dumps a savestate per frontier milestone"
```

---

### Task 4: Full test suite green

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite with the ROM**

Run:
```
PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```
Expected: all prior tests still pass, plus the new `test_env.py` (2) and `test_train_states.py` (3). No regressions.

- [ ] **Step 2: Lint**

Run:
```
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/pokemon_env.py agent/train.py tools/capture_frontier.py tests/test_env.py tests/test_train_states.py
```
Expected: no errors.

---

## Manual runbook (after merge — not part of the coded tasks)

Run in the MAIN repo (`~/Projets/Emu`, where checkpoints/roms/states live):
1. `POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_frontier.py --episodes 5 --max-steps 4096` → populates `states/explorer/`.
2. `POKEMON_EMERALD_ROM=... .venv/bin/python agent/train.py --resume checkpoints/<good_ckpt>.zip --timesteps 1500000 --max-steps 4096` → trains with the multi-reset pool.
3. Jalon eval (10 stochastic episodes, `max_steps=4096`, dedup milestones by set) from `states/initial.state`.
4. Go/no-go: success = `reach_route_101` / `meet_rival` hold or improve vs baseline (9/10, 10/10 at ~9-10.5M, then 0/10) with no collapse. Ambiguous → Palier 1 runs the A/B.

---

## Self-Review

**Spec coverage:**
- Reset-state distribution (uniform over dir + truck) → Task 1 (env) + Task 2 (train pool). ✓
- Mirror `battle_env` pattern, non-empty guard → Task 1 Step 3. ✓
- `initial.state` always in pool, graceful when `states/explorer/` empty/absent → Task 2 helper + `test_load_truck_only_when_no_frontier_dir`. ✓
- `capture_frontier.py` modeled on `capture_open_map`, saves per frontier milestone → Task 3. ✓
- Rewards untouched; heal-overlapping files untouched → File Structure note; no reward/world_reader/map_memory/orders edits in any task. ✓
- Tests: unit multi-state + empty raise + ROM smoke → Task 1 new tests reuse FakeEmulator; existing ROM-gated env tests in the suite still run via Task 4. ✓
- Experiment protocol → Manual runbook. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code. ✓

**Type consistency:** `initial_states: list[bytes]` used identically in env ctor, `make_env`, and all call sites. `load_initial_states(truck: Path, frontier_dir: Path) -> list[bytes]` matches its test and its call in `main()`. `info["milestones"]` is the list produced by `PokemonEmeraldEnv._info` (milestone names like `reach_route_101`, `meet_rival`), matching `FRONTIER_MILESTONES`. ✓
