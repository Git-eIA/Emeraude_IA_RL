# Strategist v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Strategist — a PPO RL agent that decides, at the meta-game level, whether to advance to the next important battle, grind wild battles, or heal — trained in a fast abstract simulator calibrated on the real Fighter.

**Architecture:** Two decoupled layers. `env/strategist_model.py` holds the calibration "physics" (seed constants + pure functions `win_prob`, `grind`, `heal`) with no Gym dependency. `env/strategist_env.py` wraps that model into a Gymnasium MDP (`StrategistEnv`, Discrete(3) actions, 5-float observation). `agent/train_strategist.py` runs PPO MlpPolicy training plus deterministic-baseline evaluation. Everything is pure Python — no ROM, no emulator — so the whole subsystem is deterministic given a seed.

**Tech Stack:** Python ≥3.12, Gymnasium, Stable-Baselines3 (PPO, MlpPolicy), NumPy, pytest, ruff.

---

## Reference: spec

Full design at `docs/superpowers/specs/2026-07-24-strategist-design.md`. This plan implements it exactly.

## File Structure

| File | Responsibility |
|------|----------------|
| `env/strategist_model.py` (create) | Seed constants + pure functions `win_prob`, `grind`, `heal`. No Gym. The one file recalibrated later. |
| `env/strategist_env.py` (create) | `StrategistEnv(gym.Env)`: spaces, `reset`, `step`, observation encoding, reward, terminal/truncation. Owns MDP config (`CHALLENGE_LEVELS`, `MAX_STEPS`, reward constants). |
| `agent/train_strategist.py` (create) | PPO MlpPolicy training + `eval_policy` harness + two deterministic baselines. |
| `tests/test_strategist_model.py` (create) | Win-prob shape, grind/heal effects. |
| `tests/test_strategist_env.py` (create) | Gym API compliance (`check_env`), obs bounds/shape, reward signs, terminal conditions, one full scripted episode. |
| `tests/test_strategist_train.py` (create) | `eval_policy` harness correctness against a deterministic baseline. |

**Test command (no ROM needed):** `.venv/bin/pytest tests/test_strategist_model.py tests/test_strategist_env.py tests/test_strategist_train.py -q`

**Lint after each task:** `.venv/bin/ruff check env/strategist_model.py env/strategist_env.py agent/train_strategist.py`

---

### Task 1: Win-probability model

**Files:**
- Create: `env/strategist_model.py`
- Test: `tests/test_strategist_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_strategist_model.py`:

```python
"""Calibration physics: win-prob shape, grind/heal effects (no Gym)."""
from __future__ import annotations

from env.strategist_model import win_prob


def test_equal_level_full_hp_is_favorable() -> None:
    # sigmoid(WIN_C) = sigmoid(1.0) ~= 0.73
    p = win_prob(team_level=10.0, challenge_level=10.0, team_hp=1.0)
    assert abs(p - 0.731) < 0.01


def test_under_level_by_four_is_unlikely() -> None:
    # z = 0.70*(-4) + 0 + 1.0 = -1.8 -> sigmoid ~= 0.14
    p = win_prob(team_level=6.0, challenge_level=10.0, team_hp=1.0)
    assert abs(p - 0.142) < 0.01


def test_equal_level_low_hp_is_a_coin_flip() -> None:
    # z = 0 + 1.5*(0.4 - 1.0) + 1.0 = 0.1 -> sigmoid ~= 0.52
    p = win_prob(team_level=10.0, challenge_level=10.0, team_hp=0.4)
    assert abs(p - 0.525) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategist_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.strategist_model'`

- [ ] **Step 3: Write minimal implementation**

Create `env/strategist_model.py`:

```python
"""Calibration physics for the Strategist's abstract simulator.

Pure functions and seed constants — no Gym dependency, unit-testable in
isolation. This is the ONE place recalibrated later against measured
Fighter/game data; the MDP in strategist_env.py never changes.
"""
from __future__ import annotations

import math

# Grind: one wild-battle session.
GRIND_LEVEL_GAIN = 1.0   # levels gained per grind session
GRIND_HP_COST = 0.30     # HP fraction lost per grind session

# Advance: HP cost of a won important battle.
ADVANCE_HP_COST = 0.30

# Win-probability logistic: p = sigmoid(A*dlevel + B*(hp - 1) + C).
WIN_A = 0.70   # sensitivity to level gap
WIN_B = 1.50   # sensitivity to missing HP
WIN_C = 1.00   # bias: equal level + full HP -> p ~= 0.73


def win_prob(team_level: float, challenge_level: float, team_hp: float) -> float:
    """Probability the Fighter wins the important battle, in [0, 1]."""
    dlevel = team_level - challenge_level
    z = WIN_A * dlevel + WIN_B * (team_hp - 1.0) + WIN_C
    # math.exp overflows to inf for very negative z, giving p -> 0 cleanly;
    # for very positive z, exp(-z) underflows to 0, giving p -> 1.
    return 1.0 / (1.0 + math.exp(-z))
```

Note: `math.exp(-z)` for large positive `z` underflows to `0.0` (p→1). For large negative `z`, `math.exp(-z)` may raise `OverflowError` at |z| ≳ 710. Guard it:

```python
def win_prob(team_level: float, challenge_level: float, team_hp: float) -> float:
    """Probability the Fighter wins the important battle, in [0, 1]."""
    dlevel = team_level - challenge_level
    z = WIN_A * dlevel + WIN_B * (team_hp - 1.0) + WIN_C
    if z < -700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategist_model.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add env/strategist_model.py tests/test_strategist_model.py
git commit -m "feat(strategist): win-probability logistic model"
```

---

### Task 2: Grind and heal physics

**Files:**
- Modify: `env/strategist_model.py`
- Test: `tests/test_strategist_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_strategist_model.py`:

```python
from env.strategist_model import grind, heal


def test_grind_gains_a_level_and_loses_hp() -> None:
    level, hp = grind(team_level=5.0, team_hp=1.0)
    assert level == 6.0
    assert abs(hp - 0.70) < 1e-9


def test_grind_clamps_hp_at_zero() -> None:
    _, hp = grind(team_level=5.0, team_hp=0.20)
    assert hp == 0.0


def test_heal_restores_full_hp() -> None:
    assert heal() == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategist_model.py -q`
Expected: FAIL with `ImportError: cannot import name 'grind'`

- [ ] **Step 3: Write minimal implementation**

Append to `env/strategist_model.py`:

```python
def grind(team_level: float, team_hp: float) -> tuple[float, float]:
    """One grind session: gain a level, lose HP (clamped at 0)."""
    return team_level + GRIND_LEVEL_GAIN, max(0.0, team_hp - GRIND_HP_COST)


def heal() -> float:
    """Heal restores team HP to full."""
    return 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategist_model.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add env/strategist_model.py tests/test_strategist_model.py
git commit -m "feat(strategist): grind and heal physics"
```

---

### Task 3: StrategistEnv scaffold — spaces, reset, observation

**Files:**
- Create: `env/strategist_env.py`
- Test: `tests/test_strategist_env.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_strategist_env.py`:

```python
"""StrategistEnv: Gym API compliance + MDP behavior (no ROM, no emulator)."""
from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.strategist_env import CHALLENGE_LEVELS, StrategistEnv


def test_gym_api_compliance() -> None:
    check_env(StrategistEnv(), skip_render_check=True)


def test_observation_shape_and_bounds() -> None:
    env = StrategistEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    assert info == {}


def test_reset_starts_at_first_challenge_full_hp() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    assert env.team_level == 5.0
    assert env.team_hp == 1.0
    assert env.challenge_idx == 0
    assert env.steps == 0


def test_observation_encodes_progression() -> None:
    env = StrategistEnv()
    obs, _ = env.reset(seed=0)
    # team_level 5/100, hp 1.0, challenge 5/100, gap centered at 0.5, progress 0.
    assert abs(obs[0] - 0.05) < 1e-6
    assert obs[1] == 1.0
    assert abs(obs[2] - 0.05) < 1e-6
    assert abs(obs[3] - 0.5) < 1e-6
    assert obs[4] == 0.0
    assert len(CHALLENGE_LEVELS) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategist_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.strategist_env'`

- [ ] **Step 3: Write minimal implementation**

Create `env/strategist_env.py`:

```python
"""StrategistEnv: abstract MDP for the continue/grind/heal decision.

No pixels, no map, no emulator — the Strategist's world is the progression
state (team level, team HP, which important battle is next). Battle outcomes
resolve through the calibrated win-probability model in strategist_model.py.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.strategist_model import ADVANCE_HP_COST, grind, heal, win_prob

# Rising-difficulty curriculum of important battles (target levels). Synthetic
# on purpose — v1 teaches the decision, not the real map.
CHALLENGE_LEVELS = (5, 8, 12, 16, 20)

MAX_STEPS = 50   # decision budget per episode (guards infinite grind/heal loops)

# Actions.
ADVANCE = 0   # attempt the current important battle
GRIND = 1     # one wild-battle session: gain level, lose HP, cost time
HEAL = 2      # restore HP to full, cost time (Pokemon Center detour)

# Reward.
WIN_REWARD = 20.0    # per important battle won
LOSS_REWARD = -20.0  # on a lost important battle (episode ends)
STEP_GRIND = -1.0    # time cost of a grind session
STEP_HEAL = -2.0     # time cost of a heal detour (round trip, so bigger)

START_LEVEL = 5.0
START_HP = 1.0


class StrategistEnv(gym.Env):
    """Meta-game MDP: choose advance / grind / heal to clear the curriculum."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.team_level = START_LEVEL
        self.team_hp = START_HP
        self.challenge_idx = 0
        self.steps = 0

    def _challenge_level(self) -> int:
        # Clamp: after the last win challenge_idx == len, but _obs still runs.
        idx = min(self.challenge_idx, len(CHALLENGE_LEVELS) - 1)
        return CHALLENGE_LEVELS[idx]

    def _obs(self) -> np.ndarray:
        cl = self._challenge_level()
        gap = float(np.clip((self.team_level - cl) / 40.0 + 0.5, 0.0, 1.0))
        return np.array(
            [
                np.clip(self.team_level / 100.0, 0.0, 1.0),
                np.clip(self.team_hp, 0.0, 1.0),
                np.clip(cl / 100.0, 0.0, 1.0),
                gap,
                self.challenge_idx / len(CHALLENGE_LEVELS),
            ],
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.team_level = START_LEVEL
        self.team_hp = START_HP
        self.challenge_idx = 0
        self.steps = 0
        return self._obs(), {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategist_env.py -q`
Expected: FAIL — `check_env` requires a working `step`. Only the three non-`check_env` tests pass. This is expected; `test_gym_api_compliance` fails with `NotImplementedError` (no `step`).

Run only the passing subset to confirm the scaffold is right:
Run: `.venv/bin/pytest tests/test_strategist_env.py -q -k "not gym_api"`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add env/strategist_env.py tests/test_strategist_env.py
git commit -m "feat(strategist): StrategistEnv scaffold — spaces, reset, observation"
```

---

### Task 4: StrategistEnv step — dynamics, reward, terminal

**Files:**
- Modify: `env/strategist_env.py`
- Test: `tests/test_strategist_env.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_strategist_env.py`:

```python
from env.strategist_env import ADVANCE, GRIND, HEAL, MAX_STEPS


def test_grind_levels_up_costs_hp_and_time() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    _, reward, term, trunc, _ = env.step(GRIND)
    assert env.team_level == 6.0
    assert abs(env.team_hp - 0.70) < 1e-9
    assert reward == -1.0
    assert term is False and trunc is False


def test_heal_restores_hp_and_costs_time() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    env.team_hp = 0.25
    _, reward, term, trunc, _ = env.step(HEAL)
    assert env.team_hp == 1.0
    assert reward == -2.0
    assert term is False and trunc is False


def test_advance_win_pays_bonus_costs_hp_and_advances() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    env.team_level = 200.0   # overwhelming -> win_prob == 1.0, deterministic win
    hp_before = env.team_hp
    _, reward, term, trunc, _ = env.step(ADVANCE)
    assert reward == 20.0
    assert env.challenge_idx == 1
    assert abs(env.team_hp - (hp_before - 0.30)) < 1e-9
    assert term is False and trunc is False


def test_advance_loss_ends_the_episode() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    env.team_level = -200.0  # hopeless -> win_prob == 0.0, deterministic loss
    _, reward, term, trunc, _ = env.step(ADVANCE)
    assert reward == -20.0
    assert term is True


def test_clearing_all_five_challenges_succeeds() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    total = 0.0
    terminated = False
    for _ in range(5):
        env.team_level = 200.0   # force a win each advance
        _, reward, terminated, _, _ = env.step(ADVANCE)
        total += reward
        if terminated:
            break
    assert terminated is True
    assert env.challenge_idx == 5
    assert total == 100.0   # 5 wins * WIN_REWARD


def test_truncates_at_step_budget() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(HEAL)  # never ends, only costs time
        steps += 1
    assert truncated is True
    assert terminated is False
    assert steps == MAX_STEPS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategist_env.py -q`
Expected: FAIL — `NotImplementedError` from the missing `step` (and `check_env` still fails).

- [ ] **Step 3: Write minimal implementation**

Append `step` to `StrategistEnv` in `env/strategist_env.py`:

```python
    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.steps += 1
        terminated = False

        if action == GRIND:
            self.team_level, self.team_hp = grind(self.team_level, self.team_hp)
            reward = STEP_GRIND
        elif action == HEAL:
            self.team_hp = heal()
            reward = STEP_HEAL
        else:  # ADVANCE
            p = win_prob(self.team_level, self._challenge_level(), self.team_hp)
            if self.np_random.random() < p:
                reward = WIN_REWARD
                self.team_hp = max(0.0, self.team_hp - ADVANCE_HP_COST)
                self.challenge_idx += 1
                if self.challenge_idx >= len(CHALLENGE_LEVELS):
                    terminated = True   # cleared the whole curriculum
            else:
                reward = LOSS_REWARD
                terminated = True       # lost an important battle

        truncated = self.steps >= MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategist_env.py -q`
Expected: PASS (10 passed — `check_env` now succeeds too)

Then lint:
Run: `.venv/bin/ruff check env/strategist_model.py env/strategist_env.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add env/strategist_env.py tests/test_strategist_env.py
git commit -m "feat(strategist): step dynamics, reward, and terminal conditions"
```

---

### Task 5: Trainer + baseline evaluation

**Files:**
- Create: `agent/train_strategist.py`
- Test: `tests/test_strategist_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_strategist_train.py`:

```python
"""eval_policy harness: deterministic baselines behave as expected."""
from __future__ import annotations

from agent.train_strategist import always_grind, eval_policy


def test_always_grind_never_clears_and_wastes_the_budget() -> None:
    # Grinding only never advances -> 0 challenges cleared, truncates at the
    # 50-step budget paying STEP_GRIND (-1) each step -> mean reward -50.
    mean_reward, clear_rate = eval_policy(always_grind, episodes=20, seed=0)
    assert clear_rate == 0.0
    assert abs(mean_reward - (-50.0)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_strategist_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.train_strategist'`

- [ ] **Step 3: Write minimal implementation**

Create `agent/train_strategist.py`:

```python
"""Train the Strategist: PPO MlpPolicy on the abstract StrategistEnv.

The sim is CPU-cheap (no emulator), so a single env is fine. We also evaluate
two deterministic baselines in the same sim — always-ADVANCE (loses early,
under-leveled) and always-GRIND (wastes the whole budget) — so we can confirm
the learned policy beats both.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.strategist_env import ADVANCE, CHALLENGE_LEVELS, GRIND, StrategistEnv

Policy = Callable[[np.ndarray], int]


def always_advance(_obs: np.ndarray) -> int:
    return ADVANCE


def always_grind(_obs: np.ndarray) -> int:
    return GRIND


def eval_policy(policy: Policy, episodes: int = 100, seed: int = 0) -> tuple[float, float]:
    """Run policy over episodes; return (mean episode reward, clear rate)."""
    env = StrategistEnv()
    total = 0.0
    cleared = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            ep_reward += reward
            done = terminated or truncated
        total += ep_reward
        if env.challenge_idx >= len(CHALLENGE_LEVELS):
            cleared += 1
    return total / episodes, cleared / episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--n-steps", type=int, default=2048)  # steps per PPO rollout
    args = ap.parse_args()

    # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean
    # and rollout/ep_len_mean (needed to see if the Strategist is learning).
    env = DummyVecEnv([lambda: Monitor(StrategistEnv())])
    ckpt = CheckpointCallback(
        save_freq=50_000, save_path="checkpoints/strategist", name_prefix="ppo_strategist"
    )
    model = PPO("MlpPolicy", env, n_steps=args.n_steps, verbose=1, device="cpu")

    adv_reward, adv_clear = eval_policy(always_advance)
    grind_reward, grind_clear = eval_policy(always_grind)
    print(f"baseline always-ADVANCE: mean_reward={adv_reward:.1f} clear_rate={adv_clear:.2f}")
    print(f"baseline always-GRIND:   mean_reward={grind_reward:.1f} clear_rate={grind_clear:.2f}")

    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    Path("checkpoints/strategist").mkdir(parents=True, exist_ok=True)
    model.save("checkpoints/strategist/ppo_strategist_final")

    def trained(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    tr_reward, tr_clear = eval_policy(trained)
    print(f"trained Strategist:      mean_reward={tr_reward:.1f} clear_rate={tr_clear:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_strategist_train.py -q`
Expected: PASS (1 passed)

Then lint:
Run: `.venv/bin/ruff check agent/train_strategist.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add agent/train_strategist.py tests/test_strategist_train.py
git commit -m "feat(strategist): PPO trainer + deterministic baseline eval"
```

---

### Task 6: Full-suite check + train-and-validate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest -q`
Expected: all previously-passing tests still pass, plus the new Strategist tests (model 6 + env 10 + train 1 = 17 new).

- [ ] **Step 2: Train the Strategist and read the baseline comparison**

Run: `.venv/bin/python agent/train_strategist.py --timesteps 200000 --n-steps 2048`
Expected: the two baseline lines print first (always-GRIND `mean_reward=-50.0 clear_rate=0.00`; always-ADVANCE clears < 1.0 with a lower mean reward than the trained run), PPO logs show `rollout/ep_rew_mean` climbing, and the final `trained Strategist:` line reports a higher `mean_reward` than both baselines with `clear_rate` near 1.0. Checkpoint saved to `checkpoints/strategist/ppo_strategist_final.zip`.

- [ ] **Step 3: Success criterion check**

Confirm from the printed output that the trained Strategist's `mean_reward` beats BOTH baselines and `clear_rate` is high (clears all 5 challenges). If it does not, the seed constants or timesteps need tuning — do NOT change the MDP; adjust `--timesteps` first, then, only if needed, the seed constants in `env/strategist_model.py`.

- [ ] **Step 4: Commit any checkpoint metadata (if applicable)**

`checkpoints/` is typically gitignored. Do not force-add the checkpoint. If a training log or notes file was produced and is worth keeping, commit only that. Otherwise no commit is needed for this verification task.

---

## Self-Review

**1. Spec coverage:**
- Goal / v1 scope (continue vs grind vs heal, PPO MlpPolicy, abstract sim) → Tasks 3–5. ✓
- `env/strategist_model.py` (win_prob, grind, heal, seed constants) → Tasks 1–2. ✓
- `env/strategist_env.py` (progression state, observation 5 floats, Discrete(3), dynamics, reward, episode terminal/truncation) → Tasks 3–4. ✓
- Challenge curriculum `(5, 8, 12, 16, 20)` → Task 3 (`CHALLENGE_LEVELS`). ✓
- Win-prob logistic with seed constants + sanity values (0.73 / 0.14 / 0.52) → Task 1 tests. ✓
- Reward table (+20 / −20 / −1 / −2) → Task 4. ✓
- `MAX_STEPS = 50` truncation → Task 4 (`test_truncates_at_step_budget`). ✓
- `agent/train_strategist.py` PPO + baselines (always-ADVANCE, always-GRIND) + checkpoints → Task 5. ✓
- Tests: model shape, env `check_env`, obs bounds/shape, reward signs, terminal conditions, scripted episode → Tasks 1–4. ✓
- Success criterion (beat both baselines, clear all 5) → Task 6. ✓
No gaps.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows complete code. ✓

**3. Type consistency:** `win_prob(team_level, challenge_level, team_hp)`, `grind(team_level, team_hp) -> tuple[float, float]`, `heal() -> float` — used identically in Task 4's `step`. Constants (`CHALLENGE_LEVELS`, `MAX_STEPS`, `ADVANCE`/`GRIND`/`HEAL`, reward names) referenced in Tasks 4–5 match their Task 3 definitions. `eval_policy` / `always_grind` signatures in Task 5 match the Task 5 test. ✓
