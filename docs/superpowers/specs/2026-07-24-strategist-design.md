# Strategist (3rd brain) — v1 design

**Date:** 2026-07-24
**Status:** Approved (brainstorming), pending implementation plan
**Milestone:** Strategist v1 — the "continue vs grind vs heal" decision

## Goal

Build the first version of the Strategist: an RL agent that decides, at the
meta-game level, whether to advance to the next important battle, grind wild
battles to level up, or heal — trained in a fast abstract simulator calibrated
on the real game.

## Context

The project runs three decoupled brains:

- **Explorer** (existing, PPO CnnPolicy on pixels) — moves through the world.
- **Fighter** (existing, PPO MlpPolicy on battle RAM) — wins a single battle.
  Validated: 10/10 on captured wild battles, ~6 turns average.
- **Strategist** (this spec, new) — sits above the other two and decides *what
  goal to pursue*: advance the story, grind, or heal.

Today "the story" is a fixed linear chain of milestone rewards in
`env/milestones.py` that pulls the Explorer forward. There is no notion of the
agent *choosing* a goal. The Strategist introduces that choice.

## Scope (v1)

One decision: **continue vs grind vs heal**, as a true RL agent (PPO,
MlpPolicy), trained in an abstract fast simulator whose battle outcomes are
calibrated on the real Fighter's measured performance.

The Strategist learns the reflex: *level up and heal just enough to win the next
important battle, without wasting time.*

## Non-goals (deferred, "on enrichira")

- Destination / route selection (which place to go). v1 challenges are an
  abstract rising-difficulty curriculum, not the real map.
- Capture, items, party reordering, PC deposits, move-learning choices. (Capture
  is impossible in-game until the rival is beaten anyway.)
- Party **size** in the observation (party is 1 Pokémon until Poké Balls exist).
- Real-emulator training (sim-to-real deployment is a later milestone).

## Architecture: the abstract StrategistEnv (MDP)

The Strategist never sees pixels or the map. Its world is the *progression
state*. Two layers, kept separate so the "physics" can be recalibrated without
touching the MDP:

- `env/strategist_model.py` — the calibration "physics": win-probability model,
  grind yield, heal effect, and all seed constants. This is the one place that
  later gets recalibrated against measured Fighter/game data.
- `env/strategist_env.py` — the Gymnasium env wrapping that model into an MDP.

### Progression state (internal)

- `team_level: float` — average party level. Starts at 5.0.
- `team_hp: float` — team HP fraction in [0, 1]. Starts at 1.0.
- `challenge_idx: int` — index into the challenge curriculum. Starts at 0.
- `steps: int` — Strategist decisions taken this episode (time budget).

### Challenge curriculum (v1)

A short list of rising-difficulty important battles (target levels):

```python
CHALLENGE_LEVELS = (5, 8, 12, 16, 20)  # 5 challenges
```

Synthetic on purpose — the goal is to teach the *decision*, not the real map.

### Observation — 5 floats, clamped to [0, 1]

```
obs[0] = clip(team_level / 100, 0, 1)
obs[1] = team_hp                                    # already 0..1
obs[2] = clip(challenge_level / 100, 0, 1)          # next challenge difficulty
obs[3] = clip((team_level - challenge_level) / 40 + 0.5, 0, 1)   # centered gap
obs[4] = challenge_idx / len(CHALLENGE_LEVELS)      # progress
```

`observation_space = Box(0, 1, shape=(5,), float32)`.

### Actions — Discrete(3)

- `0 = ADVANCE` — attempt the current important battle.
- `1 = GRIND` — one wild-battle session: gain level, lose HP, cost time.
- `2 = HEAL` — restore HP to full, cost time (Pokémon Center detour).

### Dynamics (seed constants live in `strategist_model.py`)

```
GRIND_LEVEL_GAIN = 1.0     # levels per grind session
GRIND_HP_COST    = 0.30    # HP fraction lost per grind session
ADVANCE_HP_COST  = 0.30    # HP fraction lost on a won battle
# Win-probability logistic: p = sigmoid(A*dlevel + B*(hp - 1) + C)
WIN_A = 0.70               # sensitivity to level gap
WIN_B = 1.50               # sensitivity to missing HP
WIN_C = 1.00               # bias: equal level + full HP -> p ~= 0.73
```

- **GRIND**: `team_level += 1.0`; `team_hp = max(0, team_hp - 0.30)`.
- **HEAL**: `team_hp = 1.0`.
- **ADVANCE**: `dlevel = team_level - challenge_level`;
  `p_win = sigmoid(0.70*dlevel + 1.50*(team_hp - 1) + 1.00)`; draw a win with
  probability `p_win` using the env's seeded RNG.
  - **Win**: advance `challenge_idx += 1`, `team_hp -= 0.30`. If it was the last
    challenge, the episode terminates with success.
  - **Loss**: episode terminates (failure).

Sanity of the seed shape: equal level + full HP → p≈0.73; under-level by 4 at
full HP → p≈0.14 (grind first); equal level at 40% HP → p≈0.52 (heal helps).

### Reward

```
WIN_REWARD  = +20.0    # per important battle won
LOSS_REWARD = -20.0    # on a lost important battle (episode ends)
STEP_GRIND  =  -1.0    # time cost of a grind session
STEP_HEAL   =  -2.0    # time cost of a heal detour (bigger: it's a round trip)
```

Clearing all 5 challenges yields +100 minus accumulated time costs. The negative
per-step costs push the agent to grind/heal the minimum needed.

### Episode

- Start: `team_level=5`, `team_hp=1.0`, `challenge_idx=0`, `steps=0`.
- `terminated = True` on the last-challenge win (success) or any loss (failure).
- `truncated = True` if `steps >= MAX_STEPS` (budget, seed `MAX_STEPS = 50`) —
  guards against an infinite grind/heal loop.

## Components / files

- `env/strategist_model.py` — seed constants + pure functions `win_prob`,
  `grind`, `heal` (no Gym dependency; unit-testable in isolation).
- `env/strategist_env.py` — `StrategistEnv(gym.Env)` implementing the MDP above.
- `agent/train_strategist.py` — PPO MlpPolicy training + baseline eval, with
  `--timesteps`, `--n-steps`. Checkpoints under `checkpoints/strategist/`.
- `tests/test_strategist_model.py` — win-prob shape, grind/heal effects.
- `tests/test_strategist_env.py` — Gym API compliance (`check_env`), obs
  bounds/shape, reward signs, terminal conditions, one full scripted episode.

## Calibration

v1 ships the seed constants above so the sim is buildable immediately. Later,
each constant is measured against the real game (grind yield from actual XP,
`win_prob` from real Fighter win rates at various level gaps) and updated in the
one file `strategist_model.py`. No MDP change needed.

## Training + eval

- PPO, MlpPolicy, single env (the sim is CPU-cheap; no emulator).
- **Baselines to beat** (deterministic policies evaluated in the same sim):
  - *always ADVANCE* — loses early important battles (under-leveled).
  - *always GRIND* — never advances or wastes the whole time budget.
- **Success criterion**: the trained Strategist clears all 5 challenges with
  higher mean episode reward than both baselines, and demonstrably HEALs before
  low-HP advances (observable in an eval trace).

## Testing strategy

Pure-Python, no ROM, no emulator — the whole subsystem is deterministic given a
seed. `strategist_model.py` functions are tested directly; `StrategistEnv`
passes `gymnasium.utils.env_checker.check_env` and a scripted end-to-end episode.

## Future (out of scope, noted)

- Sim-to-real: deploy this same policy in the real game, where ADVANCE dispatches
  the Fighter and GRIND dispatches Explorer+Fighter on wild battles.
- Add HEAL-cost as money/time, party size, item use, destination selection, and
  eventually swap the abstract challenge curriculum for the real story graph.
