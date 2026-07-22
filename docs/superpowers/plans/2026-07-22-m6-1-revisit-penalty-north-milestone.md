# M6.1 — Revisit Penalty + North Littleroot Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop tile-farming (revisited tiles now cost -0.01) and lure the agent to Littleroot's northern exit with a +10 milestone at y <= 1, so it finally tastes the Route 101 reward.

**Architecture:** Two isolated changes to the existing reward layer: `ExplorationTracker` gains a revisit penalty constant, and `starter_milestones()` gains one row between `back_outside` and `reach_route_101`. Chain total 155 → 165.

**Tech Stack:** Python 3.12, pytest (FakeEmulator, no ROM needed), ruff.

**Context (empirical, measured on the 3M checkpoint):** the agent hovers at
y=2-4 in Littleroot (map `(0, 9)`) and reached y=0 exactly once without
crossing north. `PlayerState` already exposes `x`, `y`, `clock_set`.

---

### Task 1: Revisit penalty in ExplorationTracker

**Files:**
- Modify: `env/rewards.py`
- Test: `tests/test_rewards.py`
- Modify: `tests/test_env.py` (one test's expectation changes)
- Modify: `docs/architecture/modules.md` (env/rewards.py row)

- [ ] **Step 1: Write the failing tests**

In `tests/test_rewards.py`, change the import line to:

```python
from env.rewards import REVISIT_PENALTY, REWARD_PER_LEVEL, ExplorationTracker, LevelRewardTracker
```

Update `test_new_tile_rewards_once` and add one test:

```python
def test_new_tile_rewards_once():
    tracker = ExplorationTracker()
    assert tracker.update(state(1, 1)) == 1.0
    assert tracker.update(state(1, 1)) == REVISIT_PENALTY
    assert tracker.update(state(2, 1)) == 1.0


def test_revisit_penalty_is_small_and_negative():
    assert -0.1 < REVISIT_PENALTY < 0.0
```

In `tests/test_env.py`, replace `test_staying_put_gives_zero_reward_after_first_visit` with:

```python
def test_staying_put_pays_revisit_penalty_after_first_visit():
    env = make_env()
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    env.step(noop)
    _, reward, _, _, _ = env.step(noop)
    assert reward == REVISIT_PENALTY
```

and add to that file's imports:

```python
from env.rewards import REVISIT_PENALTY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rewards.py tests/test_env.py -q`
Expected: FAIL — `ImportError: cannot import name 'REVISIT_PENALTY'`

- [ ] **Step 3: Implement the penalty**

In `env/rewards.py`:
- Module docstring becomes: `"""Reward shaping: +1 for never-visited tiles, small penalty for revisits."""`
- Add constant above the class:

```python
# Small enough that milestones dominate; large enough to make loitering lose.
REVISIT_PENALTY = -0.01
```

- Class docstring becomes: `"""+1.0 the first time each (map_group, map_num, x, y) tile is seen; REVISIT_PENALTY after."""`
- In `update()`, replace `return 0.0` for the visited branch:

```python
        if tile in self._visited:
            return REVISIT_PENALTY
```

(The `state is None` branch keeps returning `0.0`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rewards.py tests/test_env.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Update modules.md and lint**

In `docs/architecture/modules.md`, update the `env/rewards.py` row: mention
`REVISIT_PENALTY = -0.01` for revisited tiles.
Run: `.venv/bin/ruff check .` — expected: clean.

- [ ] **Step 6: Commit**

```bash
git add env/rewards.py tests/test_rewards.py tests/test_env.py docs/architecture/modules.md
git commit -m "feat: penalize revisited tiles (-0.01) to break tile-farming loops"
```

---

### Task 2: north_littleroot milestone (+10, chain = 165)

**Files:**
- Modify: `env/milestones.py`
- Test: `tests/test_milestones.py`
- Modify: `tests/test_env.py` (extend the intro-chain test)
- Modify: `docs/architecture/modules.md` (env/milestones.py row)

- [ ] **Step 1: Write the failing tests**

In `tests/test_milestones.py`:

1. Change the `make_state` default `y=0` to `y=5` (y=0 would now fire
   `north_littleroot` once clock_set is True — keep the neutral default
   truly neutral, mirroring the Oldale neutral-map decision).
2. Add these tests:

```python
def test_north_littleroot_fires_at_north_edge_with_clock_set():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(map_group=0, map_num=9, clock_set=True, y=1))
    assert "north_littleroot" in tracker.fired
    # exit_truck (5) + back_outside (10) + north_littleroot (10) also match this state
    assert reward == 25.0


def test_north_littleroot_needs_clock_set():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=9, clock_set=False, y=0))
    assert "north_littleroot" not in tracker.fired


def test_north_littleroot_not_fired_south_of_threshold():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=9, clock_set=True, y=2))
    assert "north_littleroot" not in tracker.fired


def test_north_littleroot_needs_littleroot_map():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=10, clock_set=True, y=0))
    assert "north_littleroot" not in tracker.fired
```

3. Update the full-chain test: rename `test_full_chain_sums_to_155` to
   `test_full_chain_sums_to_165`, insert a north step after the
   back-outside step:

```python
    # push to the northern exit of Littleroot
    total += tracker.update(make_state(map_group=0, map_num=9, clock_set=True, y=1))[0]
```

   and change the final asserts to `total == 165.0` and 7 fired milestones.

In `tests/test_env.py`, extend `test_intro_chain_pays_each_milestone_once`
after the back_outside block:

```python
    emu.y = 1  # walk up to the northern exit
    _, reward, _, _, info = env.step(0)
    assert reward >= 10.0
    assert "north_littleroot" in info["milestones"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_milestones.py tests/test_env.py -q`
Expected: FAIL — `north_littleroot` never fired / sum is 155.

- [ ] **Step 3: Implement the milestone**

In `env/milestones.py`:
- Add below the map constants:

```python
# Littleroot's exit to Route 101 is at the top edge; the 3M-step agent hovered
# at y=2-4 and touched y=0 once — this milestone pays for committing north.
NORTH_LITTLEROOT_MAX_Y = 1
```

- Docstring of `starter_milestones()` gains: chain total 165.
- Insert between `back_outside` and `reach_route_101`:

```python
        Milestone(
            "north_littleroot",
            lambda s: s.clock_set
            and (s.map_group, s.map_num) == LITTLEROOT
            and s.y <= NORTH_LITTLEROOT_MAX_Y,
            10.0,
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_milestones.py tests/test_env.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Update modules.md and lint**

In `docs/architecture/modules.md`, update the `env/milestones.py` row:
7-milestone chain, 165 pts, `NORTH_LITTLEROOT_MAX_Y`.
Run: `.venv/bin/ruff check .` — expected: clean.

- [ ] **Step 6: Commit**

```bash
git add env/milestones.py tests/test_milestones.py tests/test_env.py docs/architecture/modules.md
git commit -m "feat: add north_littleroot milestone (+10) to lure the agent toward Route 101"
```

---

### Task 3: Full validation with ROM

- [ ] Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest -q`
Expected: all tests pass (52+), ROM-gated tests included.
- [ ] Run: `.venv/bin/ruff check .` — expected: clean.
