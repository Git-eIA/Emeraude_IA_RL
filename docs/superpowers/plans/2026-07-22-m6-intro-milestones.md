# M6 Intro Milestones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the milestone chain backwards to the truck start so the agent learns the scripted intro (exit truck, enter house, set clock, back outside) before Route 101 and the starter.

**Architecture:** `PlayerState` gains a `clock_set` boolean read from event flag 0x51; `starter_milestones()` grows from 2 to 6 rows; FakeEmulator and test defaults move to neutral maps so no milestone fires by accident.

**Tech Stack:** Python 3.12, pytest, Gymnasium, libmgba-py, stable-baselines3 (PPO).

**Spec:** `docs/superpowers/specs/2026-07-22-m6-intro-milestones-design.md`

**Key constants (verified):**
- `FLAG_SET_WALL_CLOCK = 0x51` (pret flags.h) → flags byte index 10, bit 1
- Littleroot = `(0, 9)`, houses 1F = `(1, 0)` and `(1, 2)`, Route 101 = `(0, 16)`, Oldale (neutral) = `(0, 10)`, InsideOfTruck = `(25, 40)` (empirical RAM read)
- New savestate `states/initial.state`: truck, pos (2,2), 0 party, 0 badges

---

### Task 1: `clock_set` in game state reader

**Files:**
- Modify: `env/game_state.py`
- Test: `tests/test_game_state.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_game_state.py`, extend `build_memory` with a `clock_set` keyword and add two tests. Replace the current `build_memory` signature and body with:

```python
def build_memory(
    *,
    x: int,
    y: int,
    map_group: int,
    map_num: int,
    badge_bits: int,
    party_count: int,
    clock_set: bool = False,
) -> dict[int, bytes]:
    sb1 = 0x02025A00  # arbitrary but valid EWRAM address for the fake
    save_block1 = bytearray(0x1400)  # must fit flags region up to 0x137E
    save_block1[0:2] = x.to_bytes(2, "little", signed=True)
    save_block1[2:4] = y.to_bytes(2, "little", signed=True)
    save_block1[4] = map_group
    save_block1[5] = map_num
    # Badge flags start at flag 0x867 -> byte 0x10C bit 7 of the flags array
    flags_value = badge_bits << 7
    save_block1[0x1270 + 0x10C : 0x1270 + 0x10E] = flags_value.to_bytes(2, "little")
    if clock_set:
        # FLAG_SET_WALL_CLOCK = 0x51 -> byte 10, bit 1 of the flags array
        save_block1[0x1270 + 10] |= 0b10
    return {
        SAVE_BLOCK1_PTR: sb1.to_bytes(4, "little"),
        sb1: bytes(save_block1),
        PARTY_COUNT_ADDR: bytes([party_count]),
    }
```

Add after `test_read_flag_invalid_pointer_is_false`:

```python
def test_clock_set_flag_read_into_state():
    memory = build_memory(
        x=0, y=0, map_group=1, map_num=1, badge_bits=0, party_count=0, clock_set=True
    )
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.clock_set is True


def test_clock_not_set_by_default():
    memory = build_memory(x=0, y=0, map_group=1, map_num=1, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.clock_set is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_game_state.py -q`
Expected: FAIL — `AttributeError` / `TypeError` on `clock_set` (field does not exist yet).

- [ ] **Step 3: Implement**

In `env/game_state.py`, after `_FIRST_BADGE_FLAG = 0x867` add:

```python
# FLAG_SET_WALL_CLOCK from pret/pokeemerald include/constants/flags.h.
# The intro is complete only once the bedroom wall clock has been set.
FLAG_SET_WALL_CLOCK = 0x51
```

In `PlayerState` add the field (with default so existing constructions stay valid):

```python
@dataclass(frozen=True)
class PlayerState:
    x: int
    y: int
    map_group: int
    map_num: int
    badges: int
    party_count: int
    clock_set: bool = False
```

In `player_state()`, add to the `PlayerState(...)` constructor call:

```python
            clock_set=self._flag(sb1, FLAG_SET_WALL_CLOCK),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_game_state.py -q`
Expected: all PASS (the pre-existing `test_reads_player_state` equality still holds because `clock_set` defaults to `False` on both sides).

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check env/game_state.py tests/test_game_state.py`

```bash
git add env/game_state.py tests/test_game_state.py
git commit -m "feat: read wall-clock intro flag into PlayerState"
```

---

### Task 2: intro milestone chain

**Files:**
- Modify: `env/milestones.py`
- Test: `tests/test_milestones.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_milestones.py`, replace `make_state` defaults with a **neutral map** (Oldale `(0, 10)`) — the old default `(0, 9)` is Littleroot and would now fire `exit_truck`:

```python
def make_state(**overrides) -> PlayerState:
    # Neutral defaults: Oldale (0, 10) fires no milestone.
    defaults = dict(x=0, y=0, map_group=0, map_num=10, badges=0, party_count=0, clock_set=False)
    return PlayerState(**{**defaults, **overrides})
```

Add these tests at the end of the file:

```python
def test_exit_truck_milestone():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=0, map_num=9))
    assert reward == 5.0
    assert terminated is False
    assert tracker.fired == frozenset({"exit_truck"})


def test_enter_house_fires_for_both_houses():
    for map_num in (0, 2):
        tracker = MilestoneTracker(starter_milestones())
        reward, _ = tracker.update(make_state(map_group=1, map_num=map_num))
        assert reward == 5.0
        assert tracker.fired == frozenset({"enter_house"})


def test_clock_set_milestone():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=1, map_num=0, clock_set=True))
    # enter_house (+5) and clock_set (+15) fire together in the house
    assert reward == 20.0
    assert terminated is False


def test_back_outside_requires_clock():
    tracker = MilestoneTracker(starter_milestones())
    # Outside without clock: only exit_truck
    reward, _ = tracker.update(make_state(map_group=0, map_num=9))
    assert reward == 5.0
    assert "back_outside" not in tracker.fired
    # Outside with clock set: clock_set + back_outside fire
    reward, _ = tracker.update(make_state(map_group=0, map_num=9, clock_set=True))
    assert reward == 25.0
    assert "back_outside" in tracker.fired


def test_full_chain_sums_to_155():
    tracker = MilestoneTracker(starter_milestones())
    total = 0.0
    steps = (
        make_state(map_group=25, map_num=40),                 # in the truck: nothing
        make_state(map_group=0, map_num=9),                   # exit_truck +5
        make_state(map_group=1, map_num=0),                   # enter_house +5
        make_state(map_group=1, map_num=0, clock_set=True),   # clock_set +15
        make_state(map_group=0, map_num=9, clock_set=True),   # back_outside +10
        make_state(map_group=0, map_num=16, clock_set=True),  # reach_route_101 +20
        make_state(map_group=0, map_num=16, clock_set=True, party_count=1),  # starter +100
    )
    terminated = False
    for state in steps:
        reward, terminated = tracker.update(state)
        total += reward
    assert total == 155.0
    assert terminated is True
    assert len(tracker.fired) == 6
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv/bin/pytest tests/test_milestones.py -q`
Expected: pre-existing tests PASS (neutral defaults keep them intact), new tests FAIL (milestones missing).

- [ ] **Step 3: Implement**

In `env/milestones.py`, replace the constants block and `starter_milestones()`:

```python
# Map IDs from pret/pokeemerald data/maps/map_groups.json
LITTLEROOT = (0, 9)
ROUTE_101 = (0, 16)
# Brendan's and May's house ground floors; the player spawns in one of them.
PLAYER_HOUSES_1F = frozenset({(1, 0), (1, 2)})
```

```python
def starter_milestones() -> tuple[Milestone, ...]:
    """M6 chain: play the scripted intro, leave town northward, obtain the starter."""
    return (
        Milestone(
            "exit_truck",
            lambda s: (s.map_group, s.map_num) == LITTLEROOT,
            5.0,
        ),
        Milestone(
            "enter_house",
            lambda s: (s.map_group, s.map_num) in PLAYER_HOUSES_1F,
            5.0,
        ),
        Milestone(
            "clock_set",
            lambda s: s.clock_set,
            15.0,
        ),
        Milestone(
            "back_outside",
            lambda s: s.clock_set and (s.map_group, s.map_num) == LITTLEROOT,
            10.0,
        ),
        Milestone(
            "reach_route_101",
            lambda s: (s.map_group, s.map_num) == ROUTE_101,
            20.0,
        ),
        Milestone(
            "starter_obtained",
            lambda s: s.party_count >= 1,
            100.0,
            terminal=True,
        ),
    )
```

`Milestone` and `MilestoneTracker` are unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_milestones.py -q`
Expected: all PASS (13 tests).

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check env/milestones.py tests/test_milestones.py`

```bash
git add env/milestones.py tests/test_milestones.py
git commit -m "feat: intro milestone chain from truck start (155-point chain)"
```

---

### Task 3: FakeEmulator clock flag + env chain test

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_env.py`

- [ ] **Step 1: Update FakeEmulator**

In `tests/conftest.py`:

1. In `__init__`, change the default map from `(1, 2)` (now May's house — would fire `enter_house`) to neutral Oldale, and add the clock attribute:

```python
        self.map_group = 0
        self.map_num = 10  # Oldale: neutral map, fires no milestone
        self.clock_set = False
```

2. In `read_bytes`, add a branch **before** the final `return b"\x00" * length` fallback:

```python
        # Flags byte 10 holds FLAG_SET_WALL_CLOCK (0x51) at bit 1.
        if address == self._sb1 + 0x1270 + 10:
            return (b"\x02" if self.clock_set else b"\x00") * length
```

3. In `load_state`, add `self.clock_set = False` next to the party reset.

- [ ] **Step 2: Write the failing env chain test**

Add at the end of `tests/test_env.py`:

```python
def test_intro_chain_pays_each_milestone_once():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_state=b"state", max_steps=50)
    env.reset()

    emu.map_group, emu.map_num = 0, 9  # exit the truck into Littleroot
    _, reward, _, _, info = env.step(0)
    assert reward >= 5.0
    assert "exit_truck" in info["milestones"]

    emu.map_group, emu.map_num = 1, 2  # enter May's house
    _, reward, _, _, info = env.step(0)
    assert reward >= 5.0
    assert "enter_house" in info["milestones"]

    emu.clock_set = True  # set the bedroom wall clock
    _, reward, _, _, info = env.step(0)
    assert reward >= 15.0
    assert "clock_set" in info["milestones"]

    emu.map_group, emu.map_num = 0, 9  # back outside with the clock set
    _, reward, terminated, _, info = env.step(0)
    assert reward >= 10.0
    assert "back_outside" in info["milestones"]
    assert terminated is False
```

- [ ] **Step 3: Run the env and milestone suites**

Run: `.venv/bin/pytest tests/test_env.py tests/test_milestones.py tests/test_game_state.py -q`
Expected: all PASS. (`test_staying_put_gives_zero_reward_after_first_visit` stays green because the neutral default map fires nothing.)

- [ ] **Step 4: Lint and commit**

Run: `.venv/bin/ruff check tests/conftest.py tests/test_env.py`

```bash
git add tests/conftest.py tests/test_env.py
git commit -m "test: fake clock flag and env-level intro chain coverage"
```

---

### Task 4: real-ROM assertions and module index

**Files:**
- Modify: `tests/test_game_state.py` (real-ROM test)
- Modify: `docs/architecture/modules.md`

- [ ] **Step 1: Update the real-ROM savestate test**

Replace `test_real_rom_state_after_initial_savestate` with:

```python
@requires_rom
def test_real_rom_state_after_initial_savestate(rom_path):
    """Sanity check against the real game: truck-start savestate is readable."""
    from emulator.gba import GbaEmulator
    from env.game_state import FLAG_SET_WALL_CLOCK

    state_file = Path("states/initial.state")
    if not state_file.is_file():
        pytest.skip("states/initial.state not created yet")
    emu = GbaEmulator(rom_path)
    emu.load_state(state_file.read_bytes())
    emu.step(frames=10)
    reader = EmeraldReader(emu.read_bytes)
    state = reader.player_state()
    assert state is not None
    assert (state.map_group, state.map_num) == (25, 40)  # InsideOfTruck
    assert state.party_count == 0
    assert state.badges == 0
    assert state.clock_set is False
    assert reader.party_levels() == []
    assert reader.read_flag(FLAG_SET_WALL_CLOCK) is False
```

- [ ] **Step 2: Run with the ROM**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest -q`
Expected: full suite PASS (~47 tests).

- [ ] **Step 3: Update modules.md**

In `docs/architecture/modules.md`, update the `env/game_state.py` entry (add `clock_set` field + `FLAG_SET_WALL_CLOCK`) and the `env/milestones.py` entry (6-milestone intro chain, 155 points, constants `LITTLEROOT` / `PLAYER_HOUSES_1F`).

- [ ] **Step 4: Lint and commit**

Run: `.venv/bin/ruff check .`

```bash
git add tests/test_game_state.py docs/architecture/modules.md
git commit -m "test: real-ROM asserts for truck-start savestate, module index update"
```

---

### Task 5: sanity training run (controller-run, no subagent)

- [ ] **Step 1: 20k-step sanity run from scratch**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/python agent/train.py --timesteps 20000 --envs 4`
Expected: completes without error, checkpoints written to `checkpoints/`, no NaN losses. Do NOT pass `--resume` — retraining from scratch is a spec decision.

- [ ] **Step 2: Verify a fresh env episode fires no milestone at reset**

Quick check that reset info shows `milestones: []` and map `(25, 40)` via the existing tooling or a short Python snippet.

- [ ] **Step 3: Launch the long run (user-visible, background)**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/python agent/train.py --timesteps 1500000 --envs 4` in background; evaluate milestones fired per episode afterwards.
