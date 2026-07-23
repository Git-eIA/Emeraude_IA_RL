# M6.2 — meet_rival Unlock Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reward the agent for the rival's-bedroom Pokéball cutscene that unlocks Littleroot's north exit, by reading GBA event vars and adding 3 milestones (chain 165 → 190 pts).

**Architecture:** Two isolated changes following the M6 `clock_set` pattern: `EmeraldReader` learns to read event vars into a new `PlayerState.town_state` field, and `starter_milestones()` gains 3 rows between `back_outside` and `north_littleroot`.

**Tech Stack:** Python 3.12, pytest (FakeEmulator, no ROM needed), ruff.

**Context (empirical, validated by probe on BPEF 2026-07-23):** vars live at
`SaveBlock1 + 0x139C`, index `(var_id - 0x4000) * 2`, u16 little-endian.
`VAR_LITTLEROOT_TOWN_STATE = 0x4050` flips 0→1 after the Pokéball cutscene in
May's house 2F, which makes the twin guarding the Route 101 exit step aside.
May's house: 1F = map `(1, 2)`, 2F = map `(1, 3)`.

---

### Task 1: town_state read in EmeraldReader

**Files:**
- Modify: `env/game_state.py`
- Test: `tests/test_game_state.py`
- Modify: `tests/conftest.py` (FakeEmulator)
- Modify: `docs/architecture/modules.md` (env/game_state.py row)

- [ ] **Step 1: Write the failing tests**

In `tests/test_game_state.py`:

1. `build_memory` gains a keyword param `town_state: int = 0`, its buffer grows
   from `0x1400` to `0x1600` (vars array spans 0x139C..0x159C), and writes the
   var before the return:

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
    town_state: int = 0,
) -> dict[int, bytes]:
    sb1 = 0x02025A00  # arbitrary but valid EWRAM address for the fake
    save_block1 = bytearray(0x1600)  # must fit the vars array up to 0x159C
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
    # VAR_LITTLEROOT_TOWN_STATE = 0x4050 -> vars index 0x50, u16 LE
    save_block1[0x139C + 0xA0 : 0x139C + 0xA2] = town_state.to_bytes(2, "little")
    return {
        SAVE_BLOCK1_PTR: sb1.to_bytes(4, "little"),
        sb1: bytes(save_block1),
        PARTY_COUNT_ADDR: bytes([party_count]),
    }
```

2. Add two tests after `test_clock_not_set_by_default`:

```python
def test_town_state_read_into_state():
    memory = build_memory(
        x=0, y=0, map_group=0, map_num=9, badge_bits=0, party_count=0, town_state=1
    )
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.town_state == 1


def test_town_state_zero_by_default():
    memory = build_memory(x=0, y=0, map_group=0, map_num=9, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.town_state == 0
```

3. In `test_real_rom_state_after_initial_savestate`, add after the
   `clock_set is False` assert:

```python
    assert state.town_state == 0
```

In `tests/conftest.py` (FakeEmulator):

1. In `__init__`, after `self.clock_set = False`, add:

```python
        self.town_state = 0
```

2. In `read_bytes`, after the flags-byte branch, add:

```python
        # VAR_LITTLEROOT_TOWN_STATE (0x4050) -> vars offset 0x139C + 0x50 * 2.
        if address == self._sb1 + 0x139C + 0xA0:
            return self.town_state.to_bytes(2, "little")[:length]
```

3. In `load_state`, after `self.clock_set = False`, add:

```python
        self.town_state = 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_game_state.py -q`
Expected: FAIL — `AttributeError: 'PlayerState' object has no attribute 'town_state'` (or TypeError on the dataclass).

- [ ] **Step 3: Implement the var read**

In `env/game_state.py`:

1. After the `FLAG_SET_WALL_CLOCK` constant block, add:

```python
# Event vars array inside SaveBlock1. Offset verified empirically on BPEF
# (2026-07-23 probe): VAR_LITTLEROOT_INTRO_STATE stepped 0->7 during the intro.
_VARS_OFFSET = 0x139C  # offsetof(struct SaveBlock1, vars)
_VARS_START = 0x4000  # first var id (pret include/constants/vars.h)
# The twin guarding Littleroot's north exit steps aside once this var is >= 1
# (set by the Pokeball cutscene in the rival's bedroom).
VAR_LITTLEROOT_TOWN_STATE = 0x4050
```

2. `PlayerState` gains a field after `clock_set`:

```python
    town_state: int = 0
```

3. `EmeraldReader.player_state()` return gains, after the `clock_set=` line:

```python
            town_state=self._var(sb1, VAR_LITTLEROOT_TOWN_STATE),
```

4. Add a private helper after `_flag`:

```python
    def _var(self, sb1: int, var_id: int) -> int:
        raw = self._read(sb1 + _VARS_OFFSET + (var_id - _VARS_START) * 2, 2)
        return int.from_bytes(raw, "little")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_game_state.py tests/test_env.py tests/test_milestones.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Update modules.md and lint**

In `docs/architecture/modules.md`, update the `env/game_state.py` row: mention
event-var reading (`_VARS_OFFSET = 0x139C`, `VAR_LITTLEROOT_TOWN_STATE`) and
the `town_state` field on `PlayerState`.
Run: `.venv/bin/ruff check .` — expected: clean.

- [ ] **Step 6: Commit**

```bash
git add env/game_state.py tests/test_game_state.py tests/conftest.py docs/architecture/modules.md
git commit -m "feat: read VAR_LITTLEROOT_TOWN_STATE into PlayerState (event vars on BPEF)"
```

---

### Task 2: enter_rival_house / rival_upstairs / meet_rival milestones (chain = 190)

**Files:**
- Modify: `env/milestones.py`
- Test: `tests/test_milestones.py`
- Modify: `tests/test_env.py` (extend the intro-chain test)
- Modify: `docs/architecture/modules.md` (env/milestones.py row)

- [ ] **Step 1: Write the failing tests**

In `tests/test_milestones.py`:

1. `make_state` defaults gain `town_state=0`:

```python
    defaults = dict(
        x=0, y=5, map_group=0, map_num=10, badges=0, party_count=0,
        clock_set=False, town_state=0,
    )
```

2. Add these tests after `test_north_littleroot_needs_littleroot_map`:

```python
def test_enter_rival_house_requires_clock():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=1, map_num=2, clock_set=False))
    assert "enter_rival_house" not in tracker.fired
    assert "enter_house" in tracker.fired  # pre-intro visit still pays enter_house


def test_enter_rival_house_fires_with_clock():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(map_group=1, map_num=2, clock_set=True))
    assert "enter_rival_house" in tracker.fired
    # enter_house (5) + clock_set (15) + enter_rival_house (5) on a fresh tracker
    assert reward == 25.0


def test_rival_upstairs_fires_with_clock():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(map_group=1, map_num=3, clock_set=True))
    assert "rival_upstairs" in tracker.fired
    # clock_set (15) + rival_upstairs (5); (1,3) is not in PLAYER_HOUSES_1F
    assert reward == 20.0


def test_rival_upstairs_requires_clock():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=1, map_num=3, clock_set=False))
    assert "rival_upstairs" not in tracker.fired


def test_meet_rival_fires_on_town_state():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(town_state=1))
    assert "meet_rival" in tracker.fired
    assert reward == 15.0


def test_meet_rival_not_fired_at_zero():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(town_state=0))
    assert "meet_rival" not in tracker.fired
```

3. Replace `test_full_chain_sums_to_165` with:

```python
def test_full_chain_sums_to_190():
    tracker = MilestoneTracker(starter_milestones())
    total = 0.0
    steps = (
        make_state(map_group=25, map_num=40),                 # in the truck: nothing
        make_state(map_group=0, map_num=9),                   # exit_truck +5
        make_state(map_group=1, map_num=0),                   # enter_house +5
        make_state(map_group=1, map_num=0, clock_set=True),   # clock_set +15
        make_state(map_group=0, map_num=9, clock_set=True),   # back_outside +10
        make_state(map_group=1, map_num=2, clock_set=True),   # enter_rival_house +5
        make_state(map_group=1, map_num=3, clock_set=True),   # rival_upstairs +5
        make_state(map_group=1, map_num=3, clock_set=True, town_state=1),  # meet_rival +15
        make_state(map_group=0, map_num=9, clock_set=True, town_state=1, y=1),  # north +10
        make_state(map_group=0, map_num=16, clock_set=True, town_state=2),  # route_101 +20
        make_state(map_group=0, map_num=16, clock_set=True, town_state=2, party_count=1),  # starter +100
    )
    terminated = False
    for state in steps:
        reward, terminated = tracker.update(state)
        total += reward
    assert total == 190.0
    assert terminated is True
    assert len(tracker.fired) == 10
```

In `tests/test_env.py`, in `test_intro_chain_pays_each_milestone_once`, replace
the final block (from `emu.y = 1` to the end) with:

```python
    emu.map_group, emu.map_num = 1, 2  # into the rival's house, clock now set
    _, reward, _, _, info = env.step(0)
    assert reward >= 4.9  # milestone +5, but revisit penalty -0.01
    assert "enter_rival_house" in info["milestones"]

    emu.map_group, emu.map_num = 1, 3  # upstairs to the rival's bedroom
    _, reward, _, _, info = env.step(0)
    assert reward >= 5.0
    assert "rival_upstairs" in info["milestones"]

    emu.town_state = 1  # Pokeball cutscene watched: exit unlocked
    _, reward, _, _, info = env.step(0)
    assert reward >= 14.9  # milestone +15, but revisit penalty -0.01
    assert "meet_rival" in info["milestones"]

    emu.map_group, emu.map_num = 0, 9
    emu.y = 1  # walk up to the northern exit
    _, reward, _, _, info = env.step(0)
    assert reward >= 10.0
    assert "north_littleroot" in info["milestones"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_milestones.py tests/test_env.py -q`
Expected: FAIL — `meet_rival`/`enter_rival_house` never fired, chain sums to 165.

- [ ] **Step 3: Implement the milestones**

In `env/milestones.py`:

1. After the `PLAYER_HOUSES_1F` constant, add:

```python
# The rival's (May's) house. The Pokeball cutscene in the upstairs bedroom sets
# VAR_LITTLEROOT_TOWN_STATE to 1, which makes the twin guarding the Route 101
# exit step aside — without it the agent is pushed back at the north edge.
MAYS_HOUSE_1F = (1, 2)
MAYS_HOUSE_2F = (1, 3)
```

2. Docstring of `starter_milestones()` becomes:

```python
    """M6 chain: intro, rival's Pokeball cutscene, leave town north, get the starter.

    Total reward: 190 points across 10 milestones.
    """
```

3. Insert between `back_outside` and `north_littleroot`:

```python
        Milestone(
            "enter_rival_house",
            lambda s: s.clock_set and (s.map_group, s.map_num) == MAYS_HOUSE_1F,
            5.0,
        ),
        Milestone(
            "rival_upstairs",
            lambda s: s.clock_set and (s.map_group, s.map_num) == MAYS_HOUSE_2F,
            5.0,
        ),
        Milestone(
            "meet_rival",
            lambda s: s.town_state >= 1,
            15.0,
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_milestones.py tests/test_env.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Update modules.md and lint**

In `docs/architecture/modules.md`, update the `env/milestones.py` row:
10-milestone chain, 190 pts, `MAYS_HOUSE_1F`/`MAYS_HOUSE_2F`.
Run: `.venv/bin/ruff check .` — expected: clean.

- [ ] **Step 6: Commit**

```bash
git add env/milestones.py tests/test_milestones.py tests/test_env.py docs/architecture/modules.md
git commit -m "feat: reward the rival's Pokeball cutscene that unlocks Route 101 (chain 190 pts)"
```

---

### Task 3: Full validation with ROM

- [ ] Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest -q`
Expected: all tests pass (63+), ROM-gated tests included.
- [ ] Run: `.venv/bin/ruff check .` — expected: clean.
