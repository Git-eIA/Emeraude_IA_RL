# M5 — Story Milestones & Starter Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The PPO agent gets milestone rewards for story progress and episodes end in success when the starter is obtained (party_count 0 → 1).

**Architecture:** Three additions to the `env/` layer, nothing changes in `emulator/` or `agent/`: (1) `EmeraldReader` learns `read_flag()` and `party_levels()`; (2) new `env/milestones.py` holds an extensible one-shot milestone table; (3) `PokemonEmeraldEnv.step()` sums exploration + milestones + level rewards and terminates on the terminal milestone.

**Tech Stack:** Same as M1-M4 (Python 3.12, Gymnasium, pytest). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-22-m5-starter-milestones-design.md`

**Address provenance (already verified pattern):** `gPlayerParty` sits at EWRAM `0x020244EC`, immediately after `gPlayerPartyCount` (`0x020244E9`, already in use). Neither symbol has an `F:` override in pokebot-gen3 `modules/data/symbols/patches/language/pokeemerald.yml`, so BPEF = BPEE, same reasoning as the existing constants. `struct Pokemon` is 100 bytes; `level` is a `u8` at offset 84 in the unencrypted battle section (pret/pokeemerald `include/pokemon.h`). `MAP_ROUTE101 = (0, 16)` from pret `include/constants/map_groups.h` (savestate already confirmed Littleroot = (0, 9) with this table).

**Reward scale (from spec):** tile +1 (existing) ≪ Route 101 +20 ≪ starter +100 (terminal). Party level sum: +5 per level gained, once. Every milestone fires at most once per episode.

---

## File Structure

- Modify: `env/game_state.py` — `read_flag()`, `party_levels()`, badge code refactored on shared flag helper
- Create: `env/milestones.py` — `Milestone`, `MilestoneTracker`, `starter_milestones()`
- Modify: `env/rewards.py` — add `LevelRewardTracker`
- Modify: `env/pokemon_env.py` — wire trackers, `terminated`, single RAM read per step
- Modify: `tests/conftest.py` — FakeEmulator grows party/map controls
- Create: `tests/test_milestones.py`
- Modify: `tests/test_game_state.py`, `tests/test_rewards.py`, `tests/test_env.py`
- Modify: `docs/architecture/modules.md` — new row + updated rows

---

### Task 1: RAM readers — `read_flag()` and `party_levels()`

**Files:**
- Modify: `env/game_state.py`
- Test: `tests/test_game_state.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_game_state.py` (note: `build_memory` returns a dict you can extend; `PARTY_ADDR`, `PARTY_MON_SIZE`, `PARTY_LEVEL_OFFSET` are new public constants):

```python
def add_party(memory: dict[int, bytes], levels: list[int]) -> None:
    """Add gPlayerParty structs with the given levels to fake memory."""
    from env.game_state import PARTY_ADDR, PARTY_LEVEL_OFFSET, PARTY_MON_SIZE

    party = bytearray(PARTY_MON_SIZE * 6)
    for slot, level in enumerate(levels):
        party[slot * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET] = level
    memory[PARTY_ADDR] = bytes(party)


def test_read_flag_set_and_unset():
    # badge_bits=0b1 sets flag 0x867; flag 0x868 stays clear
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0b1, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.read_flag(0x867) is True
    assert reader.read_flag(0x868) is False


def test_read_flag_invalid_pointer_is_false():
    memory = {SAVE_BLOCK1_PTR: (0x00000000).to_bytes(4, "little")}
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.read_flag(0x867) is False


def test_party_levels_empty():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.party_levels() == []


def test_party_levels_two_pokemon():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=2)
    add_party(memory, [5, 12])
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.party_levels() == [5, 12]


def test_party_levels_count_clamped_to_six():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=200)
    add_party(memory, [5, 5, 5, 5, 5, 5])
    reader = EmeraldReader(make_fake_read(memory))
    assert len(reader.party_levels()) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_game_state.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (PARTY_ADDR, read_flag, party_levels missing)

- [ ] **Step 3: Implement in `env/game_state.py`**

Add after `PARTY_COUNT_ADDR`:

```python
# EWRAM address of gPlayerParty: 6 x 100-byte struct Pokemon, directly after
# gPlayerPartyCount. No F: override in pokebot-gen3 language patches -> BPEF = BPEE.
PARTY_ADDR = 0x020244EC
PARTY_MON_SIZE = 100
PARTY_LEVEL_OFFSET = 84  # u8 level, unencrypted battle section (pret include/pokemon.h)
```

Refactor `EmeraldReader` — extract the pointer check, share flag logic, add the two public readers:

```python
    def player_state(self) -> PlayerState | None:
        """Current player state, or None while save blocks are relocating."""
        sb1 = self._save_block1()
        if sb1 is None:
            return None
        pos = self._read(sb1 + _POS_OFFSET, 4)
        location = self._read(sb1 + _LOCATION_OFFSET, 2)
        return PlayerState(
            x=int.from_bytes(pos[0:2], "little", signed=True),
            y=int.from_bytes(pos[2:4], "little", signed=True),
            map_group=location[0],
            map_num=location[1],
            badges=self._badge_count(sb1),
            party_count=self._read(PARTY_COUNT_ADDR, 1)[0],
        )

    def read_flag(self, flag_id: int) -> bool:
        """True if the event flag is set; False while save blocks relocate."""
        sb1 = self._save_block1()
        if sb1 is None:
            return False
        return self._flag(sb1, flag_id)

    def party_levels(self) -> list[int]:
        """Levels of the party Pokémon in slot order; empty list when no party."""
        count = min(self._read(PARTY_COUNT_ADDR, 1)[0], 6)
        return [
            self._read(PARTY_ADDR + slot * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET, 1)[0]
            for slot in range(count)
        ]

    def _save_block1(self) -> int | None:
        sb1 = int.from_bytes(self._read(SAVE_BLOCK1_PTR, 4), "little")
        if not _EWRAM_START <= sb1 < _EWRAM_END:
            return None
        return sb1

    def _flag(self, sb1: int, flag_id: int) -> bool:
        byte_index, bit_index = divmod(flag_id, 8)
        raw = self._read(sb1 + _FLAGS_OFFSET + byte_index, 1)[0]
        return bool(raw >> bit_index & 1)

    def _badge_count(self, sb1: int) -> int:
        # FLAG_BADGE01_GET..FLAG_BADGE08_GET are contiguous from _FIRST_BADGE_FLAG.
        return sum(self._flag(sb1, _FIRST_BADGE_FLAG + i) for i in range(8))
```

- [ ] **Step 4: Run tests to verify they pass (all of test_game_state, not just the new ones)**

Run: `.venv/bin/pytest tests/test_game_state.py -v`
Expected: PASS (badge refactor must keep the existing badge tests green)

- [ ] **Step 5: Lint + full suite + commit**

```bash
.venv/bin/ruff check . && .venv/bin/pytest -q
git add env/game_state.py tests/test_game_state.py
git commit -m "feat: event flag and party level RAM readers"
```

---

### Task 2: `MilestoneTracker`

**Files:**
- Create: `env/milestones.py`
- Test: `tests/test_milestones.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_milestones.py`:

```python
from __future__ import annotations

from env.game_state import PlayerState
from env.milestones import Milestone, MilestoneTracker, starter_milestones


def make_state(**overrides) -> PlayerState:
    defaults = dict(x=0, y=0, map_group=0, map_num=9, badges=0, party_count=0)
    return PlayerState(**{**defaults, **overrides})


def test_milestone_fires_once():
    tracker = MilestoneTracker((Milestone("m", lambda s: s.x > 0, 10.0),))
    assert tracker.update(make_state(x=1)) == (10.0, False)
    assert tracker.update(make_state(x=1)) == (0.0, False)


def test_condition_not_met_gives_zero():
    tracker = MilestoneTracker((Milestone("m", lambda s: s.x > 0, 10.0),))
    assert tracker.update(make_state(x=0)) == (0.0, False)


def test_terminal_milestone_terminates():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(party_count=1))
    assert reward == 100.0
    assert terminated is True


def test_route_101_milestone():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=0, map_num=16))
    assert reward == 20.0
    assert terminated is False


def test_multiple_milestones_same_step_sum():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=0, map_num=16, party_count=1))
    assert reward == 120.0
    assert terminated is True


def test_none_state_gives_zero():
    tracker = MilestoneTracker(starter_milestones())
    assert tracker.update(None) == (0.0, False)


def test_reset_clears_fired():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(party_count=1))
    tracker.reset()
    assert tracker.update(make_state(party_count=1)) == (100.0, True)


def test_fired_names_exposed():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=16))
    assert tracker.fired == frozenset({"reach_route_101"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_milestones.py -v`
Expected: FAIL with `ModuleNotFoundError: env.milestones`

- [ ] **Step 3: Implement `env/milestones.py`**

```python
"""Story milestone rewards: one-time bonuses for scripted progress events.

Extending the story chain = appending rows to starter_milestones() (or a
future table). Conditions read PlayerState only; flags-based conditions can
close over EmeraldReader.read_flag when needed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from env.game_state import PlayerState

# MAP_ROUTE101 from pret/pokeemerald include/constants/map_groups.h
ROUTE_101 = (0, 16)


@dataclass(frozen=True)
class Milestone:
    name: str
    condition: Callable[[PlayerState], bool]
    points: float
    terminal: bool = False


def starter_milestones() -> tuple[Milestone, ...]:
    """M5 chain: leave town northward, then obtain the starter."""
    return (
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


class MilestoneTracker:
    """Evaluates milestones each step; each fires at most once per episode."""

    def __init__(self, milestones: tuple[Milestone, ...]) -> None:
        self._milestones = milestones
        self._fired: set[str] = set()

    @property
    def fired(self) -> frozenset[str]:
        return frozenset(self._fired)

    def reset(self) -> None:
        self._fired.clear()

    def update(self, state: PlayerState | None) -> tuple[float, bool]:
        """Returns (reward, terminated) for this step."""
        if state is None:
            return 0.0, False
        reward = 0.0
        terminated = False
        for milestone in self._milestones:
            if milestone.name in self._fired or not milestone.condition(state):
                continue
            self._fired.add(milestone.name)
            reward += milestone.points
            terminated = terminated or milestone.terminal
        return reward, terminated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_milestones.py -v`
Expected: PASS (8/8)

- [ ] **Step 5: Lint + full suite + commit**

```bash
.venv/bin/ruff check . && .venv/bin/pytest -q
git add env/milestones.py tests/test_milestones.py
git commit -m "feat: one-shot story milestone tracker with starter chain"
```

---

### Task 3: `LevelRewardTracker`

**Files:**
- Modify: `env/rewards.py`
- Test: `tests/test_rewards.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rewards.py`:

```python
from env.rewards import REWARD_PER_LEVEL, LevelRewardTracker


def test_level_empty_party_gives_zero():
    assert LevelRewardTracker().update([]) == 0.0


def test_level_gain_pays_once():
    tracker = LevelRewardTracker()
    assert tracker.update([5]) == 5 * REWARD_PER_LEVEL
    assert tracker.update([5]) == 0.0
    assert tracker.update([6]) == REWARD_PER_LEVEL


def test_level_sum_across_party():
    tracker = LevelRewardTracker()
    tracker.update([5])
    assert tracker.update([5, 3]) == 3 * REWARD_PER_LEVEL


def test_level_drop_gives_zero_not_negative():
    tracker = LevelRewardTracker()
    tracker.update([5, 3])
    assert tracker.update([5]) == 0.0


def test_level_reset():
    tracker = LevelRewardTracker()
    tracker.update([5])
    tracker.reset()
    assert tracker.update([5]) == 5 * REWARD_PER_LEVEL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rewards.py -v`
Expected: FAIL with ImportError (LevelRewardTracker missing)

- [ ] **Step 3: Implement in `env/rewards.py`**

Append:

```python
REWARD_PER_LEVEL = 5.0


class LevelRewardTracker:
    """Pays REWARD_PER_LEVEL once per party level gained (sum over slots).

    Tracks the best sum seen so a level drop (deposit, trade) never pays
    negative reward nor re-pays on recovery.
    """

    def __init__(self) -> None:
        self._best_sum = 0

    def reset(self) -> None:
        self._best_sum = 0

    def update(self, levels: list[int]) -> float:
        total = sum(levels)
        if total <= self._best_sum:
            return 0.0
        gained = total - self._best_sum
        self._best_sum = total
        return REWARD_PER_LEVEL * gained
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rewards.py -v`
Expected: PASS

- [ ] **Step 5: Lint + full suite + commit**

```bash
.venv/bin/ruff check . && .venv/bin/pytest -q
git add env/rewards.py tests/test_rewards.py
git commit -m "feat: party level-sum reward tracker"
```

---

### Task 4: Env integration + FakeEmulator upgrades

**Files:**
- Modify: `env/pokemon_env.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_env.py`

- [ ] **Step 1: Upgrade FakeEmulator in `tests/conftest.py`**

Give the fake a controllable party and map. Replace `__init__`, `read_bytes`, and `load_state` with:

```python
    def __init__(self) -> None:
        self.x = 5
        self.y = 5
        self.map_group = 1
        self.map_num = 2
        self.party_count = 0
        self.party_levels: list[int] = []
        self.loaded_states: list[bytes] = []
        self._sb1 = 0x02025A00

    def read_bytes(self, address: int, length: int) -> bytes:
        from env import game_state

        if address == game_state.SAVE_BLOCK1_PTR:
            return self._sb1.to_bytes(4, "little")[:length]
        if address == self._sb1:
            # Coordinates are s16 in Emerald; use signed=True to handle negative values.
            return self.x.to_bytes(2, "little", signed=True) + self.y.to_bytes(2, "little", signed=True)
        if address == self._sb1 + 4:
            return bytes([self.map_group, self.map_num])[:length]
        if address == game_state.PARTY_COUNT_ADDR:
            return bytes([self.party_count])[:length]
        for slot, level in enumerate(self.party_levels):
            slot_addr = game_state.PARTY_ADDR + slot * game_state.PARTY_MON_SIZE
            if address == slot_addr + game_state.PARTY_LEVEL_OFFSET:
                return bytes([level])[:length]
        return b"\x00" * length

    def load_state(self, state: bytes) -> None:
        self.loaded_states.append(state)
        self.x, self.y = 5, 5
        self.party_count = 0
        self.party_levels = []
```

(Keep `step` and `screenshot` unchanged. The default map (1, 2) matches what the old hardcoded `bytes([1, 2])` returned, so existing tests stay green.)

- [ ] **Step 2: Run existing env tests to confirm no regression**

Run: `.venv/bin/pytest tests/test_env.py tests/test_game_state.py -v`
Expected: PASS (fake still behaves identically by default)

- [ ] **Step 3: Write the failing env tests**

Append to `tests/test_env.py`:

```python
def test_starter_terminates_episode_with_jackpot():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_state=b"state", max_steps=50)
    env.reset()
    emu.party_count = 1
    emu.party_levels = [5]
    _, reward, terminated, truncated, info = env.step(0)
    # starter +100, level sum 0->5 gives +25, plus any exploration
    assert reward >= 125.0
    assert terminated is True
    assert truncated is False
    assert "starter_obtained" in info["milestones"]


def test_route_101_milestone_pays_without_terminating():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_state=b"state", max_steps=50)
    env.reset()
    emu.map_group, emu.map_num = 0, 16
    _, reward, terminated, _, info = env.step(0)
    assert reward >= 20.0
    assert terminated is False
    assert "reach_route_101" in info["milestones"]


def test_reset_clears_milestones():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_state=b"state", max_steps=50)
    env.reset()
    emu.party_count = 1
    env.step(0)
    _, info = env.reset()
    assert info["milestones"] == []
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_env.py -v`
Expected: FAIL — no milestones in env yet (`KeyError: 'milestones'`, terminated False)

- [ ] **Step 5: Wire trackers in `env/pokemon_env.py`**

Update imports:

```python
from env.game_state import EmeraldReader
from env.milestones import MilestoneTracker, starter_milestones
from env.rewards import ExplorationTracker, LevelRewardTracker
```

In `__init__`, after `self._tracker = ExplorationTracker()`:

```python
        self._milestones = MilestoneTracker(starter_milestones())
        self._levels = LevelRewardTracker()
```

In `reset()`, after `self._tracker.reset()`:

```python
        self._milestones.reset()
        self._levels.reset()
```

Replace `step()` and `_info()` (single RAM read per step, state passed down — also fixes the final-review note about double reads):

```python
    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        keys = _ACTION_KEYS[self.ACTIONS[action]]
        self.emulator.step(keys, FRAMES_PER_ACTION)
        self._frames.append(self._current_frame())
        self._step_count += 1
        state = self._reader.player_state()
        reward = self._tracker.update(state)
        milestone_reward, terminated = self._milestones.update(state)
        reward += milestone_reward + self._levels.update(self._reader.party_levels())
        truncated = not terminated and self._step_count >= self._max_steps
        return self._observation(), reward, terminated, truncated, self._info(state)

    def _info(self, state: PlayerState | None) -> dict[str, Any]:
        return {
            "visited_tiles": self._tracker.visited_count,
            "badges": state.badges if state else 0,
            "map": (state.map_group, state.map_num) if state else None,
            "milestones": sorted(self._milestones.fired),
        }
```

Add `PlayerState` to the `env.game_state` import line, and update the `reset()` return to pass the state:

```python
        return self._observation(), self._info(self._reader.player_state())
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_env.py -v`
Expected: PASS, including the existing `check_env` test (Gymnasium API unchanged: 5-tuple, terminated now real)

- [ ] **Step 7: Lint + full suite + commit**

```bash
.venv/bin/ruff check . && .venv/bin/pytest -q
git add env/pokemon_env.py tests/conftest.py tests/test_env.py
git commit -m "feat: milestone and level rewards wired into env with terminal starter"
```

---

### Task 5: Real-ROM validation + docs

**Files:**
- Modify: `docs/architecture/modules.md`
- Modify: `tests/test_game_state.py` (extend real-ROM test)

- [ ] **Step 1: Extend the real-ROM integration test**

In `tests/test_game_state.py`, inside `test_real_rom_state_after_initial_savestate`, after the existing asserts add:

```python
    reader = EmeraldReader(emu.read_bytes)
    assert reader.party_levels() == []  # savestate party is empty
    assert reader.read_flag(0x867) is False  # no badge yet
```

(Adjust the local variable if the test builds its reader inline: reuse one `EmeraldReader` instance for all asserts.)

- [ ] **Step 2: Run the full suite with the ROM**

Run: `.venv/bin/ruff check . && POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest -q`
Expected: 0 failures, ROM tests ran (not skipped)

- [ ] **Step 3: Update `docs/architecture/modules.md`**

Add row and update the two changed rows:

```markdown
| env/milestones.py | One-shot story milestone rewards | Milestone, MilestoneTracker, starter_milestones | game_state |
```

Update `env/game_state.py` row Public API to: `EmeraldReader, PlayerState` → `EmeraldReader (player_state, read_flag, party_levels), PlayerState`.
Update `env/rewards.py` row Public API to: `ExplorationTracker, LevelRewardTracker`.
Update `env/pokemon_env.py` row Depends on to: `game_state, rewards, milestones`.

- [ ] **Step 4: Sanity training run (short)**

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/python agent/train.py --timesteps 20000 --envs 4`
Expected: completes without crash; tensorboard shows episode rewards (episodes may terminate early once the agent stumbles onto the starter — not required to succeed in 20k steps).

- [ ] **Step 5: Commit**

```bash
git add tests/test_game_state.py docs/architecture/modules.md
git commit -m "test: real-ROM coverage for flags and party levels, module index update"
```

- [ ] **Step 6: Long validation run (controller/user decision, not a subagent step)**

Launch in background and monitor: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/python agent/train.py --timesteps 500000 --envs 4`
Success signal: a growing fraction of episodes end with reward ≥ +100 (starter obtained). If zero successes after the full run, first lever per spec: strengthen the Route-101 waypoint or add an intermediate waypoint — not hyperparameter tuning.

---

## Out of scope (next plan)

- Battle detection flag + full party readers (HP, moves) — Fighter's eyes
- Scripted/learned Fighter, battle reward shaping
- Rule-based Strategist
- Milestones past the starter (rival 103, Pokédex, badge 1 chain)
- `tools/watch.py`
