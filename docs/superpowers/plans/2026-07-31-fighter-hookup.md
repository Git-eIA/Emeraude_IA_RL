# Fighter Hookup on Grind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Explorer's `grind` mode win the wild battle it triggers, by driving the already-trained Fighter policy through the battle to completion.

**Architecture:** Extract the battle-turn choreography (currently private to `BattleEmeraldEnv`) into a shared `env/battle_turn.py`. Add `env/battle_player.py::play_battle` that loops a caller-injected `predict` over that choreography until the battle ends. Wire it into `env/orders.py::_execute_grind` behind two optional injected deps so `orders.py` stays SB3-free.

**Tech Stack:** Python 3.12, numpy, Gymnasium, Stable-Baselines3 (only in prod call sites and the gated ROM smoke), pytest.

---

## File Structure

- `env/battle_turn.py` (NEW) — pure, stateless battle-turn choreography + 17-dim observation + timing constants. One responsibility: how to play one battle turn and encode one observation.
- `env/battle_env.py` (MODIFY) — delegate its private turn/observation methods to `env/battle_turn.py`. Behavior unchanged.
- `env/battle_player.py` (NEW) — `play_battle(emulator, move_type_fn, predict, max_turns)` runs a live, ongoing battle to an outcome. One responsibility: drive a policy through a battle.
- `env/orders.py` (MODIFY) — `_execute_grind` hands off to `play_battle` when Fighter deps are supplied.
- `tests/test_battle_turn.py` (NEW), `tests/test_battle_player.py` (NEW), `tests/test_battle_player_rom.py` (NEW), `tests/test_orders.py` (MODIFY).

Everything runs from the worktree root `/Users/_eloi/Projets/Emu-p4-fighter-hookup`. The `.venv`, `roms/`, `checkpoints/`, and `states/` live in the MAIN repo `/Users/_eloi/Projets/Emu`; pure tests need neither. Run pure tests with:

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup \
/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```

---

## Task 1: Extract shared battle-turn choreography

**Files:**
- Create: `env/battle_turn.py`
- Modify: `env/battle_env.py`
- Test: `tests/test_battle_turn.py`, existing `tests/test_battle_env.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_battle_turn.py`:

```python
"""Pure tests for the shared battle-turn choreography (no ROM)."""
from __future__ import annotations

import numpy as np

from env.battle_turn import OBS_SIZE, observation, select_move
from env.game_state import BattleState, MoveInfo


def _state() -> BattleState:
    moves = (MoveInfo(1, 10), MoveInfo(0, 0), MoveInfo(0, 0), MoveInfo(0, 0))
    return BattleState(
        in_battle=True,
        my_hp=10, my_max_hp=20, my_level=5, my_types=(12, 12), my_moves=moves,
        opp_hp=9, opp_max_hp=18, opp_level=5, opp_types=(10, 10), opp_species=4,
        outcome=0, last_move_super_effective=False,
    )


def test_observation_shape_and_bounds() -> None:
    obs = observation(_state(), move_type_fn=lambda mid: 12)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    assert obs[0] == 0.5  # my_hp / my_max_hp = 10/20


class _MenuEmulator:
    """Reader+emulator: menu flag flips off after the first A, counts presses."""

    def __init__(self) -> None:
        self.presses: list[int] = []
        self._at_menu = True

    def step(self, keys: int, frames: int) -> None:
        if keys != 0:
            self.presses.append(keys)
            self._at_menu = False  # first real press opens the move list

    def at_action_menu(self) -> bool:
        return self._at_menu


def test_select_move_opens_list_then_navigates_and_commits() -> None:
    from emulator import buttons

    emu = _MenuEmulator()
    select_move(emu, emu, action=2)
    downs = sum(1 for k in emu.presses if k & buttons.KEY_DOWN)
    a_presses = sum(1 for k in emu.presses if k & buttons.KEY_A)
    assert downs == 2  # navigate to slot 2
    assert a_presses >= 2  # open the list + commit
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_battle_turn.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'env.battle_turn'`.

- [ ] **Step 3: Create `env/battle_turn.py`**

Move the constants and choreography out of `env/battle_env.py` verbatim (behavior-preserving), taking `emulator` and `reader` as explicit parameters:

```python
"""Shared battle-turn choreography: one turn of input + the 17-dim observation.

Extracted from BattleEmeraldEnv so the training env and the live Fighter player
share exactly one copy of the turn timing and the observation layout.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from emulator import buttons
from env.game_state import BattleState

# Button pulse: hold then release, so each press is a distinct edge. Pressing
# with no release frame in between makes the GBA debounce consecutive holds
# into a single press, which stalls all menu navigation.
PRESS_HOLD_FRAMES = 6
PRESS_RELEASE_FRAMES = 10
SETTLE_FRAMES = 8  # let a freshly-opened menu accept input before pressing
OPEN_MENU_TRIES = 10  # A-presses allowed to open the move list from the menu
MAX_ADVANCE_PRESSES = 120  # cap the A-spam loop so a stuck battle can't hang
NUM_TYPES = 18  # types 0..17
MAX_PP = 40.0
OBS_SIZE = 17

MoveTypeFn = Callable[[int], int]


def press(emulator: Any, key: int) -> None:
    """Hold then release so the GBA registers a distinct press."""
    emulator.step(key, PRESS_HOLD_FRAMES)
    emulator.step(0, PRESS_RELEASE_FRAMES)


def select_move(emulator: Any, reader: Any, action: int) -> None:
    """From the action menu: open the move list, move to `action`, commit.

    Pressing A until the menu opens absorbs an input eaten by the open animation.
    """
    for _ in range(OPEN_MENU_TRIES):
        if not reader.at_action_menu():
            break
        press(emulator, buttons.KEY_A)
    for _ in range(action):
        press(emulator, buttons.KEY_DOWN)
    press(emulator, buttons.KEY_A)


def advance_to_menu(emulator: Any, reader: Any) -> None:
    """Press A to clear dialogue until back at the action menu or battle end."""
    for _ in range(MAX_ADVANCE_PRESSES):
        state = reader.battle_state()
        if state.outcome != 0 or not state.in_battle:
            return
        if reader.at_action_menu():
            emulator.step(0, SETTLE_FRAMES)
            return
        press(emulator, buttons.KEY_A)


def observation(state: BattleState, move_type_fn: MoveTypeFn) -> np.ndarray:
    """The 17-dim RAM observation the Fighter policy was trained on."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0] = _frac(state.my_hp, state.my_max_hp)
    obs[1] = min(state.my_level / 100.0, 1.0)
    obs[2] = _frac(state.opp_hp, state.opp_max_hp)
    obs[3] = min(state.opp_level / 100.0, 1.0)
    # Clamp type encodings: a RAM-transition garbage read can return a byte
    # above the valid 0..17 range, which would break the Box [0, 1] bound.
    obs[4] = min(state.my_types[0] / NUM_TYPES, 1.0)
    obs[5] = min(state.my_types[1] / NUM_TYPES, 1.0)
    obs[6] = min(state.opp_types[0] / NUM_TYPES, 1.0)
    obs[7] = min(state.opp_types[1] / NUM_TYPES, 1.0)
    for i, move in enumerate(state.my_moves):
        obs[8 + 2 * i] = min(move_type_fn(move.move_id) / NUM_TYPES, 1.0)
        obs[9 + 2 * i] = min(move.pp / MAX_PP, 1.0)
    obs[16] = 1.0 if state.in_battle else 0.0
    return obs


def _frac(hp: int, max_hp: int) -> float:
    return hp / max_hp if max_hp > 0 else 0.0
```

- [ ] **Step 4: Refactor `env/battle_env.py` to delegate**

Make these edits to `env/battle_env.py`:

1. Delete the moved module-level constant block (`PRESS_HOLD_FRAMES`, `PRESS_RELEASE_FRAMES`, `SETTLE_FRAMES`, `OPEN_MENU_TRIES`, `MAX_ADVANCE_PRESSES`, `NUM_TYPES`, `MAX_PP`, `OBS_SIZE`). KEEP `RESET_WARMUP_FRAMES = 4` (battle_env-specific, not moved).
2. Add the import `from env.battle_turn import OBS_SIZE, advance_to_menu, observation, press, select_move` (alongside the existing `from env.battle_rewards import BattleRewardTracker`). `OBS_SIZE` is still needed for `observation_space`.
3. Remove `from emulator import buttons` — after the delegation edits `battle_env.py` no longer references `buttons` directly (keeps ruff F401-clean).
4. Keep `MoveTypeFn = Callable[[int], int]` in battle_env (still used in its ctor signature).
5. Replace the method bodies to delegate:

```python
    def _press(self, key: int) -> None:
        press(self.emulator, key)

    def _select_move(self, action: int) -> None:
        select_move(self.emulator, self._reader, int(action))

    def _advance_to_menu(self) -> None:
        advance_to_menu(self.emulator, self._reader)

    def _observation(self, state: BattleState) -> np.ndarray:
        return observation(state, self._move_type_fn)
```

6. Delete the now-unused module-level `_frac` from `battle_env.py` (it moved to `battle_turn.py`).

- [ ] **Step 5: Run tests to verify they pass**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_battle_turn.py tests/test_battle_env.py -q
```
Expected: PASS (new battle_turn tests + all existing battle_env tests still green — the refactor is behavior-preserving).

- [ ] **Step 6: Lint**

```bash
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/battle_turn.py env/battle_env.py tests/test_battle_turn.py
```
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add env/battle_turn.py env/battle_env.py tests/test_battle_turn.py
git commit -m "refactor: extract shared battle-turn choreography into env/battle_turn.py"
```

---

## Task 2: Live battle player

**Files:**
- Create: `env/battle_player.py`
- Test: `tests/test_battle_player.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_battle_player.py`:

```python
"""Pure tests for play_battle: drive a scripted battle to an outcome (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.battle_player import play_battle
from env.game_state import BATTLE_MON_SIZE


def _u16(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class _ScriptedBattle:
    """Menu->moves->resolving loop. Each committed turn deals `my_dmg` to the foe
    and `foe_dmg` to us. The foe faints -> outcome 1 (won); we faint -> outcome 2
    (lost). Mirrors the real ROM's flag-driven turn loop.
    """

    _RESOLVE_PRESSES = 2

    def __init__(self, my_dmg: int, foe_dmg: int) -> None:
        self._my_dmg = my_dmg
        self._foe_dmg = foe_dmg
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            self._commit_turn()
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def _commit_turn(self) -> None:
        self._opp_hp = max(0, self._opp_hp - self._my_dmg)
        self._my_hp = max(0, self._my_hp - self._foe_dmg)
        if self._opp_hp == 0:
            self._outcome = 1
        elif self._my_hp == 0:
            self._outcome = 2
        self._phase = "resolving"
        self._resolve_left = self._RESOLVE_PRESSES

    def _mon(self, *, hp: int, max_hp: int) -> bytearray:
        buf = bytearray(BATTLE_MON_SIZE)
        buf[0x00:0x02] = _u16(1)
        for i in range(4):
            buf[0x0C + 2 * i : 0x0C + 2 * i + 2] = _u16(1 if i == 0 else 0)
            buf[0x24 + i] = 10 if i == 0 else 0
        buf[0x21], buf[0x22] = 12, 12
        buf[0x28:0x2A] = _u16(hp)
        buf[0x2A] = 5
        buf[0x2C:0x2E] = _u16(max_hp)
        return buf

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            ACTION_MENU_VALUE,
            GBATTLE_ACTION_MENU_ADDR,
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_ACTION_MENU_ADDR:
            return bytes([ACTION_MENU_VALUE if self._phase == "menu" else 0])
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        if pbase <= addr < pbase + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._my_hp, max_hp=19)
            off = addr - pbase
            return bytes(buf[off : off + size])
        if obase <= addr < obase + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._opp_hp, max_hp=18)
            off = addr - obase
            return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def _predict(_obs) -> int:
    return 0  # always use move slot 0


def test_play_battle_wins() -> None:
    emu = _ScriptedBattle(my_dmg=6, foe_dmg=2)  # foe (18hp) faints in 3 turns
    assert play_battle(emu, move_type_fn=lambda mid: 12, predict=_predict) == "won"


def test_play_battle_loses() -> None:
    emu = _ScriptedBattle(my_dmg=1, foe_dmg=19)  # we (19hp) faint first
    assert play_battle(emu, move_type_fn=lambda mid: 12, predict=_predict) == "lost"


def test_play_battle_times_out() -> None:
    emu = _ScriptedBattle(my_dmg=0, foe_dmg=0)  # nobody faints
    result = play_battle(
        emu, move_type_fn=lambda mid: 12, predict=_predict, max_turns=4
    )
    assert result == "battle_timeout"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_battle_player.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'env.battle_player'`.

- [ ] **Step 3: Create `env/battle_player.py`**

```python
"""Drive a live, ongoing battle with a trained Fighter policy to an outcome.

Unlike BattleEmeraldEnv (which trains, resets, and shapes reward), this just
plays: the battle is already running (grind triggered it), and a caller-injected
`predict` chooses each move. Kept free of SB3/torch — production wraps a loaded
PPO model into `predict`.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from env.battle_turn import advance_to_menu, observation, select_move
from env.game_state import BattleReader

PredictFn = Callable[[np.ndarray], int]
MoveTypeFn = Callable[[int], int]


def play_battle(
    emulator: Any,
    move_type_fn: MoveTypeFn,
    predict: PredictFn,
    max_turns: int = 64,
) -> str:
    """Play the ongoing battle to the end.

    Returns "won" (outcome bit 0x1 set), "lost" (any other terminal outcome),
    or "battle_timeout" (max_turns reached without a terminal outcome).
    """
    reader = BattleReader(emulator.read_bytes)
    advance_to_menu(emulator, reader)  # reach the first action menu (or battle end)
    for _ in range(max_turns):
        state = reader.battle_state()
        if state.outcome != 0 or not state.in_battle:
            return _result(state.outcome)
        action = predict(observation(state, move_type_fn))
        select_move(emulator, reader, int(action))
        advance_to_menu(emulator, reader)
    state = reader.battle_state()
    if state.outcome != 0 or not state.in_battle:
        return _result(state.outcome)
    return "battle_timeout"


def _result(outcome: int) -> str:
    return "won" if outcome & 0x1 else "lost"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_battle_player.py -q
```
Expected: PASS (3 tests: won, lost, battle_timeout).

- [ ] **Step 5: Lint**

```bash
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/battle_player.py tests/test_battle_player.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add env/battle_player.py tests/test_battle_player.py
git commit -m "feat: play_battle drives a trained Fighter through a live battle"
```

---

## Task 3: Wire the Fighter into grind

**Files:**
- Modify: `env/orders.py`
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orders.py` (after the existing grind tests). The `GrassBattleWorld` fake first counts d-pad steps to trigger a battle, then serves battle RAM so `play_battle` can drive it to a win:

```python
# ---------------------------------------------------------------------------
# Grind + Fighter hookup tests
# ---------------------------------------------------------------------------


def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class GrassBattleWorld:
    """Treads to a battle, then plays a scripted battle the Fighter wins in 3 turns."""

    _RESOLVE_PRESSES = 2

    def __init__(self, map_id: tuple[int, int], cell: tuple[int, int],
                 steps_to_encounter: int = 3) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_enc = steps_to_encounter
        self._steps = 0
        self._battle = False
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        from emulator import buttons

        if not self._battle:
            if _KEY_TO_DIR.get(keys) is not None:
                self._steps += 1
                if self._steps >= self._to_enc:
                    self._battle = True
            return
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
            self._phase = "resolving"
            self._resolve_left = self._RESOLVE_PRESSES
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

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


def test_grind_with_fighter_wins_the_battle() -> None:
    world = GrassBattleWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(
        order, world, world, memory, WallMap(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "won"


def test_grind_without_fighter_deps_still_returns_encounter_started() -> None:
    world = GrassBattleWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "encounter_started"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py::test_grind_with_fighter_wins_the_battle -q
```
Expected: FAIL — `execute_order` has no `move_type_fn`/`predict` params (TypeError: unexpected keyword argument).

- [ ] **Step 3: Modify `env/orders.py`**

Add the import at the top (after the existing `from env.map_traveler import travel_to`):

```python
from env.battle_player import play_battle
```

Change `execute_order`'s signature and its grind branch:

```python
def execute_order(
    order: Order,
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
```

In the body, pass the deps to grind:

```python
    if order.mode == "grind":
        return _execute_grind(
            emulator, reader, memory, wallmap,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
```

Change `_execute_grind`:

```python
def _execute_grind(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Travel to a known grass cell, tread until a wild battle starts, then—if a
    Fighter is supplied—play the battle to an outcome.

    Returns "no_grass_spot_known" | a travel_to pass-through | "no_encounter" |
    "encounter_started" (no Fighter) | "won" | "lost" | "battle_timeout".
    """
    spots = memory.cells_labeled("has_grass")
    if not spots:
        return "no_grass_spot_known"
    goal_map, goal_cell = spots[0]   # v1: first known spot (nearest-choice is later)
    outcome = travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    result = _walk_until_encounter(emulator, reader)
    if result != "encounter_started":
        return result                # no_encounter
    if move_type_fn is None or predict is None:
        return "encounter_started"   # no Fighter wired: stop at the encounter
    return play_battle(emulator, move_type_fn, predict)
```

Also update the `execute_order` docstring "Returns …" line to append `| "won" | "lost" | "battle_timeout"`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -q
```
Expected: PASS (all existing order tests + the 2 new ones).

- [ ] **Step 5: Lint**

```bash
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/orders.py tests/test_orders.py
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "feat: grind hands off to the Fighter to win the triggered battle"
```

---

## Task 4: Load-bearing ROM smoke

**Files:**
- Test: `tests/test_battle_player_rom.py`

- [ ] **Step 1: Write the gated smoke test**

Create `tests/test_battle_player_rom.py`. It loads a real battle savestate + the real Fighter checkpoint and asserts `play_battle` wins (the Fighter is 10/10 on these states). It skips cleanly when the ROM, checkpoint, or states are absent (they live in the main repo, gitignored).

```python
"""Gated ROM smoke: the real Fighter wins a real wild battle via play_battle."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_MAIN = Path("/Users/_eloi/Projets/Emu")
_ROM = os.environ.get("POKEMON_EMERALD_ROM")
_CKPT = _MAIN / "checkpoints" / "fighter" / "ppo_fighter_final.zip"
_STATES = sorted((_MAIN / "states" / "battles").glob("*.state"))
_STATES = [p for p in _STATES if p.name != "probe.state"]


@pytest.mark.skipif(_ROM is None, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _CKPT.exists(), reason="Fighter checkpoint missing")
@pytest.mark.skipif(not _STATES, reason="no battle savestates")
def test_fighter_wins_a_real_battle_via_play_battle() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.battle_player import play_battle

    emu = GbaEmulator(_ROM)
    emu.load_state(_STATES[0].read_bytes())
    emu.step(0, 4)  # let the emulator render after load_state

    model = PPO.load(str(_CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    result = play_battle(emu, make_move_type_fn(emu), predict)
    assert result == "won"
```

- [ ] **Step 2: Run the smoke (from the MAIN repo, where ROM/checkpoint/states live)**

```bash
cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup \
.venv/bin/python -m pytest /Users/_eloi/Projets/Emu-p4-fighter-hookup/tests/test_battle_player_rom.py -q
```
Expected: PASS (`won`). If ROM/checkpoint/states are missing it SKIPS — that is acceptable, but the intent is for it to run and pass on this machine.

- [ ] **Step 3: Lint**

```bash
/Users/_eloi/Projets/Emu/.venv/bin/ruff check tests/test_battle_player_rom.py
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_battle_player_rom.py
git commit -m "test: load-bearing ROM smoke — real Fighter wins via play_battle"
```

---

## Final: Full suite + lint

- [ ] **Step 1: Run the whole pure suite**

```bash
PYTHONPATH=/Users/_eloi/Projets/Emu-p4-fighter-hookup /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```
Expected: all pass (the ROM smokes skip without env vars, or pass with them).

- [ ] **Step 2: Ruff the whole change**

```bash
/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/ tests/
```
Expected: no errors.
