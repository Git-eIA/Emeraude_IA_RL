# Trainer-battle combat primitive (Brique 3 part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `play_trainer_battle`, a sibling of `play_battle` that drives a multi-Pokémon trainer battle to a terminal outcome without falsely ending on the transient `opp max_hp == 0` during opponent send-out.

**Architecture:** Two changes. (1) The shared `advance_to_menu` helper gains a defaulted `wait_through_faint` flag so the trainer path waits through faint/send-out (wild path byte-identical at the default). (2) `play_trainer_battle` in `env/battle_player.py` reuses `observation`/`select_move`/`BattleReader`/`_result` and calls `advance_to_menu(..., wait_through_faint=True)`, ending only on `outcome != 0`. A disposable capture tool + gated ROM smoke exercise it live.

**Tech Stack:** Python 3.12, pytest (pure, no ROM/SB3 for units), stable-baselines3 (ROM smoke only, imported inside the test body).

Spec: `docs/superpowers/specs/2026-08-03-trainer-battle-primitive-design.md`.

---

## File Structure

- `env/battle_turn.py` — MODIFY `advance_to_menu` to accept `wait_through_faint: bool = False`.
- `env/battle_player.py` — ADD `play_trainer_battle`.
- `tests/test_battle_turn.py` — ADD one test for the flag.
- `tests/test_battle_player.py` — ADD a multi-mon scripted fake + 4 tests.
- `tools/capture_trainer_battle.py` — CREATE (disposable capture tool).
- `tests/test_battle_player_rom.py` — ADD one gated trainer-battle smoke.

---

### Task 1: `advance_to_menu` gains a `wait_through_faint` flag

**Files:**
- Modify: `env/battle_turn.py:50-62`
- Test: `tests/test_battle_turn.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_battle_turn.py`:

```python
from env.battle_turn import advance_to_menu


def _bs(in_battle: bool, outcome: int) -> BattleState:
    moves = (MoveInfo(1, 10), MoveInfo(0, 0), MoveInfo(0, 0), MoveInfo(0, 0))
    return BattleState(
        in_battle=in_battle,
        my_hp=10, my_max_hp=20, my_level=5, my_types=(12, 12), my_moves=moves,
        opp_hp=9, opp_max_hp=18, opp_level=5, opp_types=(10, 10), opp_species=4,
        outcome=outcome, last_move_super_effective=False,
    )


class _FaintEmulator:
    """Models a send-out: in_battle stays False (opp max_hp 0) for two ticks,
    then the live action menu appears. Serves as both emulator and reader.
    """

    def __init__(self) -> None:
        self.presses = 0
        self._ticks = 0

    def step(self, keys: int, frames: int) -> None:
        if keys != 0:
            self.presses += 1
            self._ticks += 1

    def at_action_menu(self) -> bool:
        return self._ticks >= 2

    def battle_state(self) -> BattleState:
        return _bs(in_battle=self._ticks >= 2, outcome=0)


def test_advance_to_menu_default_stops_on_not_in_battle() -> None:
    emu = _FaintEmulator()
    advance_to_menu(emu, emu)  # default wait_through_faint=False
    assert emu.presses == 0  # returns immediately on not in_battle (wild behaviour)


def test_advance_to_menu_waits_through_faint() -> None:
    emu = _FaintEmulator()
    advance_to_menu(emu, emu, wait_through_faint=True)
    assert emu.presses == 2  # presses A through send-out until the live menu
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_battle_turn.py::test_advance_to_menu_waits_through_faint -v`
Expected: FAIL — `advance_to_menu() got an unexpected keyword argument 'wait_through_faint'`.

- [ ] **Step 3: Write minimal implementation**

Replace `advance_to_menu` in `env/battle_turn.py`:

```python
def advance_to_menu(emulator: Any, reader: Any, wait_through_faint: bool = False) -> None:
    """Press A to clear dialogue until back at the action menu or battle end.

    `reader` must expose battle_state() -> BattleState and at_action_menu() -> bool.
    Wild battles (default) also return when in_battle drops. Trainer battles pass
    wait_through_faint=True: a faint/send-out momentarily reads opp max_hp==0
    (in_battle False) while the battle continues, so those stop ONLY on a terminal
    outcome or the next live action menu.
    """
    for _ in range(MAX_ADVANCE_PRESSES):
        state = reader.battle_state()
        if state.outcome != 0:
            return
        if not wait_through_faint and not state.in_battle:
            return
        if reader.at_action_menu():
            emulator.step(0, SETTLE_FRAMES)
            return
        press(emulator, buttons.KEY_A)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_battle_turn.py -v`
Expected: PASS (both new tests + the two pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add env/battle_turn.py tests/test_battle_turn.py
git commit -m "feat: advance_to_menu wait_through_faint flag for send-out dialogue"
```

---

### Task 2: `play_trainer_battle`

**Files:**
- Modify: `env/battle_player.py`
- Test: `tests/test_battle_player.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_battle_player.py` (keep the existing `_u16`, `_ScriptedBattle`, `_predict`):

```python
from env.battle_player import play_trainer_battle


class _ScriptedTrainerBattle:
    """Multi-mon trainer battle. Committing a turn deals my_dmg to the current
    foe and foe_dmg to us. When the current foe faints and more remain, the fake
    enters a send-out phase (opp max_hp reads 0, no action menu, outcome 0) for a
    couple of A-presses, then the next foe comes in. The last foe fainting sets
    outcome 1 (won); we fainting sets outcome 2 (lost).
    """

    _RESOLVE_PRESSES = 2
    _SENDOUT_PRESSES = 2

    def __init__(self, my_dmg: int, foe_dmg: int, opp_team=(18, 18)) -> None:
        self._my_dmg = my_dmg
        self._foe_dmg = foe_dmg
        self._opp_team = list(opp_team)
        self._opp_idx = 0
        self._opp_hp = self._opp_team[0]
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0
        self._sendout_left = 0

    def step(self, keys: int, frames: int) -> None:
        if not (keys & buttons.KEY_A):
            return  # only A drives the fake; d-pad nav is ignored
        if self._phase == "menu":
            self._phase = "moves"
        elif self._phase == "moves":
            self._commit_turn()
        elif self._phase == "resolving":
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                if self._opp_hp == 0:  # current foe down, more remain
                    self._phase = "sendout"
                    self._sendout_left = self._SENDOUT_PRESSES
                else:
                    self._phase = "menu"
        elif self._phase == "sendout":
            self._sendout_left -= 1
            if self._sendout_left <= 0:
                self._opp_idx += 1
                self._opp_hp = self._opp_team[self._opp_idx]
                self._phase = "menu"

    def _commit_turn(self) -> None:
        self._opp_hp = max(0, self._opp_hp - self._my_dmg)
        self._my_hp = max(0, self._my_hp - self._foe_dmg)
        if self._my_hp == 0:
            self._outcome = 2
        elif self._opp_hp == 0 and self._opp_idx == len(self._opp_team) - 1:
            self._outcome = 1
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
            return bytes(buf[addr - pbase : addr - pbase + size])
        if obase <= addr < obase + BATTLE_MON_SIZE:
            # Send-out reads opp max_hp 0 -> in_battle False (the transient).
            opp_max = 0 if self._phase == "sendout" else 18
            buf = self._mon(hp=self._opp_hp, max_hp=opp_max)
            return bytes(buf[addr - obase : addr - obase + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def test_trainer_battle_send_out_does_not_falsely_end() -> None:
    # Foe #1 (6hp) faints in one hit -> send-out -> foe #2 (6hp) faints -> won.
    emu = _ScriptedTrainerBattle(my_dmg=6, foe_dmg=0, opp_team=(6, 6))
    assert play_trainer_battle(emu, lambda mid: 12, _predict) == "won"


def test_play_battle_falsely_ends_on_the_same_send_out() -> None:
    # Control: the wild play_battle stops on the transient not-in_battle -> lost.
    emu = _ScriptedTrainerBattle(my_dmg=6, foe_dmg=0, opp_team=(6, 6))
    assert play_battle(emu, lambda mid: 12, _predict) == "lost"


def test_trainer_battle_wins_over_a_two_mon_team() -> None:
    # Each foe (18hp) takes 3 turns at my_dmg=6; we never faint.
    emu = _ScriptedTrainerBattle(my_dmg=6, foe_dmg=0, opp_team=(18, 18))
    assert play_trainer_battle(emu, lambda mid: 12, _predict) == "won"


def test_trainer_battle_loses_when_we_faint() -> None:
    emu = _ScriptedTrainerBattle(my_dmg=1, foe_dmg=19, opp_team=(18, 18))
    assert play_trainer_battle(emu, lambda mid: 12, _predict) == "lost"


def test_trainer_battle_times_out() -> None:
    emu = _ScriptedTrainerBattle(my_dmg=0, foe_dmg=0, opp_team=(18, 18))
    result = play_trainer_battle(emu, lambda mid: 12, _predict, max_turns=4)
    assert result == "battle_timeout"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_battle_player.py -v`
Expected: FAIL on import — `cannot import name 'play_trainer_battle' from 'env.battle_player'`.

- [ ] **Step 3: Write minimal implementation**

Add to `env/battle_player.py` (after `play_battle`, reusing the existing imports and `_result`):

```python
def play_trainer_battle(
    emulator: Any,
    move_type_fn: MoveTypeFn,
    predict: PredictFn,
    max_turns: int = 128,
) -> str:
    """Play an ongoing multi-Pokémon trainer battle to the end.

    Unlike play_battle (wild, single opponent), a trainer sends out the next
    Pokémon when its active one faints; opp max_hp reads 0 for a tick during
    send-out, so this stops ONLY on a terminal outcome, never on not in_battle.

    Returns "won" (outcome bit 0x1 set), "lost" (any other terminal outcome),
    or "battle_timeout" (max_turns reached without a terminal outcome).
    """
    reader = BattleReader(emulator.read_bytes)
    advance_to_menu(emulator, reader, wait_through_faint=True)
    for _ in range(max_turns):
        state = reader.battle_state()
        if state.outcome != 0:
            return _result(state.outcome)
        action = predict(observation(state, move_type_fn))
        select_move(emulator, reader, int(action))
        advance_to_menu(emulator, reader, wait_through_faint=True)
    state = reader.battle_state()
    if state.outcome != 0:
        return _result(state.outcome)
    return "battle_timeout"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_battle_player.py -v`
Expected: PASS (5 new + 3 pre-existing).

- [ ] **Step 5: Commit**

```bash
git add env/battle_player.py tests/test_battle_player.py
git commit -m "feat: play_trainer_battle drives a multi-mon trainer battle to a win"
```

---

### Task 3: Disposable capture tool + gated ROM smoke

**Files:**
- Create: `tools/capture_trainer_battle.py`
- Modify: `tests/test_battle_player_rom.py`

- [ ] **Step 1: Write the capture tool**

Create `tools/capture_trainer_battle.py`. It loads a savestate that sits just before a trainer's line-of-sight, walks a fixed heading to trip the trainer, and saves the moment `in_battle()` flips. The default `--state`/`--heading` are placeholders the operator overrides for whichever trainer is reachable; if none fires, exit 1 (nothing saved), and the ROM smoke stays a documented skip (spec fallback).

```python
"""Walk into a trainer's line of sight until the trainer battle fires, then cache
states/trainer_battle.state.

One-shot scaffolding: run once locally where the ROM + a pre-trainer savestate
exist. Output feeds tests/test_battle_player_rom.py (deterministic trainer-battle
smoke). Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_trainer_battle.py \
      --state states/<pre_trainer>.state --heading up
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from emulator.buttons import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP
from emulator.gba import GbaEmulator
from env.orders import GRIND_RELEASE_FRAMES, GRIND_STEP_FRAMES
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/trainer_battle.state")
_HEADINGS = {"up": KEY_UP, "down": KEY_DOWN, "left": KEY_LEFT, "right": KEY_RIGHT}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--heading", choices=sorted(_HEADINGS), default="up")
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    key = _HEADINGS[args.heading]
    start = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [start], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()

    for i in range(args.max_steps):
        env.emulator.step(key, GRIND_STEP_FRAMES)
        env.emulator.step(0, GRIND_RELEASE_FRAMES)  # release (GBA debounce)
        if reader.in_battle():
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            print(f"IN-BATTLE state saved after {i} steps -> {OUT_PATH.resolve()}", flush=True)
            return

    print(f"no trainer battle in {args.max_steps} steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the gated smoke test**

Add to `tests/test_battle_player_rom.py` (module-level `_ROM`/`_MAIN`/`_CKPT` already exist):

```python
_TRAINER_STATE = _MAIN / "states" / "trainer_battle.state"


@pytest.mark.skipif(not _ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _CKPT.exists(), reason="Fighter checkpoint missing")
@pytest.mark.skipif(not _TRAINER_STATE.exists(), reason="no trainer_battle.state artifact")
def test_fighter_wins_a_real_trainer_battle() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.battle_player import play_trainer_battle
    from env.game_state import BattleReader

    emu = GbaEmulator(_ROM)
    emu.load_state(_TRAINER_STATE.read_bytes())
    emu.step(0, 4)  # let the emulator settle after load_state

    reader = BattleReader(emu.read_bytes)
    assert reader.battle_state().in_battle  # precondition: really mid-battle

    model = PPO.load(str(_CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    result = play_trainer_battle(emu, make_move_type_fn(emu), predict)
    assert result == "won"
    assert not reader.battle_state().in_battle
```

- [ ] **Step 3: Run to verify the smoke is collected and skips cleanly**

Run: `.venv/bin/python -m pytest tests/test_battle_player_rom.py -v`
Expected: the new test is SKIPPED (`no trainer_battle.state artifact`) — no collection error. (It becomes load-bearing once the operator runs the capture tool locally.)

- [ ] **Step 4: Verify the tool imports cleanly (no ROM run)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('tools/capture_trainer_battle.py').read()); print('parse OK')"`
Expected: `parse OK`.

- [ ] **Step 5: Commit**

```bash
git add tools/capture_trainer_battle.py tests/test_battle_player_rom.py
git commit -m "chore: disposable trainer-battle capture tool + gated ROM smoke"
```

---

## Final verification

- [ ] Run the full suite: `POKEMON_EMERALD_ROM="$(cd roms && pwd)/pokemon_emerald_fr.gba" .venv/bin/python -m pytest -q 2>&1 | tail -8`
  Expected: all prior tests pass + the 7 new pure tests pass; the trainer ROM smoke SKIPS (artifact absent). No regressions in `test_battle_env.py`/`test_battle_turn.py` (the wild path is unchanged at the default flag).
- [ ] Ruff clean: `.venv/bin/ruff check env/battle_turn.py env/battle_player.py tools/capture_trainer_battle.py tests/test_battle_turn.py tests/test_battle_player.py tests/test_battle_player_rom.py`

## Self-review notes

- **Spec coverage:** helper flag (Task 1), `play_trainer_battle` end-only-on-outcome + send-out non-false-end + control (Task 2), capture tool + gated smoke (Task 3), known-risk replacement A-press exercised by the smoke. All spec sections map to a task.
- **Type consistency:** `play_trainer_battle` signature mirrors `play_battle` (`emulator, move_type_fn, predict, max_turns`); `advance_to_menu(emulator, reader, wait_through_faint=...)`; `_result`/`observation`/`select_move`/`BattleReader` reused verbatim.
- **No placeholders in code steps.** The capture tool's reachable-trainer path is intentionally an operator-supplied `--state`/`--heading` (disposable tool); if unreachable, the smoke stays a documented skip per the spec fallback.
