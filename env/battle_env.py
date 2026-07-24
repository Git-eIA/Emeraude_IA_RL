"""Gymnasium battle env for the Fighter. One episode = one battle.

Reads battle state from RAM (numbers, not pixels) and exposes a Discrete(4)
'use move i' action. Turn boundaries are driven by BattleReader.at_action_menu()
(a RAM flag that is set only while the game waits for the player to pick an
action), so each env step performs exactly one battle turn: open the move list,
select the slot, then advance dialogue until the next menu or the battle ends.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emulator import buttons
from env.battle_rewards import BattleRewardTracker
from env.game_state import BattleReader, BattleState

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
RESET_WARMUP_FRAMES = 4  # let the emulator render after load_state

MoveTypeFn = Callable[[int], int]


class BattleEmeraldEnv(gym.Env):
    """Numbers in, move choice out. Episodes start from a battle savestate."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        emulator: Any,
        initial_states: list[bytes],
        move_type_fn: MoveTypeFn,
        max_turns: int = 64,
    ) -> None:
        super().__init__()
        if not initial_states:
            raise ValueError("initial_states must be non-empty")
        self.emulator = emulator
        self._initial_states = initial_states
        self._move_type_fn = move_type_fn
        self._max_turns = max_turns
        self._reader = BattleReader(emulator.read_bytes)
        self._rewards = BattleRewardTracker()
        self._turn = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        idx = int(self.np_random.integers(len(self._initial_states)))
        self.emulator.load_state(self._initial_states[idx])
        self.emulator.step(0, RESET_WARMUP_FRAMES)
        self._turn = 0
        # The savestate may sit mid-intro; advance to the first action menu so
        # the agent's first decision is a real move choice.
        self._advance_to_menu()
        state = self._reader.battle_state()
        self._rewards.reset(state)
        return self._observation(state), self._info(state)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._select_move(int(action))
        self._advance_to_menu()
        self._turn += 1
        state = self._reader.battle_state()
        reward = self._rewards.update(state)
        terminated = state.outcome != 0 or not state.in_battle
        truncated = not terminated and self._turn >= self._max_turns
        return self._observation(state), reward, terminated, truncated, self._info(state)

    def render(self) -> np.ndarray:
        return self.emulator.screenshot()

    def _press(self, key: int) -> None:
        # Hold then release so the GBA registers a distinct press.
        self.emulator.step(key, PRESS_HOLD_FRAMES)
        self.emulator.step(0, PRESS_RELEASE_FRAMES)

    def _select_move(self, action: int) -> None:
        # From the action menu (cursor defaults to FIGHT), press A until the
        # move list is open (flag leaves the menu value), navigate to the chosen
        # slot, then commit. Pressing A until the menu opens absorbs an input
        # eaten by the menu's open animation.
        for _ in range(OPEN_MENU_TRIES):
            if not self._reader.at_action_menu():
                break
            self._press(buttons.KEY_A)
        for _ in range(action):
            self._press(buttons.KEY_DOWN)
        self._press(buttons.KEY_A)

    def _advance_to_menu(self) -> None:
        # Press A to clear dialogue until back at the action menu or battle end.
        for _ in range(MAX_ADVANCE_PRESSES):
            state = self._reader.battle_state()
            if state.outcome != 0 or not state.in_battle:
                return
            if self._reader.at_action_menu():
                self.emulator.step(0, SETTLE_FRAMES)
                return
            self._press(buttons.KEY_A)

    def _observation(self, state: BattleState) -> np.ndarray:
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
            obs[8 + 2 * i] = min(self._move_type_fn(move.move_id) / NUM_TYPES, 1.0)
            obs[9 + 2 * i] = min(move.pp / MAX_PP, 1.0)
        obs[16] = 1.0 if state.in_battle else 0.0
        return obs

    def _info(self, state: BattleState) -> dict[str, Any]:
        return {
            "turn": self._turn,
            "outcome": state.outcome,
            "won": bool(state.outcome & 0x1),
            "opp_hp": state.opp_hp,
            "my_hp": state.my_hp,
        }


def _frac(hp: int, max_hp: int) -> float:
    return hp / max_hp if max_hp > 0 else 0.0
