"""Gymnasium battle env for the Fighter. One episode = one battle.

Reads battle state from RAM (numbers, not pixels) and exposes a Discrete(4)
'use move i' action. Turn boundaries are driven by BattleReader.at_action_menu()
(a RAM flag that is set only while the game waits for the player to pick an
action), so each env step performs exactly one battle turn: open the move list,
select the slot, then advance dialogue until the next menu or the battle ends.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.battle_rewards import BattleRewardTracker
from env.battle_turn import MoveTypeFn, OBS_SIZE, advance_to_menu, observation, press, select_move
from env.game_state import BattleReader, BattleState

RESET_WARMUP_FRAMES = 4  # let the emulator render after load_state


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
        press(self.emulator, key)

    def _select_move(self, action: int) -> None:
        select_move(self.emulator, self._reader, int(action))

    def _advance_to_menu(self) -> None:
        advance_to_menu(self.emulator, self._reader)

    def _observation(self, state: BattleState) -> np.ndarray:
        return observation(state, self._move_type_fn)

    def _info(self, state: BattleState) -> dict[str, Any]:
        return {
            "turn": self._turn,
            "outcome": state.outcome,
            "won": bool(state.outcome & 0x1),
            "opp_hp": state.opp_hp,
            "my_hp": state.my_hp,
        }
