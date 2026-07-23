"""Gymnasium battle env for the Fighter. One episode = one battle.

Reads battle state from RAM (numbers, not pixels) and exposes a Discrete(4)
'use move i' action. Between decision points it spams A to advance dialogue
until the player must choose again or the battle ends.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from emulator import buttons
from env.battle_rewards import BattleRewardTracker
from env.game_state import BattleReader, BattleState

FRAMES_PER_PRESS = 8
MAX_ADVANCE_PRESSES = 60  # cap the A-spam loop so a stuck menu can't hang
NUM_TYPES = 18  # types 0..17
MAX_PP = 40.0
OBS_SIZE = 17

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
        self._turn = 0
        state = self._reader.battle_state()
        self._rewards.reset(state)
        return self._observation(state), self._info(state)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._select_move(int(action))
        self._advance_to_decision()
        self._turn += 1
        state = self._reader.battle_state()
        reward = self._rewards.update(state)
        terminated = state.outcome != 0 or not state.in_battle
        truncated = not terminated and self._turn >= self._max_turns
        return self._observation(state), reward, terminated, truncated, self._info(state)

    def render(self) -> np.ndarray:
        return self.emulator.screenshot()

    def _select_move(self, action: int) -> None:
        # Open FIGHT then pick the move slot. The scripted navigation presses A
        # to open the move list and A again on the chosen slot; the fake
        # emulator ignores navigation and just applies the hit.
        self.emulator.step(buttons.KEY_A, FRAMES_PER_PRESS)
        for _ in range(action):
            self.emulator.step(buttons.KEY_DOWN, FRAMES_PER_PRESS)
        self.emulator.step(buttons.KEY_A, FRAMES_PER_PRESS)

    def _advance_to_decision(self) -> None:
        # Spam A to clear dialogue until back at a decision point or battle end.
        for _ in range(MAX_ADVANCE_PRESSES):
            state = self._reader.battle_state()
            if state.outcome != 0 or not state.in_battle:
                return
            self.emulator.step(buttons.KEY_A, FRAMES_PER_PRESS)

    def _observation(self, state: BattleState) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0] = _frac(state.my_hp, state.my_max_hp)
        obs[1] = min(state.my_level / 100.0, 1.0)
        obs[2] = _frac(state.opp_hp, state.opp_max_hp)
        obs[3] = min(state.opp_level / 100.0, 1.0)
        obs[4] = state.my_types[0] / NUM_TYPES
        obs[5] = state.my_types[1] / NUM_TYPES
        obs[6] = state.opp_types[0] / NUM_TYPES
        obs[7] = state.opp_types[1] / NUM_TYPES
        for i, move in enumerate(state.my_moves):
            obs[8 + 2 * i] = self._move_type_fn(move.move_id) / NUM_TYPES
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
