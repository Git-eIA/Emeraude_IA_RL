"""Gymnasium environment for Pokémon Emerald over a (real or fake) GBA emulator."""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from emulator import buttons
from env.game_state import EmeraldReader, PlayerState
from env.milestones import MilestoneTracker, starter_milestones
from env.rewards import TIME_PENALTY, ExplorationTracker, LevelRewardTracker

FRAME_SIZE = (84, 84)  # (width, height) after downscale
FRAME_STACK = 3
OBS_SHAPE = (FRAME_SIZE[1], FRAME_SIZE[0], FRAME_STACK)
FRAMES_PER_ACTION = 24  # ~0.4 s of game time; one walking step per action

_ACTION_KEYS: dict[str, int] = {
    "noop": 0,
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
    "a": buttons.KEY_A,
    "b": buttons.KEY_B,
    "start": buttons.KEY_START,
}


class PokemonEmeraldEnv(gym.Env):
    """Pixels in, exploration reward out. Episodes start from a random savestate."""

    metadata = {"render_modes": ["rgb_array"]}
    ACTIONS = list(_ACTION_KEYS)

    def __init__(
        self,
        emulator: Any,
        initial_states: list[bytes],
        max_steps: int = 2048,
    ) -> None:
        super().__init__()
        if not initial_states:
            raise ValueError("initial_states must be non-empty")
        self.emulator = emulator
        self._initial_states = initial_states
        self._max_steps = max_steps
        self._reader = EmeraldReader(emulator.read_bytes)
        self._tracker = ExplorationTracker()
        self._milestones = MilestoneTracker(starter_milestones())
        self._levels = LevelRewardTracker()
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._step_count = 0
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        idx = int(self.np_random.integers(len(self._initial_states)))
        self.emulator.load_state(self._initial_states[idx])
        self._tracker.reset()
        self._milestones.reset()
        self._levels.reset()
        self._frames.clear()
        self._step_count = 0
        frame = self._current_frame()
        for _ in range(FRAME_STACK):
            self._frames.append(frame)
        return self._observation(), self._info(self._reader.player_state())

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
        reward += TIME_PENALTY
        truncated = not terminated and self._step_count >= self._max_steps
        return self._observation(), reward, terminated, truncated, self._info(state)

    def render(self) -> np.ndarray:
        return self.emulator.screenshot()

    def _current_frame(self) -> np.ndarray:
        image = Image.fromarray(self.emulator.screenshot()).convert("L").resize(FRAME_SIZE)
        return np.asarray(image, dtype=np.uint8)

    def _observation(self) -> np.ndarray:
        return np.stack(self._frames, axis=-1)

    def _info(self, state: PlayerState | None) -> dict[str, Any]:
        return {
            "visited_tiles": self._tracker.visited_count,
            "badges": state.badges if state else 0,
            "map": (state.map_group, state.map_num) if state else None,
            "pos": (state.x, state.y) if state else None,
            "step": self._step_count,
            "milestones": sorted(self._milestones.fired),
        }
