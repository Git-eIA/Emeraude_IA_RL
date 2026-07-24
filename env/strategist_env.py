"""StrategistEnv: abstract MDP for the continue/grind/heal decision.

No pixels, no map, no emulator — the Strategist's world is the progression
state (team level, team HP, which important battle is next). Battle outcomes
resolve through the calibrated win-probability model in strategist_model.py.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.strategist_model import ADVANCE_HP_COST, grind, heal, win_prob

# Rising-difficulty curriculum of important battles (target levels). Synthetic
# on purpose — v1 teaches the decision, not the real map.
CHALLENGE_LEVELS = (5, 8, 12, 16, 20)

MAX_STEPS = 50   # decision budget per episode (guards infinite grind/heal loops)

# Actions.
ADVANCE = 0   # attempt the current important battle
GRIND = 1     # one wild-battle session: gain level, lose HP, cost time
HEAL = 2      # restore HP to full, cost time (Pokemon Center detour)

# Reward.
WIN_REWARD = 20.0    # per important battle won
LOSS_REWARD = -20.0  # on a lost important battle (episode ends)
STEP_GRIND = -1.0    # time cost of a grind session
STEP_HEAL = -2.0     # time cost of a heal detour (round trip, so bigger)

START_LEVEL = 5.0
START_HP = 1.0


class StrategistEnv(gym.Env):
    """Meta-game MDP: choose advance / grind / heal to clear the curriculum."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.team_level = START_LEVEL
        self.team_hp = START_HP
        self.challenge_idx = 0
        self.steps = 0

    def _challenge_level(self) -> int:
        # Clamp: after the last win challenge_idx == len, but _obs still runs.
        idx = min(self.challenge_idx, len(CHALLENGE_LEVELS) - 1)
        return CHALLENGE_LEVELS[idx]

    def _obs(self) -> np.ndarray:
        cl = self._challenge_level()
        gap = float(np.clip((self.team_level - cl) / 40.0 + 0.5, 0.0, 1.0))
        return np.array(
            [
                np.clip(self.team_level / 100.0, 0.0, 1.0),
                np.clip(self.team_hp, 0.0, 1.0),
                np.clip(cl / 100.0, 0.0, 1.0),
                gap,
                min(self.challenge_idx / len(CHALLENGE_LEVELS), 1.0),
            ],
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.team_level = START_LEVEL
        self.team_hp = START_HP
        self.challenge_idx = 0
        self.steps = 0
        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self.steps += 1
        terminated = False

        if action == GRIND:
            self.team_level, self.team_hp = grind(self.team_level, self.team_hp)
            reward = STEP_GRIND
        elif action == HEAL:
            self.team_hp = heal()
            reward = STEP_HEAL
        else:  # ADVANCE
            p = win_prob(self.team_level, self._challenge_level(), self.team_hp)
            if self.np_random.random() < p:
                reward = WIN_REWARD
                self.team_hp = max(0.0, self.team_hp - ADVANCE_HP_COST)
                self.challenge_idx += 1
                if self.challenge_idx >= len(CHALLENGE_LEVELS):
                    terminated = True   # cleared the whole curriculum
            else:
                reward = LOSS_REWARD
                terminated = True       # lost an important battle

        truncated = self.steps >= MAX_STEPS
        return self._obs(), reward, terminated, truncated, {}

