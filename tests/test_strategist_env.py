"""StrategistEnv: Gym API compliance + MDP behavior (no ROM, no emulator)."""
from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.strategist_env import CHALLENGE_LEVELS, StrategistEnv


def test_gym_api_compliance() -> None:
    check_env(StrategistEnv(), skip_render_check=True)


def test_observation_shape_and_bounds() -> None:
    env = StrategistEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    assert info == {}


def test_reset_starts_at_first_challenge_full_hp() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    assert env.team_level == 5.0
    assert env.team_hp == 1.0
    assert env.challenge_idx == 0
    assert env.steps == 0


def test_observation_encodes_progression() -> None:
    env = StrategistEnv()
    obs, _ = env.reset(seed=0)
    assert abs(obs[0] - 0.05) < 1e-6
    assert obs[1] == 1.0
    assert abs(obs[2] - 0.05) < 1e-6
    assert abs(obs[3] - 0.5) < 1e-6
    assert obs[4] == 0.0
    assert len(CHALLENGE_LEVELS) == 5
