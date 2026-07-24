"""StrategistEnv: Gym API compliance + MDP behavior (no ROM, no emulator)."""
from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.strategist_env import (
    ADVANCE,
    CHALLENGE_LEVELS,
    GRIND,
    HEAL,
    MAX_STEPS,
    StrategistEnv,
)


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


def test_grind_levels_up_costs_hp_and_time() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    _, reward, term, trunc, _ = env.step(GRIND)
    assert env.team_level == 6.0
    assert abs(env.team_hp - 0.70) < 1e-9
    assert reward == -1.0
    assert term is False and trunc is False


def test_heal_restores_hp_and_costs_time() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    env.team_hp = 0.25
    _, reward, term, trunc, _ = env.step(HEAL)
    assert env.team_hp == 1.0
    assert reward == -2.0
    assert term is False and trunc is False


def test_advance_win_pays_bonus_costs_hp_and_advances() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    env.team_level = 200.0   # overwhelming -> win_prob ~= 1 (1 - 1e-62), forced win
    hp_before = env.team_hp
    _, reward, term, trunc, _ = env.step(ADVANCE)
    assert reward == 20.0
    assert env.challenge_idx == 1
    assert abs(env.team_hp - (hp_before - 0.30)) < 1e-9
    assert term is False and trunc is False


def test_advance_loss_ends_the_episode() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    env.team_level = -200.0  # hopeless -> win_prob ~= 0 (1e-62), forced loss
    _, reward, term, trunc, _ = env.step(ADVANCE)
    assert reward == -20.0
    assert term is True


def test_clearing_all_five_challenges_succeeds() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    total = 0.0
    terminated = False
    for _ in range(5):
        env.team_level = 200.0   # force a win each advance
        _, reward, terminated, _, _ = env.step(ADVANCE)
        total += reward
        if terminated:
            break
    assert terminated is True
    assert env.challenge_idx == 5
    assert total == 100.0   # 5 wins * WIN_REWARD


def test_truncates_at_step_budget() -> None:
    env = StrategistEnv()
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(HEAL)  # never ends, only costs time
        steps += 1
    assert truncated is True
    assert terminated is False
    assert steps == MAX_STEPS
