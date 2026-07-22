from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.pokemon_env import OBS_SHAPE, PokemonEmeraldEnv
from tests.conftest import FakeEmulator


def make_env(max_steps: int = 50) -> PokemonEmeraldEnv:
    return PokemonEmeraldEnv(FakeEmulator(), initial_state=b"fake", max_steps=max_steps)


def test_gymnasium_api_compliance():
    check_env(make_env(), skip_render_check=True)


def test_reset_loads_initial_state_and_returns_obs():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.emulator.loaded_states == [b"fake"]
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.uint8


def test_moving_to_new_tile_gives_positive_reward():
    env = make_env()
    env.reset(seed=0)
    right = env.ACTIONS.index("right")
    _, reward, _, _, _ = env.step(right)
    assert reward > 0.0


def test_staying_put_gives_zero_reward_after_first_visit():
    env = make_env()
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    env.step(noop)
    _, reward, _, _, _ = env.step(noop)
    assert reward == 0.0


def test_truncates_at_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(noop)
    assert truncated
