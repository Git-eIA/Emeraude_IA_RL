from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.pokemon_env import OBS_SHAPE, PokemonEmeraldEnv
from env.rewards import REVISIT_PENALTY, TIME_PENALTY
from tests.conftest import FakeEmulator


def make_env(max_steps: int = 50) -> PokemonEmeraldEnv:
    return PokemonEmeraldEnv(FakeEmulator(), initial_states=[b"fake"], max_steps=max_steps)


def test_gymnasium_api_compliance():
    check_env(make_env(), skip_render_check=True)


def test_reset_loads_initial_state_and_returns_obs():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.emulator.loaded_states == [b"fake"]
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.uint8


def test_moving_to_new_tile_pays_only_time_penalty():
    # Palier 1: no per-tile bonus. A fresh tile pays just TIME_PENALTY, but still
    # beats a revisit (which also eats REVISIT_PENALTY).
    env = make_env()
    env.reset(seed=0)
    right = env.ACTIONS.index("right")
    _, reward, _, _, _ = env.step(right)
    assert reward == TIME_PENALTY
    assert reward > REVISIT_PENALTY + TIME_PENALTY


def test_staying_put_pays_revisit_and_time_penalty():
    env = make_env()
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    env.step(noop)
    _, reward, _, _, _ = env.step(noop)
    assert reward == REVISIT_PENALTY + TIME_PENALTY


def test_time_penalty_is_small_and_negative():
    # Must stay well under the smallest milestone (+5) over a whole episode.
    assert -0.1 < TIME_PENALTY < 0.0


def test_truncates_at_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(noop)
    assert truncated


def test_starter_terminates_episode_with_jackpot():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_states=[b"state"], max_steps=50)
    env.reset()
    emu.party_count = 1
    emu.party_levels = [5]
    _, reward, terminated, truncated, info = env.step(0)
    # starter +100, level sum 0->5 gives +25; Palier 1 tile bonus is 0, so the
    # only shaving is TIME_PENALTY.
    assert reward >= 125.0 + TIME_PENALTY
    assert terminated is True
    assert truncated is False
    assert "starter_obtained" in info["milestones"]


def test_route_101_milestone_pays_without_terminating():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_states=[b"state"], max_steps=50)
    env.reset()
    emu.map_group, emu.map_num = 0, 16
    _, reward, terminated, _, info = env.step(0)
    assert reward >= 20.0 + TIME_PENALTY
    assert terminated is False
    assert "reach_route_101" in info["milestones"]


def test_reset_clears_milestones():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_states=[b"state"], max_steps=50)
    env.reset()
    emu.party_count = 1
    env.step(0)
    _, info = env.reset()
    assert info["milestones"] == []


def test_intro_chain_pays_each_milestone_once():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_states=[b"state"], max_steps=50)
    env.reset()

    emu.map_group, emu.map_num = 0, 9  # exit the truck into Littleroot
    _, reward, _, _, info = env.step(0)
    assert reward >= 5.0 + TIME_PENALTY
    assert "exit_truck" in info["milestones"]

    emu.map_group, emu.map_num = 1, 0  # enter the player's house (Brendan's)
    _, reward, _, _, info = env.step(0)
    assert reward >= 5.0 + TIME_PENALTY
    assert "enter_house" in info["milestones"]

    emu.clock_set = True  # set the bedroom wall clock
    _, reward, _, _, info = env.step(0)
    assert reward >= 14.9  # milestone +15, but revisit penalty -0.01
    assert "clock_set" in info["milestones"]

    emu.map_group, emu.map_num = 0, 9  # back outside with the clock set
    _, reward, terminated, _, info = env.step(0)
    assert reward >= 9.9  # milestone +10, but revisit penalty -0.01
    assert "back_outside" in info["milestones"]
    assert terminated is False

    emu.map_group, emu.map_num = 1, 2  # into the rival's house (May's), clock now set
    _, reward, _, _, info = env.step(0)
    assert reward >= 4.9  # milestone +5, but revisit penalty -0.01
    assert "enter_rival_house" in info["milestones"]

    emu.map_group, emu.map_num = 1, 3  # upstairs to the rival's bedroom
    _, reward, _, _, info = env.step(0)
    assert reward >= 5.0 + TIME_PENALTY
    assert "rival_upstairs" in info["milestones"]

    emu.town_state = 1  # Pokeball cutscene watched: exit unlocked
    _, reward, _, _, info = env.step(0)
    assert reward >= 14.9  # milestone +15, but revisit penalty -0.01
    assert "meet_rival" in info["milestones"]

    emu.map_group, emu.map_num = 0, 9
    emu.y = 1  # walk up to the northern exit
    _, reward, _, _, info = env.step(0)
    assert reward >= 10.0 + TIME_PENALTY
    assert "north_littleroot" in info["milestones"]


def test_reset_draws_one_of_the_initial_states():
    emu = FakeEmulator()
    env = PokemonEmeraldEnv(emu, initial_states=[b"a", b"b", b"c"], max_steps=50)
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        seen.add(emu.loaded_states[-1])
    # Over 20 seeds the uniform draw should hit more than one state.
    assert seen.issubset({b"a", b"b", b"c"})
    assert len(seen) > 1


def test_empty_initial_states_raises():
    import pytest

    with pytest.raises(ValueError):
        PokemonEmeraldEnv(FakeEmulator(), initial_states=[], max_steps=50)


def test_info_exposes_pos_and_step():
    env = PokemonEmeraldEnv(FakeEmulator(), initial_states=[b"x"], max_steps=10)
    _, info = env.reset()
    assert info["pos"] == (5, 5)  # FakeEmulator starts at (5, 5)
    assert info["step"] == 0

    # action index for "right" moves x from 5 -> 6
    right = PokemonEmeraldEnv.ACTIONS.index("right")
    _, _, _, _, info = env.step(right)
    assert info["pos"] == (6, 5)
    assert info["step"] == 1
