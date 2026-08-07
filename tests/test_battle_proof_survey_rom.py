"""Deterministic live smoke: explore_grid survives a real route_101 wild battle.

Double-skips without the ROM or the states/route101_in_battle.state artifact
(produced by tools/capture_route101_in_battle.py). Loading that mid-battle state
means explore_grid's first action always sees an in-progress battle: the real
Fighter must win it and the survey must resume rather than abort.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from env.map_traveler import BATTLE_OUTCOMES

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = Path("states/route101_in_battle.state")
ROUTE_101 = (0, 16)


@pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not STATE.exists(), reason="states/route101_in_battle.state missing")
def test_explore_grid_survives_a_real_route101_battle() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.grid_explorer import explore_grid
    from env.map_memory import MapMemory
    from env.pokemon_env import PokemonEmeraldEnv
    from env.world_reader import WorldReader

    state = STATE.read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(ROM), [state], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()
    assert reader.in_battle()  # precondition: the artifact really is mid-battle

    fighter = PPO.load("checkpoints/fighter/ppo_fighter_final.zip", device="cpu")

    def predict(obs) -> int:
        return int(fighter.predict(obs, deterministic=True)[0])

    # max_steps=1 isolates the loaded mid-battle: explore_grid's first loop
    # iteration hands the in-progress battle to the Fighter, wins it, resumes
    # the survey for one candidate, then stops. Keeping the budget at 1 avoids
    # wandering deeper into route_101's grass where a *later* wild battle could
    # be lost to team attrition (RNG) — that would be a Fighter/team outcome,
    # not a failure of explore_grid to survive the mid-battle start this test
    # exists to prove.
    result = explore_grid(
        env.emulator, reader, MapMemory(), ROUTE_101,
        max_steps=1,
        move_type_fn=make_move_type_fn(env.emulator), predict=predict,
    )

    # The Fighter won the loaded battle and the survey resumed: not a battle
    # outcome, and the battle is actually resolved (assertion above is not
    # vacuous).
    assert result not in BATTLE_OUTCOMES
    assert not reader.in_battle()
