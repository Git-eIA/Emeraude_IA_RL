"""Gated ROM smoke: run run_pokedex_return from post_rival.state end to end.

The single load-bearing proof of the A2 chain: hop_via_explore hops route_103 -> Oldale,
cross_scripted_npc plays Flora's gate into route_101, reach_map descends to the lab, and
the OnFrame GivePokedex cutscene delivers the Pokédex. Asserts has_pokedex() True + 5 balls
+ lab arrival, dumps states/post_pokedex.state FIRST, then reloads the dump and verifies
control (anti-false-lock pin). Slow (~minute, Fighter-driven). Triple-skips without ROM /
Fighter checkpoint / post_rival.state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"
START_STATE = "states/post_rival.state"

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(FIGHTER_CKPT).exists(), reason="Fighter checkpoint missing"),
    pytest.mark.skipif(not Path(START_STATE).exists(), reason="post_rival.state missing"),
]


def test_run_pokedex_return_delivers_the_pokedex() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.campaign import LAB, run_pokedex_return
    from env.game_state import EmeraldReader
    from env.map_memory import MapMemory
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    with open(START_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)
    memory = MapMemory()

    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs):
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)

    # run_pokedex_return needs snapshot/grid_reader (world) AND has_pokedex (reader);
    # a thin adapter forwards both, matching test_phase2_rom.py.
    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_pokedex_return(
        emu, _Reader(), memory,
        move_type_fn=move_type_fn, predict=predict,
    )

    assert result == "pokedex_delivered", result
    assert reader.has_pokedex() is True
    settled = world.snapshot()
    assert settled is not None and settled.map_id == LAB, settled

    # _finish_lab_cutscene now guarantees full delivery, not just the early flag.
    assert reader.has_poke_balls(5) is True
    assert reader.birch_lab_state() == 5

    # Dump FIRST, then reload the dump and verify control on the RELOADED session:
    # the pin must prove the DUMP is healthy, and verification must not mutate it.
    state_bytes = emu.save_state()
    Path("states/post_pokedex.state").write_bytes(state_bytes)

    from tests.conftest import control_returns

    emu2 = GbaEmulator(ROM)
    emu2.load_state(state_bytes)
    emu2.step(0, 4)
    assert control_returns(emu2), "reloaded post_pokedex.state is control-locked"
