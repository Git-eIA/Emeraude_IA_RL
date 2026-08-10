"""Gated ROM smoke: run PHASE2_CAMPAIGN from post_rival.state end to end.

Load-bearing once tools/probe_phase2_facts.py has confirmed the constants: seeds
the return portals, drives the scripted campaign (return -> lab -> Pokedex ->
Balls -> shoes) with the real Fighter, and asserts the deliverables landed. Dumps
states/post_phase2.state for the next phase.
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


def test_phase2_campaign_delivers_pokedex_and_running_shoes() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.campaign import PHASE2_CAMPAIGN, seed_return_portals, run_campaign
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
    seed_return_portals(memory)

    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs):
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)

    # The campaign reads world/game state off `emu`; run_campaign expects a reader
    # that also exposes snapshot/party_* — a thin adapter forwards both readers.
    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_campaign(
        emu, _Reader(), memory,
        curriculum=PHASE2_CAMPAIGN,
        move_type_fn=move_type_fn, predict=predict,
    )

    assert result == "campaign_complete", result
    assert reader.has_pokedex()
    assert reader.has_running_shoes()

    Path("states/post_phase2.state").write_bytes(emu.save_state())
