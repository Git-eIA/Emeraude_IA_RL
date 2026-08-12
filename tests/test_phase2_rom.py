"""Gated ROM smoke: run PHASE2_CAMPAIGN (B2) from post_starter.state end to end.

reach_map greedy-descends route_101 -> Littleroot -> lab (portals discovered live, no
seed), then the story A-spam delivers the Pokédex + 5 Poké Balls. Asserts the
deliverables landed and dumps states/post_phase2.state. Triple-skips without ROM /
Fighter checkpoint / post_starter.state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"
START_STATE = "states/post_starter.state"

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(FIGHTER_CKPT).exists(), reason="Fighter checkpoint missing"),
    pytest.mark.skipif(not Path(START_STATE).exists(), reason="post_starter.state missing"),
]


def test_phase2_campaign_delivers_pokedex_and_balls() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.campaign import PHASE2_CAMPAIGN, run_campaign
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
    assert reader.has_item(0x4, 5)

    Path("states/post_phase2.state").write_bytes(emu.save_state())
