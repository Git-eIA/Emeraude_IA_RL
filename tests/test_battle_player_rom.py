"""Gated ROM smoke: the real Fighter wins a real wild battle via play_battle."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROM = os.environ.get("POKEMON_EMERALD_ROM")
_MAIN = Path("/Users/_eloi/Projets/Emu")
_CKPT = _MAIN / "checkpoints" / "fighter" / "ppo_fighter_final.zip"
_STATES = sorted((_MAIN / "states" / "battles").glob("*.state"))
_TRAINER_STATE = _MAIN / "states" / "trainer_battle.state"


@pytest.mark.skipif(not _ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _CKPT.exists(), reason="Fighter checkpoint missing")
@pytest.mark.skipif(not _STATES, reason="no battle savestates in states/battles/")
def test_fighter_wins_a_real_battle_via_play_battle() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.battle_player import play_battle

    emu = GbaEmulator(_ROM)
    emu.load_state(_STATES[0].read_bytes())
    emu.step(0, 4)  # let the emulator settle after load_state

    model = PPO.load(str(_CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    result = play_battle(emu, make_move_type_fn(emu), predict)
    assert result == "won"


@pytest.mark.skipif(not _ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _CKPT.exists(), reason="Fighter checkpoint missing")
@pytest.mark.skipif(not _TRAINER_STATE.exists(), reason="states/trainer_battle.state missing")
def test_fighter_wins_a_real_trainer_battle() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.battle_player import play_trainer_battle
    from env.game_state import BattleReader

    emu = GbaEmulator(_ROM)
    emu.load_state(_TRAINER_STATE.read_bytes())
    emu.step(0, 4)  # let the emulator settle after load_state

    reader = BattleReader(emu.read_bytes)
    assert reader.battle_state().in_battle, "precondition: artifact must be mid-battle"

    model = PPO.load(str(_CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)
    result = play_trainer_battle(emu, move_type_fn, predict)
    assert result == "won"
    assert not reader.battle_state().in_battle
