"""Gated ROM smoke: the env dispatches the Fighter on a real trainer battle and
(when the captured state is on route_103) latches _rival_beaten + beat_rival
milestone end-to-end.

Triple-skip: POKEMON_EMERALD_ROM unset | Fighter checkpoint missing |
states/trainer_battle.state missing (the route_103 rival capture is deferred).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from env.game_state import BattleReader, EmeraldReader
from env.milestones import ROUTE_103, route103_milestones
from env.pokemon_env import PokemonEmeraldEnv

_ROM = os.environ.get("POKEMON_EMERALD_ROM")
_MAIN = Path("/Users/_eloi/Projets/Emu")
_CKPT = _MAIN / "checkpoints" / "fighter" / "ppo_fighter_final.zip"
_TRAINER_STATE = _MAIN / "states" / "trainer_battle.state"


@pytest.mark.skipif(not _ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _CKPT.exists(), reason="Fighter checkpoint missing")
@pytest.mark.skipif(not _TRAINER_STATE.exists(), reason="states/trainer_battle.state missing")
def test_env_dispatches_fighter_on_real_trainer_battle() -> None:
    """Env resolves a mid-trainer-battle state via play_trainer_battle.

    Load-bearing half: after one env.step(), in_battle is False (the Fighter
    played the battle to completion).

    Conditional-on-route_103 half: when the captured state is on route_103,
    the _rival_beaten latch is set and the beat_rival milestone fires.
    The route_103 capture is currently deferred, so this branch is exercised
    only once that artifact is produced.
    """
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator

    state = _TRAINER_STATE.read_bytes()
    emu = GbaEmulator(_ROM)

    model = PPO.load(str(_CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)

    env = PokemonEmeraldEnv(
        emu,
        initial_states=[state],
        max_steps=64,
        milestones=route103_milestones(),
        move_type_fn=move_type_fn,
        predict=predict,
    )
    env.reset()
    emu.step(0, 4)  # let the emulator settle after load_state (mirrors test_battle_player_rom.py)

    reader = BattleReader(emu.read_bytes)
    assert reader.battle_state().in_battle, "precondition: artifact must be mid-battle"
    assert reader.is_trainer_battle(), "precondition: must be a TRAINER battle"

    world = EmeraldReader(emu.read_bytes)
    pre = world.player_state()
    on_route103 = pre is not None and (pre.map_group, pre.map_num) == ROUTE_103

    env.step(0)  # noop — battle hook fires, play_trainer_battle resolves it

    assert not reader.battle_state().in_battle, "battle must be over after env.step()"

    if on_route103:
        assert env._rival_beaten is True, "_rival_beaten latch must be set"
        assert "beat_rival" in env._milestones.fired, "beat_rival milestone must have fired"
