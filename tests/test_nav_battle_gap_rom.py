"""Load-bearing ROM smoke: the battle-transition fix un-poisons the north nav.

Reproduction (verified on the real ROM from post_starter.state, target (11,0)):
- PRE-FIX  : navigate_grid returns 'unreachable', player stuck at (11,10) —
  wild-battle intro/fade windows get recorded as walls and poison the blocked
  set until plan_path_grid returns None.
- POST-FIX : navigate_grid returns an honest 'timeout' at ~(10,12) — the false
  walls are gone; the player is no longer trapped by battle transitions.

The single discriminating assertion is therefore `result != 'unreachable'`: it
fails pre-fix and passes post-fix. A positional assertion would NOT discriminate
(the pre-fix stuck pos (11,10) has the same y as several post-fix timeout
positions), so it is intentionally omitted. Reaching (11,0) or crossing the
trap band is NOT the target — the Oldale north-exit is a separate, out-of-scope
geometry/attrition gap downstream of this fix.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = Path("states/post_starter.state")
CKPT = Path("checkpoints/fighter/ppo_fighter_final.zip")


@pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not STATE.exists(), reason="post_starter.state missing")
@pytest.mark.skipif(not CKPT.exists(), reason="Fighter checkpoint missing")
def test_north_nav_not_poisoned_by_battle_transitions() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.grid_navigator import navigate_grid
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)

    reader = WorldReader(emu.read_bytes)
    model = PPO.load(str(CKPT), device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])

    result = navigate_grid(
        emu, reader, target=(11, 0), max_steps=400,
        move_type_fn=make_move_type_fn(emu), predict=predict,
    )
    end = reader.snapshot()
    # The fix removes the false-wall poison: pre-fix this exact call returns
    # 'unreachable'; post-fix it does not. That is the whole regression guard.
    assert result != "unreachable", f"still poisoned: {result} at {end and end.pos}"
