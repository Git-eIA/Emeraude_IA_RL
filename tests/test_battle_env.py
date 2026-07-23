"""Battle env: Gym API compliance + one simulated turn (no ROM)."""
from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.battle_env import BattleEmeraldEnv
from env.game_state import BATTLE_MON_SIZE


def _u16(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class _ScriptedBattleEmulator:
    """Fake emulator: opponent HP drops each time the agent acts (a 'hit')."""

    def __init__(self) -> None:
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0

    def load_state(self, data: bytes) -> None:
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0

    def step(self, keys: int, frames: int) -> None:
        # Each action deals 6 damage; three hits faint the opponent and win.
        if self._opp_hp > 0:
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1

    def screenshot(self):
        return np.zeros((160, 240, 3), dtype=np.uint8)

    def _mon(self, *, hp: int, max_hp: int, level: int, types, species: int) -> bytearray:
        buf = bytearray(BATTLE_MON_SIZE)
        buf[0x00:0x02] = _u16(species)
        for i in range(4):
            buf[0x0C + 2 * i : 0x0C + 2 * i + 2] = _u16(1 if i == 0 else 0)
            buf[0x24 + i] = 10 if i == 0 else 0
        buf[0x21], buf[0x22] = types
        buf[0x28:0x2A] = _u16(hp)
        buf[0x2A] = level
        buf[0x2C:0x2E] = _u16(max_hp)
        return buf

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16(0)
        # BattleReader reads fields one at a time, so serve any sub-offset
        # within each mon's 0x58-byte struct, not just the base address.
        player_base = GBATTLE_MONS_ADDR
        opp_base = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        if player_base <= addr < player_base + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._my_hp, max_hp=19, level=5, types=(12, 12), species=1)
            off = addr - player_base
            return bytes(buf[off : off + size])
        if opp_base <= addr < opp_base + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._opp_hp, max_hp=18, level=5, types=(10, 10), species=4)
            off = addr - opp_base
            return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def _make_env() -> BattleEmeraldEnv:
    return BattleEmeraldEnv(
        _ScriptedBattleEmulator(),
        initial_states=[b"battle-savestate"],
        move_type_fn=lambda move_id: 12,
        max_turns=32,
    )


def test_gym_api_compliance() -> None:
    check_env(_make_env(), skip_render_check=True)


def test_observation_shape_and_bounds() -> None:
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (17,)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0


def test_dealing_damage_gives_positive_reward() -> None:
    env = _make_env()
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    assert reward > 0  # opponent lost HP this turn


def test_winning_terminates_and_pays_win_bonus() -> None:
    env = _make_env()
    env.reset(seed=0)
    total = 0.0
    terminated = False
    for _ in range(8):
        _, reward, terminated, truncated, _ = env.step(0)
        total += reward
        if terminated or truncated:
            break
    assert terminated is True
    assert total > 100  # WIN bonus dominates
