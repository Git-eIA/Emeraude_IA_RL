"""Pure tests for play_battle: drive a scripted battle to an outcome (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.battle_player import play_battle
from env.game_state import BATTLE_MON_SIZE


def _u16(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class _ScriptedBattle:
    """Menu->moves->resolving loop. Each committed turn deals `my_dmg` to the foe
    and `foe_dmg` to us. The foe faints -> outcome 1 (won); we faint -> outcome 2
    (lost). Mirrors the real ROM's flag-driven turn loop.
    """

    _RESOLVE_PRESSES = 2

    def __init__(self, my_dmg: int, foe_dmg: int) -> None:
        self._my_dmg = my_dmg
        self._foe_dmg = foe_dmg
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            self._commit_turn()
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def _commit_turn(self) -> None:
        self._opp_hp = max(0, self._opp_hp - self._my_dmg)
        self._my_hp = max(0, self._my_hp - self._foe_dmg)
        if self._opp_hp == 0:
            self._outcome = 1
        elif self._my_hp == 0:
            self._outcome = 2
        self._phase = "resolving"
        self._resolve_left = self._RESOLVE_PRESSES

    def _mon(self, *, hp: int, max_hp: int) -> bytearray:
        buf = bytearray(BATTLE_MON_SIZE)
        buf[0x00:0x02] = _u16(1)
        for i in range(4):
            buf[0x0C + 2 * i : 0x0C + 2 * i + 2] = _u16(1 if i == 0 else 0)
            buf[0x24 + i] = 10 if i == 0 else 0
        buf[0x21], buf[0x22] = 12, 12
        buf[0x28:0x2A] = _u16(hp)
        buf[0x2A] = 5
        buf[0x2C:0x2E] = _u16(max_hp)
        return buf

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            ACTION_MENU_VALUE,
            GBATTLE_ACTION_MENU_ADDR,
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_ACTION_MENU_ADDR:
            return bytes([ACTION_MENU_VALUE if self._phase == "menu" else 0])
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        if pbase <= addr < pbase + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._my_hp, max_hp=19)
            off = addr - pbase
            return bytes(buf[off : off + size])
        if obase <= addr < obase + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._opp_hp, max_hp=18)
            off = addr - obase
            return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def _predict(_obs) -> int:
    return 0  # always use move slot 0


def test_play_battle_wins() -> None:
    emu = _ScriptedBattle(my_dmg=6, foe_dmg=2)  # foe (18hp) faints in 3 turns
    assert play_battle(emu, move_type_fn=lambda mid: 12, predict=_predict) == "won"


def test_play_battle_loses() -> None:
    emu = _ScriptedBattle(my_dmg=1, foe_dmg=19)  # we (19hp) faint first
    assert play_battle(emu, move_type_fn=lambda mid: 12, predict=_predict) == "lost"


def test_play_battle_times_out() -> None:
    emu = _ScriptedBattle(my_dmg=0, foe_dmg=0)  # nobody faints
    result = play_battle(
        emu, move_type_fn=lambda mid: 12, predict=_predict, max_turns=4
    )
    assert result == "battle_timeout"
