"""Pure tests for play_battle / play_trainer_battle: scripted outcomes (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.battle_player import play_battle, play_trainer_battle
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


class _ScriptedTrainerBattle:
    """Multi-mon trainer battle fake.

    Opponent team is a list of HP values. When the active mon faints there is a
    send-out window of `_SENDOUT_PRESSES` A-presses during which opp max_hp == 0
    (so in_battle reads False) and at_action_menu() is False, while outcome stays
    0. After send-out the next mon becomes active and the machine returns to the
    menu phase. When the LAST mon faints, outcome is set to 1 (won). If our HP
    reaches 0, outcome is set to 2 (lost).
    """

    _RESOLVE_PRESSES = 2
    _SENDOUT_PRESSES = 2

    def __init__(
        self,
        opp_team: list[int],
        my_hp: int,
        my_dmg: int,
        foe_dmg: int,
    ) -> None:
        self._opp_team = list(opp_team)  # mutable copy; current HP per mon
        self._opp_max_hp = list(opp_team)  # immutable snapshot; initial HP per mon
        self._active_idx = 0
        self._my_hp = my_hp
        self._my_dmg = my_dmg
        self._foe_dmg = foe_dmg
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0
        self._sendout_left = 0

    # -- emulator interface --------------------------------------------------

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
                # Check if the active mon just fainted and there is a next one.
                if self._opp_team[self._active_idx] == 0 and self._active_idx + 1 < len(
                    self._opp_team
                ):
                    self._phase = "sendout"
                    self._sendout_left = self._SENDOUT_PRESSES
                else:
                    self._phase = "menu"
        elif self._phase == "sendout" and keys & buttons.KEY_A:
            self._sendout_left -= 1
            if self._sendout_left <= 0:
                # Advance to the next mon and resume battle.
                self._active_idx += 1
                self._phase = "menu"

    def _commit_turn(self) -> None:
        cur = self._opp_team[self._active_idx]
        self._opp_team[self._active_idx] = max(0, cur - self._my_dmg)
        self._my_hp = max(0, self._my_hp - self._foe_dmg)
        if self._opp_team[self._active_idx] == 0:
            # Fainted — outcome is set only if this was the last mon.
            if self._active_idx + 1 >= len(self._opp_team):
                self._outcome = 1
        elif self._my_hp == 0:
            self._outcome = 2
        self._phase = "resolving"
        self._resolve_left = self._RESOLVE_PRESSES

    # -- reader interface ----------------------------------------------------

    def _active_opp_hp(self) -> int:
        return self._opp_team[self._active_idx]

    def _active_opp_max_hp(self) -> int:
        # During send-out the active slot reports max_hp == 0, making in_battle False.
        # Otherwise return the ACTIVE mon's initial HP (not mon[0]'s, which may be 0).
        return 0 if self._phase == "sendout" else self._opp_max_hp[self._active_idx]

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
            # flags != 0 while the trainer battle is running (even during send-out).
            return _u16(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        if pbase <= addr < pbase + BATTLE_MON_SIZE:
            buf = self._mon(hp=self._my_hp, max_hp=30)
            off = addr - pbase
            return bytes(buf[off : off + size])
        if obase <= addr < obase + BATTLE_MON_SIZE:
            buf = self._mon(
                hp=self._active_opp_hp(),
                max_hp=self._active_opp_max_hp(),
            )
            off = addr - obase
            return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


# ---------------------------------------------------------------------------
# Tests for play_trainer_battle
# ---------------------------------------------------------------------------

_move_type_fn = lambda mid: 0  # noqa: E731


def test_trainer_battle_send_out_does_not_falsely_end() -> None:
    """play_trainer_battle must survive the send-out window (opp max_hp == 0)."""
    emu = _ScriptedTrainerBattle(opp_team=[6, 6], my_hp=30, my_dmg=6, foe_dmg=0)
    assert play_trainer_battle(emu, move_type_fn=_move_type_fn, predict=_predict) == "won"


def test_play_battle_falsely_ends_on_the_same_send_out() -> None:
    # Control: wild play_battle uses default wait_through_faint=False, so advance_to_menu
    # returns on the transient not-in_battle during send-out. The loop then reads
    # not in_battle with outcome==0 and calls _result(0) == "lost" — the exact
    # regression that play_trainer_battle fixes.
    emu = _ScriptedTrainerBattle(opp_team=[6, 6], my_hp=30, my_dmg=6, foe_dmg=0)
    assert play_battle(emu, move_type_fn=_move_type_fn, predict=_predict) == "lost"


def test_trainer_battle_wins_over_a_two_mon_team() -> None:
    """Beat mon#1 (send-out), beat mon#2 → won. Uses different HP to distinguish."""
    emu = _ScriptedTrainerBattle(opp_team=[6, 12], my_hp=30, my_dmg=6, foe_dmg=0)
    assert play_trainer_battle(emu, move_type_fn=_move_type_fn, predict=_predict) == "won"


def test_trainer_battle_loses_when_we_faint() -> None:
    """Foe deals 19 damage/turn; we faint before beating mon#1."""
    emu = _ScriptedTrainerBattle(opp_team=[6, 6], my_hp=30, my_dmg=0, foe_dmg=19)
    assert play_trainer_battle(emu, move_type_fn=_move_type_fn, predict=_predict) == "lost"


def test_trainer_battle_times_out() -> None:
    """With my_dmg=0 the foe never faints; max_turns exhausted → battle_timeout."""
    emu = _ScriptedTrainerBattle(opp_team=[6, 6], my_hp=30, my_dmg=0, foe_dmg=0)
    result = play_trainer_battle(
        emu, move_type_fn=_move_type_fn, predict=_predict, max_turns=4
    )
    assert result == "battle_timeout"
