"""Pure tests for the shared battle-turn choreography (no ROM)."""
from __future__ import annotations

import numpy as np

from env.battle_turn import OBS_SIZE, advance_to_menu, observation, select_move
from env.game_state import BattleState, MoveInfo


def _bs(in_battle: bool, outcome: int) -> BattleState:
    moves = (MoveInfo(1, 10), MoveInfo(0, 0), MoveInfo(0, 0), MoveInfo(0, 0))
    return BattleState(
        in_battle=in_battle,
        my_hp=10, my_max_hp=20, my_level=5, my_types=(12, 12), my_moves=moves,
        opp_hp=9, opp_max_hp=18, opp_level=5, opp_types=(10, 10), opp_species=4,
        outcome=outcome, last_move_super_effective=False,
    )


class _FaintEmulator:
    """Models a send-out: in_battle stays False (opp max_hp 0) for two ticks,
    then the live action menu appears. Serves as both emulator and reader.
    """

    def __init__(self) -> None:
        self.presses = 0
        self._ticks = 0

    def step(self, keys: int, frames: int) -> None:
        if keys != 0:
            self.presses += 1
            self._ticks += 1

    def at_action_menu(self) -> bool:
        return self._ticks >= 2

    def battle_state(self) -> BattleState:
        return _bs(in_battle=self._ticks >= 2, outcome=0)


def test_advance_to_menu_default_stops_on_not_in_battle() -> None:
    emu = _FaintEmulator()
    advance_to_menu(emu, emu)  # default wait_through_faint=False
    assert emu.presses == 0  # returns immediately on not in_battle (wild behaviour)


def test_advance_to_menu_waits_through_faint() -> None:
    emu = _FaintEmulator()
    advance_to_menu(emu, emu, wait_through_faint=True)
    assert emu.presses == 2  # presses A through send-out until the live menu


def _state() -> BattleState:
    moves = (MoveInfo(1, 10), MoveInfo(0, 0), MoveInfo(0, 0), MoveInfo(0, 0))
    return BattleState(
        in_battle=True,
        my_hp=10, my_max_hp=20, my_level=5, my_types=(12, 12), my_moves=moves,
        opp_hp=9, opp_max_hp=18, opp_level=5, opp_types=(10, 10), opp_species=4,
        outcome=0, last_move_super_effective=False,
    )


def test_observation_shape_and_bounds() -> None:
    obs = observation(_state(), move_type_fn=lambda mid: 12)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    assert obs[0] == 0.5  # my_hp / my_max_hp = 10/20


class _MenuEmulator:
    """Reader+emulator: menu flag flips off after the first A, counts presses."""

    def __init__(self) -> None:
        self.presses: list[int] = []
        self._at_menu = True

    def step(self, keys: int, frames: int) -> None:
        if keys != 0:
            self.presses.append(keys)
            self._at_menu = False  # first real press opens the move list

    def at_action_menu(self) -> bool:
        return self._at_menu


def test_select_move_opens_list_then_navigates_and_commits() -> None:
    from emulator import buttons

    emu = _MenuEmulator()
    select_move(emu, emu, action=2)
    downs = sum(1 for k in emu.presses if k & buttons.KEY_DOWN)
    a_presses = sum(1 for k in emu.presses if k & buttons.KEY_A)
    assert downs == 2  # navigate to slot 2
    assert a_presses >= 2  # open the list + commit
