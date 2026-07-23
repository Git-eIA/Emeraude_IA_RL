"""Pure unit tests for the battle reward table (no ROM, no env)."""
from __future__ import annotations

from env.battle_rewards import (
    DEAL_BAR,
    ENEMY_FAINT,
    OWN_FAINT,
    SUPER_EFFECTIVE,
    TAKE_BAR,
    TURN_PENALTY,
    WIN,
    BattleRewardTracker,
)
from env.game_state import BattleState, MoveInfo

_NO_MOVES = (MoveInfo(0, 0), MoveInfo(0, 0), MoveInfo(0, 0), MoveInfo(0, 0))


def _state(
    *,
    my_hp: int = 20,
    my_max: int = 20,
    opp_hp: int = 20,
    opp_max: int = 20,
    outcome: int = 0,
    in_battle: bool = True,
    super_effective: bool = False,
) -> BattleState:
    return BattleState(
        in_battle=in_battle,
        my_hp=my_hp, my_max_hp=my_max, my_level=5, my_types=(12, 12),
        my_moves=_NO_MOVES,
        opp_hp=opp_hp, opp_max_hp=opp_max, opp_level=5, opp_types=(10, 10),
        opp_species=4, outcome=outcome, last_move_super_effective=super_effective,
    )


def test_dealing_damage_pays_deal_bar_times_fraction() -> None:
    t = BattleRewardTracker()
    t.reset(_state(opp_hp=20, opp_max=20))
    r = t.update(_state(opp_hp=10, opp_max=20))  # 50% bar removed
    assert r == DEAL_BAR * 0.5 + TURN_PENALTY


def test_taking_damage_pays_take_bar_times_fraction() -> None:
    t = BattleRewardTracker()
    t.reset(_state(my_hp=20, my_max=20))
    r = t.update(_state(my_hp=15, my_max=20))  # 25% bar lost
    assert r == TAKE_BAR * 0.25 + TURN_PENALTY


def test_dealing_is_worth_twice_taking() -> None:
    assert DEAL_BAR == -2 * TAKE_BAR


def test_enemy_faint_pays_bonus() -> None:
    t = BattleRewardTracker()
    t.reset(_state(opp_hp=5, opp_max=20))
    r = t.update(_state(opp_hp=0, opp_max=20))
    assert r == DEAL_BAR * 0.25 + ENEMY_FAINT + TURN_PENALTY


def test_own_faint_pays_penalty() -> None:
    t = BattleRewardTracker()
    t.reset(_state(my_hp=5, my_max=20))
    r = t.update(_state(my_hp=0, my_max=20))
    assert r == TAKE_BAR * 0.25 + OWN_FAINT + TURN_PENALTY


def test_win_pays_win_bonus() -> None:
    t = BattleRewardTracker()
    t.reset(_state())
    r = t.update(_state(outcome=1, in_battle=False))
    assert r == WIN + TURN_PENALTY


def test_super_effective_pays_bonus() -> None:
    t = BattleRewardTracker()
    t.reset(_state(opp_hp=20, opp_max=20))
    r = t.update(_state(opp_hp=10, opp_max=20, super_effective=True))
    assert r == DEAL_BAR * 0.5 + SUPER_EFFECTIVE + TURN_PENALTY


def test_no_flat_defeat_penalty() -> None:
    t = BattleRewardTracker()
    t.reset(_state(my_hp=0, my_max=20))
    r = t.update(_state(my_hp=0, my_max=20, outcome=2, in_battle=False))
    assert r == TURN_PENALTY
