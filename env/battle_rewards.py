"""Battle reward shaping: HP measured in health bars (fraction of max HP)."""
from __future__ import annotations

from env.game_state import BattleState

# Tunable constants. Dealing damage is worth 2x taking it (DEAL_BAR = -2*TAKE_BAR).
DEAL_BAR = 10.0
TAKE_BAR = -5.0
WIN = 100.0
ENEMY_FAINT = 20.0
OWN_FAINT = -10.0
SUPER_EFFECTIVE = 5.0
TURN_PENALTY = -0.1


def _bar(hp: int, max_hp: int) -> float:
    """HP as a fraction of max HP (a health bar); 0.0 if max_hp is 0."""
    return hp / max_hp if max_hp > 0 else 0.0


class BattleRewardTracker:
    """Scores each turn from the previous and current BattleState.

    reset() seeds the previous state at battle start; update() returns the
    reward for the transition into the new state. There is no flat defeat
    penalty: losing is captured only through OWN_FAINT and TURN_PENALTY.
    """

    def __init__(self) -> None:
        self._prev: BattleState | None = None

    def reset(self, state: BattleState) -> None:
        self._prev = state

    def update(self, state: BattleState) -> float:
        prev = self._prev
        self._prev = state
        if prev is None:
            return 0.0

        reward = TURN_PENALTY

        opp_removed = _bar(prev.opp_hp, prev.opp_max_hp) - _bar(
            state.opp_hp, state.opp_max_hp
        )
        if opp_removed > 0:
            reward += DEAL_BAR * opp_removed
        my_lost = _bar(prev.my_hp, prev.my_max_hp) - _bar(
            state.my_hp, state.my_max_hp
        )
        if my_lost > 0:
            reward += TAKE_BAR * my_lost

        if prev.opp_hp > 0 and state.opp_hp == 0:
            reward += ENEMY_FAINT
        if prev.my_hp > 0 and state.my_hp == 0:
            reward += OWN_FAINT

        if state.last_move_super_effective:
            reward += SUPER_EFFECTIVE
        if prev.outcome == 0 and (state.outcome & 0x1):
            reward += WIN

        return reward
