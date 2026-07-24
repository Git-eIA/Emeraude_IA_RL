"""Calibration physics for the Strategist's abstract simulator.

Pure functions and seed constants — no Gym dependency, unit-testable in
isolation. This is the ONE place recalibrated later against measured
Fighter/game data; the MDP in strategist_env.py never changes.
"""
from __future__ import annotations

import math

# Grind: one wild-battle session.
GRIND_LEVEL_GAIN = 1.0   # levels gained per grind session
GRIND_HP_COST = 0.30     # HP fraction lost per grind session

# Advance: HP cost of a won important battle.
ADVANCE_HP_COST = 0.30

# Win-probability logistic: p = sigmoid(A*dlevel + B*(hp - 1) + C).
WIN_A = 0.70   # sensitivity to level gap
WIN_B = 1.50   # sensitivity to missing HP
WIN_C = 1.00   # bias: equal level + full HP -> p ~= 0.73


def win_prob(team_level: float, challenge_level: float, team_hp: float) -> float:
    """Probability the Fighter wins the important battle, in [0, 1]."""
    dlevel = team_level - challenge_level
    z = WIN_A * dlevel + WIN_B * (team_hp - 1.0) + WIN_C
    # Guard the exp: for very negative z, math.exp(-z) would overflow.
    if z < -700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))
