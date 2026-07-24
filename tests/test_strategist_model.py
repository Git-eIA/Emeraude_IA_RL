"""Calibration physics: win-prob shape, grind/heal effects (no Gym)."""
from __future__ import annotations

from env.strategist_model import grind, heal, win_prob


def test_equal_level_full_hp_is_favorable() -> None:
    # sigmoid(WIN_C) = sigmoid(1.0) ~= 0.73
    p = win_prob(team_level=10.0, challenge_level=10.0, team_hp=1.0)
    assert abs(p - 0.731) < 0.01


def test_under_level_by_four_is_unlikely() -> None:
    # z = 0.70*(-4) + 0 + 1.0 = -1.8 -> sigmoid ~= 0.14
    p = win_prob(team_level=6.0, challenge_level=10.0, team_hp=1.0)
    assert abs(p - 0.142) < 0.01


def test_equal_level_low_hp_is_a_coin_flip() -> None:
    # z = 0 + 1.5*(0.4 - 1.0) + 1.0 = 0.1 -> sigmoid ~= 0.52
    p = win_prob(team_level=10.0, challenge_level=10.0, team_hp=0.4)
    assert abs(p - 0.525) < 0.01


def test_grind_gains_a_level_and_loses_hp() -> None:
    level, hp = grind(team_level=5.0, team_hp=1.0)
    assert level == 6.0
    assert abs(hp - 0.70) < 1e-9


def test_grind_clamps_hp_at_zero() -> None:
    _, hp = grind(team_level=5.0, team_hp=0.20)
    assert hp == 0.0


def test_heal_restores_full_hp() -> None:
    assert heal() == 1.0
