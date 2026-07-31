"""heal_detector: pure HP-full detection, no emulator."""
from __future__ import annotations

from env.heal_detector import HealWatcher, party_is_full, party_needs_heal


def test_party_is_full_true_when_all_at_max() -> None:
    assert party_is_full([(34, 34), (5, 5)]) is True


def test_party_is_full_false_when_a_member_is_hurt() -> None:
    assert party_is_full([(34, 34), (3, 5)]) is False


def test_party_is_full_false_when_empty() -> None:
    assert party_is_full([]) is False


def test_watcher_no_heal_while_staying_full() -> None:
    w = HealWatcher()
    assert w.observe([(5, 5)]) is False   # first read, already full
    assert w.observe([(5, 5)]) is False


def test_watcher_fires_once_on_transition_to_full() -> None:
    w = HealWatcher()
    assert w.observe([(2, 5)]) is False   # hurt
    assert w.observe([(1, 5)]) is False   # still hurt
    assert w.observe([(5, 5)]) is True    # healed!
    assert w.observe([(5, 5)]) is False   # stays full, no re-fire


def test_needs_heal_true_when_a_member_is_ko_even_if_totals_fine() -> None:
    # 5/10 total is above the 0.4 threshold, but a fainted member forces a heal.
    assert party_needs_heal([(0, 5), (5, 5)], 0.4) is True


def test_needs_heal_true_when_total_fraction_below_threshold() -> None:
    # Nobody KO'd, but 3/10 = 0.3 < 0.4.
    assert party_needs_heal([(1, 5), (2, 5)], 0.4) is True


def test_needs_heal_false_when_full_and_above_threshold() -> None:
    assert party_needs_heal([(5, 5), (4, 5)], 0.4) is False


def test_needs_heal_false_when_empty() -> None:
    assert party_needs_heal([], 0.4) is False
