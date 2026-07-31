"""heal_detector: pure HP-full detection, no emulator."""
from __future__ import annotations

from env.heal_detector import HealWatcher, party_is_full


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
