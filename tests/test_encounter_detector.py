"""EncounterWatcher: fire once on the not-in-battle -> in-battle edge (no ROM)."""
from __future__ import annotations

from env.encounter_detector import EncounterWatcher


def test_fires_on_absent_to_present_edge() -> None:
    w = EncounterWatcher()
    assert w.observe(False) is False   # walking, no battle
    assert w.observe(True) is True     # a wild battle just started


def test_silent_when_already_in_battle_on_first_read() -> None:
    w = EncounterWatcher()
    assert w.observe(True) is False    # optimistic init: not counted as a new start


def test_silent_on_present_to_absent() -> None:
    w = EncounterWatcher()
    w.observe(False)
    w.observe(True)
    assert w.observe(False) is False   # battle ended: not an edge we care about


def test_fires_again_after_a_battle_ends_and_a_new_one_starts() -> None:
    w = EncounterWatcher()
    w.observe(False)
    assert w.observe(True) is True     # first battle
    w.observe(False)                   # battle ends
    assert w.observe(True) is True     # a second battle starts
