from __future__ import annotations

from env.game_state import PlayerState
from env.milestones import Milestone, MilestoneTracker, starter_milestones


def make_state(**overrides) -> PlayerState:
    defaults = dict(x=0, y=0, map_group=0, map_num=9, badges=0, party_count=0)
    return PlayerState(**{**defaults, **overrides})


def test_milestone_fires_once():
    tracker = MilestoneTracker((Milestone("m", lambda s: s.x > 0, 10.0),))
    assert tracker.update(make_state(x=1)) == (10.0, False)
    assert tracker.update(make_state(x=1)) == (0.0, False)


def test_condition_not_met_gives_zero():
    tracker = MilestoneTracker((Milestone("m", lambda s: s.x > 0, 10.0),))
    assert tracker.update(make_state(x=0)) == (0.0, False)


def test_terminal_milestone_terminates():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(party_count=1))
    assert reward == 100.0
    assert terminated is True


def test_route_101_milestone():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=0, map_num=16))
    assert reward == 20.0
    assert terminated is False


def test_multiple_milestones_same_step_sum():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=0, map_num=16, party_count=1))
    assert reward == 120.0
    assert terminated is True


def test_none_state_gives_zero():
    tracker = MilestoneTracker(starter_milestones())
    assert tracker.update(None) == (0.0, False)


def test_reset_clears_fired():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(party_count=1))
    tracker.reset()
    assert tracker.update(make_state(party_count=1)) == (100.0, True)


def test_fired_names_exposed():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=16))
    assert tracker.fired == frozenset({"reach_route_101"})
