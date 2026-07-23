from __future__ import annotations

from env.game_state import PlayerState
from env.milestones import Milestone, MilestoneTracker, starter_milestones


def make_state(**overrides) -> PlayerState:
    # Neutral defaults: Oldale (0, 10) at y=5 fires no milestone.
    defaults = dict(x=0, y=5, map_group=0, map_num=10, badges=0, party_count=0, clock_set=False, town_state=0)
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


def test_exit_truck_milestone():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=0, map_num=9))
    assert reward == 5.0
    assert terminated is False
    assert tracker.fired == frozenset({"exit_truck"})


def test_enter_house_fires_for_both_houses():
    for map_num in (0, 2):
        tracker = MilestoneTracker(starter_milestones())
        reward, _ = tracker.update(make_state(map_group=1, map_num=map_num))
        assert reward == 5.0
        assert tracker.fired == frozenset({"enter_house"})


def test_clock_set_milestone():
    tracker = MilestoneTracker(starter_milestones())
    reward, terminated = tracker.update(make_state(map_group=1, map_num=0, clock_set=True))
    # enter_house (+5) and clock_set (+15) fire together in the house
    assert reward == 20.0
    assert terminated is False


def test_back_outside_requires_clock():
    tracker = MilestoneTracker(starter_milestones())
    # Outside without clock: only exit_truck
    reward, _ = tracker.update(make_state(map_group=0, map_num=9))
    assert reward == 5.0
    assert "back_outside" not in tracker.fired
    # Outside with clock set: clock_set + back_outside fire
    reward, _ = tracker.update(make_state(map_group=0, map_num=9, clock_set=True))
    assert reward == 25.0
    assert "back_outside" in tracker.fired


def test_north_littleroot_fires_at_north_edge_with_clock_set():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(map_group=0, map_num=9, clock_set=True, y=1))
    assert "north_littleroot" in tracker.fired
    # exit_truck (5) + clock_set (15) + back_outside (10) + north_littleroot (10)
    assert reward == 40.0


def test_north_littleroot_needs_clock_set():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=9, clock_set=False, y=0))
    assert "north_littleroot" not in tracker.fired


def test_north_littleroot_not_fired_south_of_threshold():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=9, clock_set=True, y=2))
    assert "north_littleroot" not in tracker.fired


def test_north_littleroot_needs_littleroot_map():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=0, map_num=10, clock_set=True, y=0))
    assert "north_littleroot" not in tracker.fired


def test_enter_rival_house_requires_clock():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=1, map_num=2, clock_set=False))
    assert "enter_rival_house" not in tracker.fired
    assert "enter_house" in tracker.fired  # pre-intro visit still pays enter_house


def test_enter_rival_house_fires_with_clock():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(map_group=1, map_num=2, clock_set=True))
    assert "enter_rival_house" in tracker.fired
    # enter_house (5) + clock_set (15) + enter_rival_house (5) on a fresh tracker
    assert reward == 25.0


def test_rival_upstairs_fires_with_clock():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(map_group=1, map_num=3, clock_set=True))
    assert "rival_upstairs" in tracker.fired
    # clock_set (15) + rival_upstairs (5); (1,3) is not in PLAYER_HOUSES_1F
    assert reward == 20.0


def test_rival_upstairs_requires_clock():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(map_group=1, map_num=3, clock_set=False))
    assert "rival_upstairs" not in tracker.fired


def test_meet_rival_fires_on_town_state():
    tracker = MilestoneTracker(starter_milestones())
    reward, _ = tracker.update(make_state(clock_set=True, town_state=1))
    assert "meet_rival" in tracker.fired
    # clock_set (15) + meet_rival (15) on a fresh tracker
    assert reward == 30.0


def test_meet_rival_not_fired_at_zero():
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(clock_set=True, town_state=0))
    assert "meet_rival" not in tracker.fired


def test_meet_rival_requires_clock():
    # Guards against transient garbage reads during map warps: a one-step
    # glitch was observed reading clock_set=False alongside town_state=40.
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(clock_set=False, town_state=1))
    assert "meet_rival" not in tracker.fired


def test_meet_rival_rejects_out_of_range_town_state():
    # Legit TOWN_STATE values are 1..4; 40 came from a glitched RAM read.
    tracker = MilestoneTracker(starter_milestones())
    tracker.update(make_state(clock_set=True, town_state=40))
    assert "meet_rival" not in tracker.fired


def test_full_chain_sums_to_190():
    tracker = MilestoneTracker(starter_milestones())
    total = 0.0
    steps = (
        make_state(map_group=25, map_num=40),                 # in the truck: nothing
        make_state(map_group=0, map_num=9),                   # exit_truck +5
        make_state(map_group=1, map_num=0),                   # enter_house +5
        make_state(map_group=1, map_num=0, clock_set=True),   # clock_set +15
        make_state(map_group=0, map_num=9, clock_set=True),   # back_outside +10
        make_state(map_group=1, map_num=2, clock_set=True),   # enter_rival_house +5
        make_state(map_group=1, map_num=3, clock_set=True),   # rival_upstairs +5
        make_state(map_group=1, map_num=3, clock_set=True, town_state=1),  # meet_rival +15
        make_state(map_group=0, map_num=9, clock_set=True, town_state=1, y=1),  # north +10
        make_state(map_group=0, map_num=16, clock_set=True, town_state=2),  # route_101 +20
        make_state(map_group=0, map_num=16, clock_set=True, town_state=2, party_count=1),  # starter +100
    )
    terminated = False
    for state in steps:
        reward, terminated = tracker.update(state)
        total += reward
    assert total == 190.0
    assert terminated is True
    assert len(tracker.fired) == 10
