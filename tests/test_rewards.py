from __future__ import annotations

from env.game_state import PlayerState
from env.rewards import ExplorationTracker, REWARD_PER_LEVEL, LevelRewardTracker


def state(x: int, y: int, group: int = 0, num: int = 0) -> PlayerState:
    return PlayerState(x=x, y=y, map_group=group, map_num=num, badges=0, party_count=0)


def test_new_tile_rewards_once():
    tracker = ExplorationTracker()
    assert tracker.update(state(1, 1)) == 1.0
    assert tracker.update(state(1, 1)) == 0.0
    assert tracker.update(state(2, 1)) == 1.0


def test_same_coords_on_different_map_are_distinct():
    tracker = ExplorationTracker()
    assert tracker.update(state(1, 1, group=0, num=0)) == 1.0
    assert tracker.update(state(1, 1, group=0, num=1)) == 1.0


def test_none_state_gives_zero():
    tracker = ExplorationTracker()
    assert tracker.update(None) == 0.0


def test_visited_count():
    tracker = ExplorationTracker()
    tracker.update(state(1, 1))
    tracker.update(state(2, 1))
    tracker.update(state(2, 1))
    assert tracker.visited_count == 2


def test_reset_clears_history():
    tracker = ExplorationTracker()
    tracker.update(state(1, 1))
    tracker.reset()
    assert tracker.visited_count == 0
    assert tracker.update(state(1, 1)) == 1.0


def test_level_empty_party_gives_zero():
    assert LevelRewardTracker().update([]) == 0.0


def test_level_gain_pays_once():
    tracker = LevelRewardTracker()
    assert tracker.update([5]) == 5 * REWARD_PER_LEVEL
    assert tracker.update([5]) == 0.0
    assert tracker.update([6]) == REWARD_PER_LEVEL


def test_level_sum_across_party():
    tracker = LevelRewardTracker()
    tracker.update([5])
    assert tracker.update([5, 3]) == 3 * REWARD_PER_LEVEL


def test_level_drop_gives_zero_not_negative():
    tracker = LevelRewardTracker()
    tracker.update([5, 3])
    assert tracker.update([5]) == 0.0


def test_level_reset():
    tracker = LevelRewardTracker()
    tracker.update([5])
    tracker.reset()
    assert tracker.update([5]) == 5 * REWARD_PER_LEVEL
