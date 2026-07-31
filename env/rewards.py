"""Reward shaping: small bonus for never-visited tiles, small penalty for revisits."""
from __future__ import annotations

from env.game_state import PlayerState

# Small enough that milestones dominate; large enough to make loitering lose.
REVISIT_PENALTY = -0.01
# Palier 1: cut to 0.0. Even at +0.5/tile the agent kept farming fresh tiles in
# large towns instead of following the milestone chain, and a strong 9.9M
# checkpoint collapsed to route_101 0/10 after more training (detachment). With
# no per-tile income, the milestone chain is the only positive reward worth
# chasing; a new tile now pays only TIME_PENALTY but still beats a revisit.
NEW_TILE_REWARD = 0.0
# Flat cost per step: makes the shortest path to the chain the best-paying
# one (-82 max over a 4096-step episode vs +290 for the full chain).
TIME_PENALTY = -0.02


class ExplorationTracker:
    """NEW_TILE_REWARD the first time each (map_group, map_num, x, y) tile is seen; REVISIT_PENALTY after."""

    def __init__(self) -> None:
        self._visited: set[tuple[int, int, int, int]] = set()

    @property
    def visited_count(self) -> int:
        return len(self._visited)

    def reset(self) -> None:
        self._visited.clear()

    def update(self, state: PlayerState | None) -> float:
        if state is None:
            return 0.0
        tile = (state.map_group, state.map_num, state.x, state.y)
        if tile in self._visited:
            return REVISIT_PENALTY
        self._visited.add(tile)
        return NEW_TILE_REWARD


REWARD_PER_LEVEL = 5.0


class LevelRewardTracker:
    """Pays REWARD_PER_LEVEL once per party level gained (sum over slots).

    Tracks the best sum seen so a level drop (deposit, trade) never pays
    negative reward nor re-pays on recovery.
    """

    def __init__(self) -> None:
        self._best_sum = 0

    def reset(self) -> None:
        self._best_sum = 0

    def update(self, levels: list[int]) -> float:
        total = sum(levels)
        if total <= self._best_sum:
            return 0.0
        gained = total - self._best_sum
        self._best_sum = total
        return REWARD_PER_LEVEL * gained
