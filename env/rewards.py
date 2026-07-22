"""Reward shaping. v1: reward discovery of never-visited tiles (map-qualified)."""
from __future__ import annotations

from env.game_state import PlayerState


class ExplorationTracker:
    """Gives +1.0 the first time each (map_group, map_num, x, y) tile is seen."""

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
            return 0.0
        self._visited.add(tile)
        return 1.0
