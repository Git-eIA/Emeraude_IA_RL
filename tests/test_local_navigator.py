"""local_navigator: observed-collision detector, WallMap, A* (no ROM, no emulator)."""
from __future__ import annotations

from env.local_navigator import (
    DIRECTIONS,
    resolve_move,
)
from env.world_reader import WorldSnapshot


def _snap(map_id: tuple[int, int], pos: tuple[int, int]) -> WorldSnapshot:
    return WorldSnapshot(map_id=map_id, pos=pos, tile_behavior=None)


def test_directions_are_the_four_cardinals() -> None:
    assert set(DIRECTIONS) == {"up", "down", "left", "right"}


def test_moved_when_pos_changes_same_map() -> None:
    before = _snap((0, 9), (2, 3))
    after = _snap((0, 9), (2, 2))
    assert resolve_move(before, after) == "moved"


def test_blocked_when_pos_unchanged_same_map() -> None:
    before = _snap((0, 9), (2, 3))
    after = _snap((0, 9), (2, 3))
    assert resolve_move(before, after) == "blocked"


def test_transition_when_map_changes() -> None:
    before = _snap((0, 9), (2, 3))
    after = _snap((0, 16), (5, 10))
    assert resolve_move(before, after) == "transition"
