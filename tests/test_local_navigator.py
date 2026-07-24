"""local_navigator: observed-collision detector, WallMap, A* (no ROM, no emulator)."""
from __future__ import annotations

from env.local_navigator import (
    DIRECTIONS,
    WallMap,
    plan_path,
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


def test_wallmap_records_blocked_edge() -> None:
    walls = WallMap()
    walls.block((0, 9), (2, 3), "up")
    assert walls.is_blocked((0, 9), (2, 3), "up") is True


def test_wallmap_unknown_edge_is_optimistically_open() -> None:
    walls = WallMap()
    assert walls.is_blocked((0, 9), (2, 3), "up") is False


def test_wallmap_blocking_is_bidirectional() -> None:
    walls = WallMap()
    walls.block((0, 9), (2, 3), "up")   # wall between (2,3) and (2,2)
    # the neighbour going back the opposite way is also blocked
    assert walls.is_blocked((0, 9), (2, 2), "down") is True


def test_wallmap_is_per_map() -> None:
    walls = WallMap()
    walls.block((0, 9), (2, 3), "up")
    assert walls.is_blocked((0, 16), (2, 3), "up") is False


def test_path_start_equals_goal_is_empty() -> None:
    walls = WallMap()
    assert plan_path(walls, (0, 9), (2, 3), (2, 3)) == []


def test_path_straight_line_no_walls() -> None:
    walls = WallMap()
    # (0,0) -> (2,0): two steps right
    assert plan_path(walls, (0, 9), (0, 0), (2, 0)) == ["right", "right"]


def test_path_detours_around_a_wall() -> None:
    walls = WallMap()
    # Block the direct step right from (0,0); A* must go around via down.
    walls.block((0, 9), (0, 0), "right")
    path = plan_path(walls, (0, 9), (0, 0), (1, 0))
    assert path is not None
    # any valid detour reaches the goal; verify by walking it
    x, y = 0, 0
    for d in path:
        assert walls.is_blocked((0, 9), (x, y), d) is False
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        x, y = x + dx, y + dy
    assert (x, y) == (1, 0)


def test_path_none_when_fully_walled_in() -> None:
    walls = WallMap()
    for d in ("up", "down", "left", "right"):
        walls.block((0, 9), (0, 0), d)
    assert plan_path(walls, (0, 9), (0, 0), (5, 5)) is None


def test_path_replans_after_new_wall_discovered() -> None:
    walls = WallMap()
    # first plan goes straight right
    first = plan_path(walls, (0, 9), (0, 0), (2, 0))
    assert first == ["right", "right"]
    # discover a wall on the direct route, replan
    walls.block((0, 9), (1, 0), "right")
    second = plan_path(walls, (0, 9), (0, 0), (2, 0))
    assert second is not None
    assert second != first  # forced to detour
