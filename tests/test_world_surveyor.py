"""world_surveyor: chart a multi-map overworld over a fake WorldGrid (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DELTAS, OPPOSITE, WallMap
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot
from env.world_surveyor import SurveyReport, survey_world

_KEY_TO_DIR = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}


class WorldGrid:
    """Hidden per-map grids joined by borders; answers step + snapshot.

    walls: set[(map_id, cell, direction)] — a blocked directed move.
    borders: dict[(map_id, cell, direction)] -> (to_map, entry_cell).
    Movement inside a map is otherwise free.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        walls: set[tuple[tuple[int, int], tuple[int, int], str]],
        borders: dict[
            tuple[tuple[int, int], tuple[int, int], str],
            tuple[tuple[int, int], tuple[int, int]],
        ],
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._walls = walls
        self._borders = borders

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # release / noop
        key = (self.map_id, self.pos, direction)
        if key in self._borders:
            self.map_id, self.pos = self._borders[key]
            return
        if key in self._walls:
            return  # wall: stay put
        dx, dy = DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        # Always full: watcher stays quiet, no healing behaviour change.
        return [(1, 1)]

    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False


def _sealed_room(
    map_id: tuple[int, int], width: int, height: int
) -> set[tuple[tuple[int, int], tuple[int, int], str]]:
    """Every outward boundary edge of a width x height room, keyed by map_id."""
    walls: set[tuple[tuple[int, int], tuple[int, int], str]] = set()
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if x == 0:
                walls.add((map_id, cell, "left"))
            if x == width - 1:
                walls.add((map_id, cell, "right"))
            if y == 0:
                walls.add((map_id, cell, "up"))
            if y == height - 1:
                walls.add((map_id, cell, "down"))
    return walls


def _reversible_border(
    a_map: tuple[int, int], a_cell: tuple[int, int], direction: str,
    b_map: tuple[int, int], b_cell: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int], str],
          tuple[tuple[int, int], tuple[int, int]]]:
    """A two-way door: a_map@a_cell --direction--> b_map@b_cell and back."""
    return {
        (a_map, a_cell, direction): (b_map, b_cell),
        (b_map, b_cell, OPPOSITE[direction]): (a_map, a_cell),
    }


class _BlindReader:
    """A reader that never settles — snapshot() always returns None."""

    def snapshot(self) -> None:
        return None

    def step(self, keys: int, frames: int) -> None:  # emulator half, unused
        return None


def test_no_start_returns_no_start_failure() -> None:
    reader = _BlindReader()
    report = survey_world(reader, reader, MapMemory(), WallMap(), max_maps=10)

    assert report.surveyed == ()
    assert len(report.failed) == 1
    assert report.failed[0][1] == "no_start"


def test_two_maps_linked_by_reversible_border() -> None:
    a, b = (0, 0), (0, 1)
    # Each map is a fully-sealed 2x1 room. The door edges live in `borders`,
    # and WorldGrid.step checks borders BEFORE walls, so a door tile transitions
    # even though it is also a boundary wall — no need to punch holes in `walls`
    # (punching a hole on a NON-door edge would open an infinite void).
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1)
    borders = _reversible_border(a, (1, 0), "right", b, (0, 0))
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    assert isinstance(report, SurveyReport)
    assert set(report.surveyed) == {a, b}
    assert report.failed == ()


def _one_way_border(
    a_map: tuple[int, int], a_cell: tuple[int, int], direction: str,
    b_map: tuple[int, int], b_cell: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int], str],
          tuple[tuple[int, int], tuple[int, int]]]:
    """A one-way building warp: a_map@a_cell --direction--> b_map@b_cell, no return."""
    return {(a_map, a_cell, direction): (b_map, b_cell)}


def test_chain_of_three_maps_surveyed_in_bfs_order() -> None:
    a, b, c = (0, 0), (0, 1), (0, 2)
    # Fully-sealed rooms; doors live in `borders` (checked before walls).
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1) | _sealed_room(c, 2, 1)
    borders = {
        **_reversible_border(a, (1, 0), "right", b, (0, 0)),
        **_reversible_border(b, (1, 0), "right", c, (0, 0)),
    }
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    assert report.surveyed == (a, b, c)
    assert report.failed == ()


def test_warp_only_map_is_logged_left_map_and_target_never_enqueued() -> None:
    # Start map A's ONLY exit is a one-way building warp to H (down from (0,0)).
    # map_map(A) probes the warp -> "left_map"; A is still counted surveyed and
    # logged as map:left_map. H is a non-reversible target: never enqueued.
    a, h = (0, 0), (5, 5)
    walls = _sealed_room(a, 1, 1)          # single-cell room, fully sealed
    borders = _one_way_border(a, (0, 0), "down", h, (0, 0))  # warp wins over the wall
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=10)

    assert report.surveyed == (a,)                 # A surveyed despite the early exit
    assert (a, "map:left_map") in report.failed    # map-side failure logged
    assert h not in report.surveyed                # warp target never surveyed
    assert all(m != h for (m, _r) in report.failed)  # nor even attempted


def test_max_maps_stops_cleanly_with_partial_report() -> None:
    a, b, c = (0, 0), (0, 1), (0, 2)
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1) | _sealed_room(c, 2, 1)
    borders = {
        **_reversible_border(a, (1, 0), "right", b, (0, 0)),
        **_reversible_border(b, (1, 0), "right", c, (0, 0)),
    }
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), WallMap(), max_maps=1)

    assert report.surveyed == (a,)        # exactly one map surveyed then stopped
