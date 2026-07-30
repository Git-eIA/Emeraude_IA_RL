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
