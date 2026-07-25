"""Unit tests for map_map with an ExploreWorld fake (plays emulator + reader).

ExploreWorld models one hidden grid keyed by map_id, with `walls` (blocked
directed edges the survey must learn by bumping) and optional `borders`
(reversible/non-reversible map crossings). It exposes both the emulator API
(`step(keys, frames)` decodes the d-pad bit and moves on the hidden grid) and
the reader API (`snapshot()` -> WorldSnapshot). No ROM, no emulator.
"""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DELTAS, WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot

_KEY_TO_DIR = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}


class ExploreWorld:
    """Hidden grid that answers both emulator.step and reader.snapshot."""

    def __init__(
        self,
        map_id: tuple[int, int],
        start: tuple[int, int],
        walls: set[tuple[tuple[int, int], str]],
        borders: dict[
            tuple[tuple[int, int], tuple[int, int], str],
            tuple[tuple[int, int], tuple[int, int]],
        ]
        | None = None,
    ) -> None:
        self.map_id = map_id
        self.pos = start
        self._walls = walls
        self._borders = borders or {}
        self.presses = 0

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # release frame: no movement
        self.presses += 1
        key = (self.map_id, self.pos, direction)
        if key in self._borders:
            to_map, entry = self._borders[key]
            self.map_id = to_map
            self.pos = entry
            return
        if (self.pos, direction) in self._walls:
            return  # wall: stay put
        dx, dy = DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def _sealed_room_walls(
    width: int, height: int
) -> set[tuple[tuple[int, int], str]]:
    """All outward edges on the boundary of a width x height room at origin."""
    walls: set[tuple[tuple[int, int], str]] = set()
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if x == 0:
                walls.add((cell, "left"))
            if x == width - 1:
                walls.add((cell, "right"))
            if y == 0:
                walls.add((cell, "up"))
            if y == height - 1:
                walls.add((cell, "down"))
    return walls


def test_sealed_room_complete_survey():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)  # cells (0,0),(1,0),(0,1),(1,1)
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "complete"
    for (cell, direction) in walls:
        assert wallmap.is_blocked(target, cell, direction)
    assert memory.edges() == set()
