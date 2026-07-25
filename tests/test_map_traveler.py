"""map_traveler: door-to-door inter-map travel over a fake multi-map world (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DIRECTIONS, WallMap
from env.map_memory import MapMemory
from env.map_traveler import travel_to
from env.world_reader import WorldSnapshot

_KEY_TO_DIR: dict[int, str] = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}
_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class MultiMapWorld:
    """Several hidden grids joined by border crossings, for the traveler.

    Acts as emulator (`step`) and reader (`snapshot`).
    `borders` maps (map_id, cell, direction) -> (next_map, entry_cell): pressing
    `direction` on `cell` of `map_id` drops the player onto `entry_cell` of
    `next_map`. `walls` blocks a (map_id, cell, direction) move. Movement inside a
    map is otherwise free.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        borders: dict[tuple[tuple[int, int], tuple[int, int], str],
                       tuple[tuple[int, int], tuple[int, int]]] | None = None,
        walls: set[tuple[tuple[int, int], tuple[int, int], str]] | None = None,
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._borders = dict(borders or {})
        self._walls = set(walls or ())

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        border = self._borders.get((self.map_id, self.pos, direction))
        if border is not None:
            self.map_id, self.pos = border
            return
        if (self.map_id, self.pos, direction) in self._walls:
            return  # wall: no move
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot | None:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def test_same_map_delegates_to_navigate() -> None:
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, MapMemory(), WallMap(),
        goal_map=(0, 0), goal_cell=(2, 0),
    )
    assert result == "arrived"
    assert world.pos == (2, 0)


def test_single_hop_crosses_one_known_door() -> None:
    # Map A=(0,0): door at (2,0) pressing 'right' lands on map B=(0,1) cell (0,0).
    borders = {((0, 0), (2, 0), "right"): ((0, 1), (0, 0))}
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "arrived"
    assert world.map_id == (0, 1)
    assert world.pos == (1, 0)


from env.map_memory import WorldEvent


def test_three_map_chain() -> None:
    # A=(0,0) --right@(2,0)--> B=(0,1) --right@(2,0)--> C=(0,2)
    borders = {
        ((0, 0), (2, 0), "right"): ((0, 1), (0, 0)),
        ((0, 1), (2, 0), "right"): ((0, 2), (0, 0)),
    }
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    memory.record_portal((0, 1), (2, 0), "right", (0, 2))
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 2), goal_cell=(1, 0),
    )
    assert result == "arrived"
    assert world.map_id == (0, 2)
    assert world.pos == (1, 0)


def test_unknown_route_when_portal_missing() -> None:
    # Edge A->B exists (from observe) but no portal was ever recorded.
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 0), (0, 0), None), WorldEvent())
    memory.observe(WorldSnapshot((0, 1), (0, 0), None), WorldEvent())
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unknown_route"


def test_unknown_route_when_goal_never_visited() -> None:
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, MapMemory(), WallMap(),
        goal_map=(9, 9), goal_cell=(0, 0),
    )
    assert result == "unknown_route"


def test_unreachable_when_door_cell_is_walled_off() -> None:
    # Start is sealed on all sides: the door cell can never be reached.
    walls = {((0, 0), (0, 0), d) for d in DIRECTIONS}
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), walls=walls)
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unreachable"


def test_lost_when_crossing_lands_on_unexpected_map() -> None:
    # Portal claims the door leads to B=(0,1), but the world sends us to C=(0,5).
    borders = {((0, 0), (2, 0), "right"): ((0, 5), (0, 0))}
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    result = travel_to(
        world, world, memory, WallMap(),
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "lost"
