"""orders: the shared Order language + execute_order (pure, no ROM)."""
from __future__ import annotations

import dataclasses

from emulator import buttons
from env.local_navigator import WallMap
from env.map_memory import MapMemory
from env.orders import DESTINATIONS, Order, execute_order
from env.world_reader import WorldSnapshot


def test_order_is_a_frozen_dataclass_with_three_fields() -> None:
    order = Order(destination="route_101", mode="advance", combat="win")
    assert order.destination == "route_101"
    assert order.mode == "advance"
    assert order.combat == "win"
    try:
        order.mode = "grind"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Order must be frozen")


def test_destinations_registry_holds_known_places() -> None:
    assert DESTINATIONS["littleroot"] == ((0, 9), (3, 10))
    assert DESTINATIONS["route_101"] == ((0, 16), (5, 12))


def test_unknown_destination_returns_unknown_destination() -> None:
    order = Order(destination="atlantide", mode="advance", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "unknown_destination"


def test_non_nav_mode_is_not_implemented_even_for_a_known_place() -> None:
    order = Order(destination="littleroot", mode="grind", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "not_implemented"


_KEY_TO_DIR: dict[int, str] = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}
_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class NamedWorld:
    """Hidden multi-map grid that plays emulator (step) and reader (snapshot).

    `borders` maps (map_id, cell, direction) -> (next_map, entry_cell). All
    other moves are free (no wall simulation needed for these tests).
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        borders: dict[tuple[tuple[int, int], tuple[int, int], str],
                      tuple[tuple[int, int], tuple[int, int]]] | None = None,
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._borders = dict(borders or {})

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        border = self._borders.get((self.map_id, self.pos, direction))
        if border is not None:
            self.map_id, self.pos = border
            return
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)


def test_advance_to_same_map_destination_arrives() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="littleroot", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "arrived"
    assert world.map_id == (0, 9)
    assert world.pos == (3, 10)


def test_advance_across_one_known_door_arrives() -> None:
    borders = {((0, 9), (2, 10), "right"): ((0, 16), (0, 12))}
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10), borders=borders)
    memory = MapMemory()
    memory.record_portal(
        (0, 9), (2, 10), "right", (0, 16), reversible=True, to_cell=(0, 12)
    )
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "arrived"
    assert world.map_id == (0, 16)
    assert world.pos == (5, 12)


def test_advance_passes_through_unknown_route() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "unknown_route"
