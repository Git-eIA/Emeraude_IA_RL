"""orders: the shared Order language + execute_order (pure, no ROM)."""
from __future__ import annotations

import dataclasses

from env.orders import DESTINATIONS, Order


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


from env.local_navigator import WallMap
from env.map_memory import MapMemory
from env.orders import execute_order


def test_unknown_destination_returns_unknown_destination() -> None:
    order = Order(destination="atlantide", mode="advance", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "unknown_destination"


def test_non_nav_mode_is_not_implemented_even_for_a_known_place() -> None:
    order = Order(destination="littleroot", mode="grind", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "not_implemented"
