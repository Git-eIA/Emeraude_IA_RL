"""orders: the shared "order" language between the three brains.

The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. This step wires only the navigation
mode ("advance") through to travel_to; "grind"/"heal" are stubbed and the combat
directive is stored for a future Fighter hookup. No Strategist, no reward here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env.map_traveler import travel_to


@dataclass(frozen=True)
class Order:
    """A single order: go to `destination`, in `mode`, fighting per `combat`."""

    destination: str   # named place, e.g. "route_101"
    mode: str          # "advance" | "grind" | "heal" (only "advance" acts now)
    combat: str        # "win" | "capture" | "min_loss" (stored for later)


# Hand-written name -> (map_id, cell) registry. Chosen over a map-memory lookup on
# purpose: a name means something to the chef before any exploration has happened.
# Cells are known landmarks; fix them here if one turns out wrong.
DESTINATIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "littleroot": ((0, 9), (3, 10)),   # Bourg-en-Vol, truck landing cell
    "route_101": ((0, 16), (5, 12)),   # Route 101, south entrance from Littleroot
}


def execute_order(
    order: Order,
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
) -> str:
    """Resolve the order's destination and hand navigation to travel_to.

    Returns "unknown_destination" | "not_implemented" | one of travel_to's
    outcomes ("arrived" | "unknown_route" | "unreachable" | "lost" | "timeout").
    """
    dest = DESTINATIONS.get(order.destination)
    if dest is None:
        return "unknown_destination"
    if order.mode != "advance":
        return "not_implemented"   # grind/heal wiring is a later step
    goal_map, goal_cell = dest
    return travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
