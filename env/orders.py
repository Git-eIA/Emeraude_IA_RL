"""orders: the shared "order" language between the three brains.

The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. This step wires only the navigation
mode ("advance") through to travel_to; "grind"/"heal" are stubbed and the combat
directive is stored for a future Fighter hookup. No Strategist, no reward here.
"""
from __future__ import annotations

from dataclasses import dataclass


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
