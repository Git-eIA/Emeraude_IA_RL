"""orders: the shared "order" language between the three brains.

The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. "advance" navigates via travel_to;
"heal" travels to a known healing spot and presses A until the party is full;
"grind" is stubbed. The combat directive is stored for a future Fighter hookup.
No Strategist, no reward here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emulator import buttons
from env.heal_detector import party_is_full
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
    "route_101": ((0, 16), (5, 12)),   # Route 101 south entrance (cell unverified)
}

HEAL_PRESS_A_FRAMES = 6
HEAL_RELEASE_FRAMES = 10
HEAL_MAX_PRESSES = 60   # bound the interaction (code-safety #2)


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
    if order.mode == "heal":
        return _execute_heal(emulator, reader, memory, wallmap, max_hops=max_hops)
    if order.mode != "advance":
        return "not_implemented"   # grind wiring is a later step
    goal_map, goal_cell = dest
    return travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )


def _execute_heal(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
) -> str:
    """Travel to a known healing spot, then press A until the party is full.

    Returns "no_healing_spot_known" | a travel_to pass-through | "healed" |
    "heal_failed".
    """
    spots = memory.healing_spots()
    if not spots:
        return "no_healing_spot_known"
    goal_map, goal_cell = spots[0]   # v1: the first known spot (nearest-choice is later)
    outcome = travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    return _heal_here(emulator, reader)


def _heal_here(emulator: Any, reader: Any) -> str:
    """Advance the nurse dialog (press A) until the party reads full HP."""
    for _ in range(HEAL_MAX_PRESSES):
        if party_is_full(reader.party_hp()):
            return "healed"
        emulator.step(buttons.KEY_A, HEAL_PRESS_A_FRAMES)
        emulator.step(0, HEAL_RELEASE_FRAMES)   # release between presses (GBA debounce)
    return "healed" if party_is_full(reader.party_hp()) else "heal_failed"
