"""campaign: a scripted Strategist that drives the Order loop over milestones.

The chef holds a hand-written curriculum of (named destination, required mean
party level). For each milestone: if the team is under the required level, emit a
level_up Order (which grinds + heals itself to the target); then emit an advance
Order to reach the destination. run_campaign composes execute_order — it adds no
navigation, combat, or RAM logic of its own, only the sequencing.

advance is navigation-only (reach the place); for trainer milestones a battle_trainer
Order is emitted after arrival. No trained Strategist, no capture directive here.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from env.map_memory import MapMemory
from env.orders import Order, execute_order, reached

# Map-group ids for the southbound return path (probe-confirmed).
ROUTE_103 = (0, 18)
OLDALE = (0, 10)
ROUTE_101 = (0, 16)
LITTLEROOT = (0, 9)
LAB = (1, 4)

# Southbound return crossings, hand-seeded because a fresh savestate load carries
# an empty MapMemory. from_cell/to_cell are hand-seeded candidates; direction and
# reversibility are the real overworld/warp semantics. NOTE: the intermediate
# to_cell landing tiles ((0, 0) for Oldale/route_101) are ASSUMED, not probe-
# verified — the Task 6 probe confirmed the flags/items/SaveBlock2 and start map,
# not these crossings. The Task 7 ROM smoke validates the landings on first run.
class _PortalSeed(NamedTuple):
    from_map: tuple[int, int]
    from_cell: tuple[int, int]
    direction: str
    to_map: tuple[int, int]
    reversible: bool
    to_cell: tuple[int, int]


_RETURN_PORTALS: tuple[_PortalSeed, ...] = (
    _PortalSeed(ROUTE_103, (0, 18), "down", OLDALE, True, (0, 0)),
    _PortalSeed(OLDALE, (0, 9), "down", ROUTE_101, True, (0, 0)),
    _PortalSeed(ROUTE_101, (0, 19), "down", LITTLEROOT, True, (10, 1)),
    _PortalSeed(LITTLEROOT, (3, 10), "up", LAB, False, (6, 12)),
)


def seed_return_portals(memory: MapMemory) -> None:
    """Register the 4 southbound return edges so travel_to can path home.

    A fresh post_rival.state load has an empty MapMemory (not serialized), so
    the first story milestone's travel_to would return 'unknown_route' on step
    zero. This hand-seeds route_103 -> Oldale -> route_101 -> Littleroot -> lab.
    """
    for from_map, from_cell, direction, to_map, reversible, to_cell in _RETURN_PORTALS:
        memory.record_portal(from_map, from_cell, direction, to_map, reversible, to_cell)


@dataclass(frozen=True)
class Milestone:
    """One curriculum step: reach `destination` once the mean party level is at
    least `target_level`; if `trainer`, fight the trainer there on arrival."""

    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # mean, not max — one powerhouse shouldn't unlock advance
    trainer: bool = False   # end the milestone with a battle_trainer Order
    story_target: Callable[[Any], bool] | None = None   # story mode: A-spam until this holds


# Hand-written curriculum. Like DESTINATIONS, a name means something to the chef
# before any exploration. Seeded minimally; extend as destinations are verified.
CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("route_101", 5),
    Milestone("route_103", 5, trainer=True),
)

# Phase 2 curriculum: return to Littleroot, enter Birch's lab, receive Pokédex +
# 5 Poké Balls (one cutscene -> the Balls target is idempotent), then the running
# shoes on Route 101. Return/enter are pure arrivals (advance); the rest are story.
PHASE2_CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("littleroot", 0),
    Milestone("lab", 0),
    Milestone("lab", 0, story_target=lambda r: r.has_pokedex()),
    Milestone("lab", 0, story_target=lambda r: r.has_item(0x4, 5)),
    Milestone("route_101_shoes", 0, story_target=lambda r: r.has_running_shoes()),
)


def run_campaign(
    emulator: Any,
    reader: Any,
    memory: Any,
    curriculum: tuple[Milestone, ...] = CAMPAIGN,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
    heal_threshold: float = 0.4,
    max_cycles: int = 50,
    order_fn: Any = execute_order,
) -> str:
    """Walk the curriculum: for each milestone, level_up if under the required
    mean level, then advance to the destination. Abort on the first non-terminal
    outcome, surfaced verbatim so a future Strategist can react.

    Returns "campaign_complete" | any non-"leveled_up" outcome from a level_up
    Order | any non-"arrived" outcome from an advance Order | any non-"won"
    outcome from a battle_trainer Order | any non-"story_done" outcome from a
    story Order.
    """
    for milestone in curriculum:
        if milestone.story_target is not None:
            told = order_fn(
                Order(milestone.destination, "story", "win"),
                emulator, reader, memory,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                story_target=milestone.story_target,
            )
            if told != "story_done":
                return told
            continue
        if not reached(reader.party_levels(), milestone.target_level):
            leveled = order_fn(
                Order(milestone.destination, "level_up", "win"),
                emulator, reader, memory,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                target_level=milestone.target_level, heal_threshold=heal_threshold,
                max_cycles=max_cycles,
            )
            if leveled != "leveled_up":
                return leveled
        advanced = order_fn(
            Order(milestone.destination, "advance", "win"),
            emulator, reader, memory,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
        if advanced != "arrived":
            return advanced
        if milestone.trainer:
            fought = order_fn(
                Order(milestone.destination, "battle_trainer", "win"),
                emulator, reader, memory,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
            )
            if fought != "won":
                return fought
    return "campaign_complete"
