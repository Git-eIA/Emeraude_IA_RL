"""campaign: a scripted Strategist that drives the Order loop over milestones.

The chef holds a hand-written curriculum of (named destination, required mean
party level). For each milestone: if the team is under the required level, emit a
level_up Order (which grinds + heals itself to the target); then emit an advance
Order to reach the destination. run_campaign composes execute_order — it adds no
navigation, combat, or RAM logic of its own, only the sequencing.

advance is navigation-only in v1 (reach the place); fighting the leader there is
deferred. No trained Strategist, no capture directive here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env.orders import Order, _reached, execute_order


@dataclass(frozen=True)
class Milestone:
    """One curriculum step: reach `destination` once the mean party level is at
    least `target_level`."""

    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # required mean party level before advancing


# Hand-written curriculum. Like DESTINATIONS, a name means something to the chef
# before any exploration. Seeded minimally; extend as destinations are verified.
CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("route_101", 5),
)


def run_campaign(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
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
    Order | any non-"arrived" outcome from an advance Order.
    """
    for milestone in curriculum:
        if not _reached(reader.party_levels(), milestone.target_level):
            leveled = order_fn(
                Order(milestone.destination, "level_up", "win"),
                emulator, reader, memory, wallmap,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                target_level=milestone.target_level, heal_threshold=heal_threshold,
                max_cycles=max_cycles,
            )
            if leveled != "leveled_up":
                return leveled
        advanced = order_fn(
            Order(milestone.destination, "advance", "win"),
            emulator, reader, memory, wallmap, max_hops=max_hops,
        )
        if advanced != "arrived":
            return advanced
    return "campaign_complete"
