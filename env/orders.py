"""orders: the shared "order" language between the three brains.

The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. "advance" navigates via travel_to;
"heal" travels to a known healing spot and presses A until the party is full;
"grind" travels to a known grass cell and treads until a wild battle starts, then
the Fighter plays it; "level_up" loops grind to a target average level, healing
when the party gets low. The combat directive is stored for a future Fighter use.
No Strategist, no reward here.

ROM smoke for grind is deferred: it needs a savestate standing on/near grass
with a party and a deterministic encounter, which we do not have yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emulator import buttons
from env.battle_player import play_battle, play_trainer_battle
from env.encounter_detector import EncounterWatcher
from env.heal_detector import party_is_full, party_needs_heal
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
    "route_103": ((0, 18), (9, 5)),    # Route 103 rival approach (map+cell unverified)
}

# Engage heading per trainer destination: the d-pad direction that walks into the
# trainer to trigger the battle. Unverified until the capture tool discovers it.
TRAINER_APPROACH: dict[str, int] = {
    "route_103": buttons.KEY_UP,   # walk north into the rival (unverified)
}

HEAL_PRESS_A_FRAMES = 6
HEAL_RELEASE_FRAMES = 10
HEAL_MAX_PRESSES = 60   # bound the interaction (code-safety #2)

GRIND_STEP_FRAMES = 24
GRIND_RELEASE_FRAMES = 8
GRIND_MAX_STEPS = 60    # bound the walk (code-safety #2)


def execute_order(
    order: Order,
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
    target_level: int = 0,
    heal_threshold: float = 0.4,
    max_cycles: int = 50,
) -> str:
    """Execute an order dispatched by the Strategist.

    heal and grind are resolved first and ignore `destination` (pure intention:
    the healing spot / grass cell comes from memory, not the order's name).
    advance requires `destination` to be in DESTINATIONS. level_up loops grind to
    the given target_level (mean of the party), healing when party_needs_heal.

    Returns "unknown_destination" | "no_healing_spot_known" | "no_grass_spot_known" |
    one of travel_to's outcomes ("arrived" | "unknown_route" | "unreachable" |
    "lost" | "timeout" | "battle_lost" | "battle_timeout" | "battle_interrupted") |
    "healed" | "heal_failed" | "encounter_started" |
    "no_encounter" | a play_battle outcome ("won" | "lost" | "battle_timeout").
    level_up adds: "leveled_up" | "grind_exhausted".
    battle_trainer adds: "no_trainer" (Fighter wired, no battle triggered).
    """
    if order.mode == "heal":
        return _execute_heal(emulator, reader, memory, wallmap, max_hops=max_hops)
    if order.mode == "grind":
        return _execute_grind(
            emulator, reader, memory, wallmap,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
    if order.mode == "level_up":
        return _execute_level_up(
            emulator, reader, memory, wallmap,
            target_level=target_level, heal_threshold=heal_threshold,
            max_cycles=max_cycles, max_hops=max_hops,
            move_type_fn=move_type_fn, predict=predict,
        )
    if order.mode == "battle_trainer":
        return _execute_battle_trainer(
            emulator, reader, memory, wallmap, order.destination,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
    dest = DESTINATIONS.get(order.destination)
    if dest is None:
        return "unknown_destination"
    goal_map, goal_cell = dest
    return travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell,
        max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
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


def _execute_grind(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Travel to a known grass cell, tread until a wild battle starts, then—if a
    Fighter is supplied—play the battle to an outcome.

    Returns "no_grass_spot_known" | a travel_to pass-through | "no_encounter" |
    "encounter_started" (no Fighter) | a play_battle outcome
    ("won" | "lost" | "battle_timeout").
    """
    spots = memory.cells_labeled("has_grass")
    if not spots:
        return "no_grass_spot_known"
    goal_map, goal_cell = spots[0]   # v1: first known spot (nearest-choice is later)
    outcome = travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    result = _walk_until_encounter(emulator, reader)
    if result != "encounter_started":
        return result                # no_encounter
    if move_type_fn is None or predict is None:
        return "encounter_started"   # no Fighter wired: stop at the encounter
    return play_battle(emulator, move_type_fn, predict)


def _walk_until_encounter(emulator: Any, reader: Any) -> str:
    """Cycle d-pad directions in place until the battle flag rises."""
    watcher = EncounterWatcher()
    directions = (buttons.KEY_UP, buttons.KEY_DOWN, buttons.KEY_LEFT, buttons.KEY_RIGHT)
    for i in range(GRIND_MAX_STEPS):
        if watcher.observe(reader.in_battle()):
            return "encounter_started"
        emulator.step(directions[i % len(directions)], GRIND_STEP_FRAMES)
        emulator.step(0, GRIND_RELEASE_FRAMES)   # release between presses (GBA debounce)
    return "encounter_started" if reader.in_battle() else "no_encounter"


def reached(levels: list[int], target: int) -> bool:
    """True when the party's mean level is at or above the target."""
    return bool(levels) and sum(levels) / len(levels) >= target


def _execute_level_up(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    target_level: int,
    heal_threshold: float = 0.4,
    max_cycles: int = 50,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Grind wild battles until the party's mean level reaches target_level,
    healing whenever party_needs_heal fires. Composes _execute_grind (one battle)
    and _execute_heal (one heal); the loop is bounded by max_cycles.

    Returns "leveled_up" | "grind_exhausted" | a grind pass-through
    ("no_grass_spot_known" | "no_encounter"-driven continue | "lost" |
    "battle_timeout" | travel outcomes) | a heal pass-through
    ("no_healing_spot_known" | "heal_failed" | travel outcomes).

    NOTE: expects a Fighter (move_type_fn + predict). Without them the first
    _execute_grind returns "encounter_started", surfaced verbatim.
    """
    for _ in range(max_cycles):
        if reached(reader.party_levels(), target_level):
            return "leveled_up"
        result = _execute_grind(
            emulator, reader, memory, wallmap,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
        if result == "won":
            if party_needs_heal(reader.party_hp(), heal_threshold):
                healed = _execute_heal(
                    emulator, reader, memory, wallmap, max_hops=max_hops
                )
                if healed != "healed":
                    return healed
        elif result == "no_encounter":
            continue   # no battle this cycle (RNG); retry, budget-bounded
        else:
            return result
    return "leveled_up" if reached(reader.party_levels(), target_level) else "grind_exhausted"


def _execute_battle_trainer(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    destination: str,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Travel to a trainer's approach cell, engage, then—if a Fighter is
    supplied—play the trainer battle to an outcome.

    ROM smoke deferred: a mode-level smoke needs a pre-trigger overworld state
    near the trainer plus a surveyed route so travel_to can path there. A
    walk-until-in_battle capture tool produces an already-in-battle state, on
    which travel_to spins to timeout before the battle. Pure tests cover the
    sequencing; play_trainer_battle on a real rival is covered by Brique 3
    part 1's gated smoke.

    Returns "unknown_destination" | a travel_to pass-through | "no_trainer" |
    "encounter_started" (no Fighter) | a play_trainer_battle outcome
    ("won" | "lost" | "battle_timeout").
    """
    dest = DESTINATIONS.get(destination)
    if dest is None:
        return "unknown_destination"
    goal_map, goal_cell = dest
    outcome = travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell,
        max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    heading = TRAINER_APPROACH.get(destination, buttons.KEY_UP)
    engaged = _walk_until_trainer(emulator, reader, heading)
    if engaged != "engaged":
        return engaged               # no_trainer
    if move_type_fn is None or predict is None:
        return "encounter_started"   # no Fighter wired: stop at the trigger
    return play_trainer_battle(emulator, move_type_fn, predict)


def _walk_until_trainer(emulator: Any, reader: Any, heading: int) -> str:
    """Press a fixed approach heading in place until the battle flag rises."""
    watcher = EncounterWatcher()
    for _ in range(GRIND_MAX_STEPS):
        if watcher.observe(reader.in_battle()):
            return "engaged"
        emulator.step(heading, GRIND_STEP_FRAMES)
        emulator.step(0, GRIND_RELEASE_FRAMES)   # release between presses (GBA debounce)
    return "engaged" if reader.in_battle() else "no_trainer"
