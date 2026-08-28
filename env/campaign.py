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
from typing import Any

from emulator import buttons
from env.map_traveler import cross_scripted_npc, hop_via_explore, reach_map
from env.maps import LAB, LITTLEROOT, OLDALE, ROUTE_101, ROUTE_103
from env.orders import Order, _advance_story_dialogue, execute_order, reached

# Return chain for reach_map: cross route_101 south (down) into Littleroot, then the
# lab door warp (up). Oldale/route_103 are dropped — the north-entry Oldale hop is not
# crossable in a continuous descent (see the B2 spec), and B2 starts on route_101.
_RETURN_DIRECTIONS: dict[tuple[int, int], str] = {ROUTE_101: "down", LITTLEROOT: "up"}

# Settle frames after the route_103 -> Oldale hop, before the Flora walk (let the
# map transition finish so _precision_walk_to reads a stable grid).
_SETTLE_FRAMES = 60

# Flora gate: she stands on Oldale's south connection tile (11,19). Stand just north
# at (10,19) facing EAST, A-spam her dialogue, then DOWN crosses into route_101.
_FLORA_STAND = (10, 19)
_FLORA_FACE = "right"
_FLORA_CROSS = "down"
_FLORA_MAX_PRESSES = 60

# GivePokedex cutscene completion (2026-08-17 investigation + 2026-08-21 probe):
# the Pokédex flag flips ~55 A-presses BEFORE the 5 Poke Balls land and the script's
# releaseall; continued A-spam re-talks Birch, so B presses must drain the re-opened
# dialogue boxes before any direction input can move the player.
_RELEASE_B_PRESSES = 10  # 5+ works, 3 insufficient (probe-measured)
_BUTTON_FRAMES = 8       # A/B press and release frames (probe-proven cadence)

# Story-var completion values (TODOS.md I6). Both vars read None during save-block
# relocation windows; every predicate below compares with == so a None read is
# simply "not done yet", never a crash or a false positive.
_LAB_STATE_CUTSCENE_DONE = 5   # VAR_BIRCH_LAB_STATE after the full GivePokedex script
_TOWN_STATE_SHOES_DONE = 4     # VAR_LITTLEROOT_TOWN_STATE after the mom/shoes event

# run_shoes_leg bounds (probe-measured 2026-08-21, margin >= x2). Direction presses
# use the 12/4 cadence the probes and precision walks share.
_MOVE_PRESS_FRAMES = 12
_MOVE_REST_FRAMES = 4
_LAB_EXIT_MAX_PRESSES = 60  # measured: 11 DOWN presses lab -> Littleroot
_SHOES_MAX_CYCLES = 80      # measured: ~14 cycles to shoes + town_state 4
_SHOES_A_PER_CYCLE = 4
_CONTROL_MAX_CYCLES = 30    # measured: 1 cycle for control to return
_CONTROL_B_PRESSES = 2      # drain a box the drain's last A re-opened (probe P6)
_STATE_READ_RETRIES = 3     # transient None reads (save-block relocation window)
_STATE_READ_RETRY_FRAMES = 4


@dataclass(frozen=True)
class Milestone:
    """One curriculum step: reach `destination` once the mean party level is at
    least `target_level`; if `trainer`, fight the trainer there on arrival; if
    `reach` is set, greedy-descend to that goal map via reach_map instead."""

    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # mean, not max — one powerhouse shouldn't unlock advance
    trainer: bool = False   # end the milestone with a battle_trainer Order
    story_target: Callable[[Any], bool] | None = None   # story mode: A-spam until this holds
    reach: tuple[int, int] | None = None   # reach-home mode: reach_map(goal=reach)


# Hand-written curriculum. Like DESTINATIONS, a name means something to the chef
# before any exploration. Seeded minimally; extend as destinations are verified.
CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("route_101", 5),
    Milestone("route_103", 5, trainer=True),
)

# Phase 2 curriculum (B2). Start on route_101 (post_starter.state) and greedy-descend
# home to the lab. The Pokédex/Balls/shoes cutscene is NOT reachable from here: post_starter
# is BEFORE the route_103 rival, and Emerald arms the Pokédex event only on the SECOND lab
# visit (after that rival). Entering the lab from post_starter fires no cutscene (proven:
# the player walks freely, no dialogue lock, has_pokedex stays False). So the deliverable is
# the durable descent itself — arrival at the lab. The Pokédex objective moves to a future
# phase that solves the post-rival return (the uncrossable Oldale crux).
PHASE2_CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("lab", 0, reach=LAB),
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
        if milestone.reach is not None:
            arrived = reach_map(
                emulator, reader, memory, milestone.reach, _RETURN_DIRECTIONS,
                move_type_fn=move_type_fn, predict=predict,
            )
            if arrived != "arrived":
                return arrived
            continue
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


def _press(emulator: Any, key: int, hold: int, rest: int) -> None:
    """Press one key for hold frames, then rest with no input."""
    emulator.step(key, hold)
    emulator.step(0, rest)


def _finish_lab_cutscene(emulator: Any, reader: Any) -> bool:
    """Play the GivePokedex cutscene to completion and return WITH control.

    Stopping at has_pokedex() alone dumps a mid-cutscene state that resumes as a
    false control-lock; real completion is dex + 5 balls + VAR_BIRCH_LAB_STATE==5.
    After that, B presses close the Birch boxes the extra A-spam re-opened."""
    done = _advance_story_dialogue(
        emulator, reader,
        lambda r: r.has_pokedex()
        and r.has_poke_balls(5)
        and r.birch_lab_state() == _LAB_STATE_CUTSCENE_DONE,
    )
    if done != "story_done":
        return False
    for _ in range(_RELEASE_B_PRESSES):
        _press(emulator, buttons.KEY_B, _BUTTON_FRAMES, _BUTTON_FRAMES)
    return True


def _exit_lab(emulator: Any, reader: Any) -> bool:
    """Walk DOWN out of the lab until the map is Littleroot (bounded)."""
    for _ in range(_LAB_EXIT_MAX_PRESSES):
        _press(emulator, buttons.KEY_DOWN, _MOVE_PRESS_FRAMES, _MOVE_REST_FRAMES)
        ps = reader.player_state()
        if ps is not None and (ps.map_group, ps.map_num) == LITTLEROOT:
            return True
    return False


def _drain_mom_event(emulator: Any, reader: Any) -> bool:
    """A/B cycles until the shoes land AND town_state reaches 4 (bounded).

    The shoes flag flips before the event script finishes; town_state 3 -> 4 marks
    real completion, so both are required (anti-false-lock, same as the cutscene)."""

    def _done() -> bool:
        ps = reader.player_state()
        return (
            reader.has_running_shoes()
            and ps is not None
            and ps.town_state == _TOWN_STATE_SHOES_DONE
        )

    # Check-first loop with a final re-check: presses are bounded at exactly
    # _SHOES_MAX_CYCLES cycles, and a state completed BY the last cycle's presses
    # is still detected by the trailing _done().
    for _ in range(_SHOES_MAX_CYCLES):
        if _done():
            return True
        for _ in range(_SHOES_A_PER_CYCLE):
            _press(emulator, buttons.KEY_A, _BUTTON_FRAMES, _BUTTON_FRAMES)
        _press(emulator, buttons.KEY_B, _BUTTON_FRAMES, _BUTTON_FRAMES)
    return _done()


def _read_player_state(emulator: Any, reader: Any) -> Any:
    """Retry a None player_state read (save-block relocation window) a few
    frames later instead of letting callers burn their bounded budgets."""
    for _ in range(_STATE_READ_RETRIES):
        state = reader.player_state()
        if state is not None:
            return state
        emulator.step(0, _STATE_READ_RETRY_FRAMES)
    return reader.player_state()


def _verify_control(emulator: Any, reader: Any) -> bool:
    """Prove control returned: a DOWN press changes the position or map (bounded).

    A still-open dialogue box swallows direction input, so each failed press is
    followed by B presses to drain it before retrying (probe P6 pattern)."""
    for _ in range(_CONTROL_MAX_CYCLES):
        before = _read_player_state(emulator, reader)
        _press(emulator, buttons.KEY_DOWN, _MOVE_PRESS_FRAMES, _MOVE_REST_FRAMES)
        after = _read_player_state(emulator, reader)
        if (
            before is not None and after is not None
            and ((before.x, before.y) != (after.x, after.y)
                 or (before.map_group, before.map_num) != (after.map_group, after.map_num))
        ):
            return True
        for _ in range(_CONTROL_B_PRESSES):
            _press(emulator, buttons.KEY_B, _BUTTON_FRAMES, _BUTTON_FRAMES)
    return False


def run_shoes_leg(
    emulator: Any,
    reader: Any,
    memory: Any,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Drive a healthy post-Pokédex lab state through the mom/running-shoes event.

    Exits the lab (bounded DOWN walk), then walks north via hop_via_explore whose
    result is DELIBERATELY ignored: the scripted mom event intercepts any northbound
    nav in Littleroot, so 'no_portal' IS the expected success path — the
    shoes/town_state predicate is the arbiter, not the hop status. Bounded A/B
    cycles drain the event, then a DOWN press proves control returned. The reader
    is the same composite contract as run_pokedex_return (WorldReader snapshot/grid
    attributes for the hop, EmeraldReader flags/vars for the predicates).

    Returns 'shoes_delivered' | 'lab_exit_timeout' | 'shoes_timeout' |
    'control_timeout'.
    """
    if not _exit_lab(emulator, reader):
        return "lab_exit_timeout"
    emulator.step(0, _SETTLE_FRAMES)
    # Result intentionally unchecked — see docstring; a timeout in the drain below
    # surfaces honestly if the event unexpectedly never fires.
    hop_via_explore(
        emulator, reader, memory, LITTLEROOT, ROUTE_101, "up",
        move_type_fn=move_type_fn, predict=predict,
    )
    if not _drain_mom_event(emulator, reader):
        return "shoes_timeout"
    if not _verify_control(emulator, reader):
        return "control_timeout"
    return "shoes_delivered"


def run_pokedex_return(
    emulator: Any,
    reader: Any,
    memory: Any,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Drive post_rival.state back to Birch's lab and deliver the Pokédex.

    hop_via_explore hops route_103 -> Oldale (explore sweep + portal cross, landing the
    Flora-reachable Oldale tile); cross_scripted_npc plays Flora's gate into route_101;
    reach_map descends route_101 -> Littleroot -> lab; then _finish_lab_cutscene plays
    the lab OnFrame GivePokedex cutscene to completion (dex + 5 balls + VAR_BIRCH_LAB_STATE==5)
    and drains the re-opened Birch dialogue with B, so the driver returns WITH control.
    Every leg is bounded and surfaces the first failure.

    Returns 'pokedex_delivered' on success, or on failure the first failing leg's status:
    hop_via_explore / reach_map outcomes ('stall' | 'no_portal' | 'timeout' | 'battle_lost'
    | 'battle_timeout' | 'battle_interrupted' | 'unreachable') propagate verbatim; the Flora
    leg surfaces 'flora_<sub-step>' ('flora_off_map' | 'flora_no_grid' | 'flora_walk_failed'
    | 'flora_push_timeout' — review I10) and the cutscene leg 'pokedex_not_delivered'.
    """
    hopped = hop_via_explore(
        emulator, reader, memory, ROUTE_103, OLDALE, "down",
        move_type_fn=move_type_fn, predict=predict,
    )
    if hopped != "arrived":
        return hopped

    emulator.step(0, _SETTLE_FRAMES)
    crossed = cross_scripted_npc(
        emulator, reader, memory, OLDALE,
        stand_tile=_FLORA_STAND, face_dir=_FLORA_FACE,
        cross_dir=_FLORA_CROSS, max_presses=_FLORA_MAX_PRESSES,
    )
    if crossed != "crossed":
        return "flora_" + crossed

    descended = reach_map(
        emulator, reader, memory, LAB, _RETURN_DIRECTIONS,
        move_type_fn=move_type_fn, predict=predict,
    )
    if descended != "arrived":
        return descended

    if not _finish_lab_cutscene(emulator, reader):
        return "pokedex_not_delivered"
    return "pokedex_delivered"
