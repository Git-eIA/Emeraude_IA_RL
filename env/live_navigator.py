"""live_navigator: drive the real emulator to a target cell on the current map.

Read where the player is (P1 WorldReader), plan a path over walls learned so far
(P2 plan_path), press a d-pad key, classify the result (moved / blocked /
transition), and record any wall it bumps so the next plan routes around it
(replan-on-bump). No training, no reward. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from emulator import buttons
from env.battle_player import play_battle
from env.encounter_detector import EncounterWatcher
from env.heal_detector import HealWatcher
from env.local_navigator import WallMap, plan_path, resolve_move
from env.map_memory import MapMemory, WorldEvent

_DIRECTION_KEYS: dict[str, int] = {
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
}

STEP_FRAMES = 24      # hold a d-pad key ~0.4 s: one walking tile (matches env FRAMES_PER_ACTION)
RELEASE_FRAMES = 8    # idle after each press so the GBA doesn't fuse consecutive presses
TURN_RETRIES = 2      # a first press may only turn the character; retry to tell turn from wall
SETTLE_TRIES = 4      # re-read snapshot this many times to skip SaveBlock None frames


def _handle_battle_interruption(
    emulator: Any, reader: Any, move_type_fn: Any, predict: Any
) -> str | None:
    """If a wild battle is in progress, hand it to the Fighter and report.

    Returns None when there is no battle (or the battle was won) so the caller
    resumes navigating; returns a terminal outcome when navigation must abort:
    "battle_interrupted" (no Fighter supplied), "battle_lost", "battle_timeout".
    """
    if not reader.in_battle():
        return None
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result == "won":
        return None
    return "battle_lost" if result == "lost" else "battle_timeout"


def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
    memory: MapMemory | None = None,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Walk the player to `target` on its current map.

    When `memory` is given, a map transition is recorded as a portal
    (from_cell + direction + landed-on map), a heal observed en route
    (party HP refilled to full) tags the current place as a healing spot,
    and a battle starting here tags the cell as has_grass.
    NOTE: with `memory` set, `reader` must expose `party_hp()` and `in_battle()`
    (WorldReader does).
    Returns 'arrived' | 'unreachable' | 'left_map' | 'timeout' |
    'battle_lost' | 'battle_timeout' | 'battle_interrupted'.
    """
    heal_watcher = HealWatcher()
    enc_watcher = EncounterWatcher()
    for _ in range(max_steps):
        before = reader.snapshot()
        if before is None:
            emulator.step(0, RELEASE_FRAMES)   # relocating; idle a beat and retry
            continue
        if memory is not None:
            if heal_watcher.observe(reader.party_hp()):
                memory.observe(before, WorldEvent(healed=True))
            if enc_watcher.observe(reader.in_battle()):
                memory.observe(before, WorldEvent(encounter_started=True))
        interruption = _handle_battle_interruption(
            emulator, reader, move_type_fn, predict
        )
        if interruption is not None:
            return interruption
        if before.pos == target:
            return "arrived"
        path = plan_path(wallmap, before.map_id, before.pos, target)
        if path is None:
            return "unreachable"
        direction = path[0]
        outcome = probe_step(emulator, reader, before, direction)
        if outcome == "transition":
            if memory is not None:
                landed = snapshot_settled(reader)
                if landed is not None:
                    # A live crossing is not step-back tested, so reversibility
                    # cannot be proven here: record the cautious default False.
                    memory.record_portal(
                        before.map_id, before.pos, direction, landed.map_id,
                        False, landed.pos,
                    )
            return "left_map"
        if outcome == "blocked":
            wallmap.block(before.map_id, before.pos, direction)
    return "timeout"


def probe_step(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    """Press `direction`, retrying so a first-press turn isn't mistaken for a wall."""
    outcome = "blocked"
    for _ in range(TURN_RETRIES):
        emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
        emulator.step(0, RELEASE_FRAMES)
        after = snapshot_settled(reader)
        if after is None:
            return "blocked"
        outcome = resolve_move(before, after)
        if outcome != "blocked":
            return outcome
    return outcome


def snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap
