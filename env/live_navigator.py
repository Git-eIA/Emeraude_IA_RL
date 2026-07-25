"""live_navigator: drive the real emulator to a target cell on the current map.

Read where the player is (P1 WorldReader), plan a path over walls learned so far
(P2 plan_path), press a d-pad key, classify the result (moved / blocked /
transition), and record any wall it bumps so the next plan routes around it
(replan-on-bump). No training, no reward. Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from emulator import buttons
from env.local_navigator import WallMap, plan_path, resolve_move

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


def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
) -> str:
    """Walk the player to `target` on its current map.

    Returns 'arrived' | 'unreachable' | 'left_map' | 'timeout'.
    """
    for _ in range(max_steps):
        before = reader.snapshot()
        if before is None:
            emulator.step(0, RELEASE_FRAMES)   # relocating; idle a beat and retry
            continue
        if before.pos == target:
            return "arrived"
        path = plan_path(wallmap, before.map_id, before.pos, target)
        if path is None:
            return "unreachable"
        direction = path[0]
        outcome = _press_until_moved(emulator, reader, before, direction)
        if outcome == "transition":
            return "left_map"
        if outcome == "blocked":
            wallmap.block(before.map_id, before.pos, direction)
    return "timeout"


def _press_until_moved(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    """Press `direction`, retrying so a first-press turn isn't mistaken for a wall."""
    outcome = "blocked"
    for _ in range(TURN_RETRIES):
        emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
        emulator.step(0, RELEASE_FRAMES)
        after = _snapshot_settled(reader)
        if after is None:
            return "blocked"
        outcome = resolve_move(before, after)
        if outcome != "blocked":
            return outcome
    return outcome


def _snapshot_settled(reader: Any) -> Any:
    """Read a snapshot, skipping up to SETTLE_TRIES None frames during relocation."""
    snap = None
    for _ in range(SETTLE_TRIES):
        snap = reader.snapshot()
        if snap is not None:
            return snap
    return snap
