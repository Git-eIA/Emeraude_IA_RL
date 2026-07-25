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


def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
) -> str:
    """Walk the player to `target`. Returns 'arrived' or 'timeout' (for now)."""
    for _ in range(max_steps):
        before = reader.snapshot()
        if before.pos == target:
            return "arrived"
        direction = plan_path(wallmap, before.map_id, before.pos, target)[0]
        outcome = _press_until_moved(emulator, reader, before, direction)
        if outcome == "blocked":
            wallmap.block(before.map_id, before.pos, direction)
    return "timeout"


def _press_until_moved(emulator: Any, reader: Any, before: Any, direction: str) -> str:
    emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
    emulator.step(0, RELEASE_FRAMES)
    after = reader.snapshot()
    return resolve_move(before, after)
