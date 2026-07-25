"""live_navigator: drive the real emulator to a target cell on the current map.

First live loop of the Explorer: read where the player is (P1 WorldReader), plan
a path over the walls learned so far (P2 plan_path), press a d-pad key, and
repeat. Later tasks add wall-learning, turn/wall disambiguation, and the
unreachable / left_map / None-tolerance branches. No training, no reward.
Emerald (BPEF) only.
"""
from __future__ import annotations

from typing import Any

from emulator import buttons
from env.local_navigator import WallMap, plan_path

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
        _press(emulator, direction)
    return "timeout"


def _press(emulator: Any, direction: str) -> None:
    emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
    emulator.step(0, RELEASE_FRAMES)
