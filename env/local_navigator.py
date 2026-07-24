"""local_navigator: intra-map movement from observed collisions.

The Explorer learns which cell-to-cell moves are blocked by bumping into walls
(pressed a direction, position did not change -> wall). Those observations fill a
WallMap, and A* plans a path over the known-walkable grid, replanning whenever a
new wall is discovered. No tile-behavior probe, no emulator, no ROM. Emerald
(BPEF) only.
"""
from __future__ import annotations

from env.world_reader import WorldSnapshot

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# Grid convention: x grows right, y grows down. up decreases y.
DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def resolve_move(before: WorldSnapshot, after: WorldSnapshot) -> str:
    """Classify one attempted step: 'moved' | 'blocked' | 'transition'."""
    if before.map_id != after.map_id:
        return "transition"
    if before.pos != after.pos:
        return "moved"
    return "blocked"
