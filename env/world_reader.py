"""WorldReader: RAM-based perception snapshot for the Explorer.

Reads only where the player is (map id, position) and what tile it stands on.
No pixels. Isolated on purpose so the perception layer is a single swappable
unit. Emerald (BPEF) only — game-specific RAM is used freely.
"""
from __future__ import annotations

from dataclasses import dataclass

from env.game_state import EmeraldReader


@dataclass(frozen=True)
class WorldSnapshot:
    map_id: tuple[int, int]        # (map_group, map_num) — which map
    pos: tuple[int, int]           # (x, y) — position within the map
    tile_behavior: int | None      # current-tile behavior id; None until probed


class WorldReader:
    """Wraps EmeraldReader and returns an immutable WorldSnapshot each step."""

    def __init__(self, reader: EmeraldReader) -> None:
        self._reader = reader

    def snapshot(self) -> WorldSnapshot | None:
        """Snapshot the world, or None while the save blocks relocate."""
        ps = self._reader.player_state()
        if ps is None:
            return None
        return WorldSnapshot(
            map_id=(ps.map_group, ps.map_num),
            pos=(ps.x, ps.y),
            tile_behavior=self._tile_behavior(),
        )

    def _tile_behavior(self) -> int | None:
        # TODO(probe): read the metatile-behavior byte of the tile the player
        # stands on (tall grass, water, wall, door, ...). Its RAM address on
        # BPEF is not yet known; returns None until a probe session finds it.
        return None
