"""GridSnapshot: an immutable classified-grid value object.

Captured from a MapGridReader once per navigation step and consumed by the pure
planner (plan_path_grid) and stored by MapMemory. Standalone module on purpose:
both the navigator and map_memory depend on it, so keeping it here avoids a
grid_navigator <-> map_memory import cycle. Emerald (BPEF) only.
"""
from __future__ import annotations

from dataclasses import dataclass

from env.map_grid_reader import TileKind


@dataclass(frozen=True)
class GridSnapshot:
    map_id: tuple[int, int]
    width: int
    height: int
    tiles: tuple[tuple[TileKind, ...], ...]  # [y][x], WALL-pinned, never None

    def classify_at(self, x: int, y: int) -> TileKind:
        """Bounds-checked tile lookup; out-of-range returns WALL."""
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.tiles[y][x]
        return TileKind.WALL

    @classmethod
    def from_reader(cls, grid_reader, map_id: tuple[int, int]) -> "GridSnapshot | None":
        """Capture grid_reader.grid(); None if the map is not ready.

        map_id is a label the caller supplies; the grid reader always decodes the
        currently-loaded map and does not verify the two agree — the caller reads
        pos and map_id from the same WorldSnapshot.
        """
        rows = grid_reader.grid()
        if rows is None:
            return None
        tiles = tuple(tuple(row) for row in rows)
        height = len(tiles)
        width = len(tiles[0]) if height else 0
        return cls(map_id=map_id, width=width, height=height, tiles=tiles)
