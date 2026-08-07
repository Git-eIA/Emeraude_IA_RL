# tests/test_grid_snapshot.py
from __future__ import annotations

from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind


def _grid_3x2() -> GridSnapshot:
    # tiles[y][x]; width 3, height 2
    tiles = (
        (TileKind.FREE, TileKind.WALL, TileKind.GRASS),
        (TileKind.LEDGE_DOWN, TileKind.FREE, TileKind.FREE),
    )
    return GridSnapshot(map_id=(0, 16), width=3, height=2, tiles=tiles)


def test_classify_at_returns_the_tile_at_xy():
    g = _grid_3x2()
    assert g.classify_at(0, 0) is TileKind.FREE
    assert g.classify_at(1, 0) is TileKind.WALL
    assert g.classify_at(2, 0) is TileKind.GRASS
    assert g.classify_at(0, 1) is TileKind.LEDGE_DOWN


def test_classify_at_out_of_bounds_is_wall():
    g = _grid_3x2()
    assert g.classify_at(-1, 0) is TileKind.WALL
    assert g.classify_at(3, 0) is TileKind.WALL
    assert g.classify_at(0, -1) is TileKind.WALL
    assert g.classify_at(0, 2) is TileKind.WALL


def test_from_reader_captures_the_reader_grid():
    class FakeGridReader:
        def grid(self):
            return [
                [TileKind.FREE, TileKind.WALL],
                [TileKind.GRASS, TileKind.FREE],
            ]

    snap = GridSnapshot.from_reader(FakeGridReader(), map_id=(0, 16))
    assert snap is not None
    assert snap.map_id == (0, 16)
    assert snap.width == 2
    assert snap.height == 2
    assert snap.classify_at(1, 0) is TileKind.WALL
    assert isinstance(snap.tiles, tuple)
    assert isinstance(snap.tiles[0], tuple)


def test_from_reader_returns_none_when_map_not_ready():
    class NotReadyReader:
        def grid(self):
            return None

    assert GridSnapshot.from_reader(NotReadyReader(), map_id=(0, 16)) is None
