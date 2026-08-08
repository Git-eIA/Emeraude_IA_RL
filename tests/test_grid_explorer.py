from __future__ import annotations

from env.grid_explorer import explore_grid
from env.map_grid_reader import TileKind as TK
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot

F, W = TK.FREE, TK.WALL


class _FakeGridReader:
    def __init__(self, rows):
        self._rows = rows

    def grid(self):
        return [list(r) for r in self._rows]


class _ExploreWorld:
    """A tiny map with exactly one warp: stepping off the top edge from the
    single top-row FREE cell transitions to another map; every other border step
    is blocked. Records how many outward border steps were attempted so the test
    can assert there is no thrash (each candidate tested at most once).
    """

    def __init__(self, rows, start, warp_cell, warp_dir):
        self._rows = rows
        self._pos = start
        self._map = (0, 16)
        self._grid = _FakeGridReader(rows)
        self._warp_cell = warp_cell
        self._warp_dir = warp_dir
        self._returned = True
        self.border_attempts = 0

    def snapshot(self):
        return WorldSnapshot(map_id=self._map, pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return False

    def battle_starting(self):
        return False

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    def step(self, key, _frames):
        from emulator import buttons

        keymap = {
            buttons.KEY_UP: "up",
            buttons.KEY_DOWN: "down",
            buttons.KEY_LEFT: "left",
            buttons.KEY_RIGHT: "right",
        }
        d = keymap.get(key)
        if d is None:
            return
        delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        target = (self._pos[0] + delta[0], self._pos[1] + delta[1])
        on_map = 0 <= target[1] < len(self._rows) and 0 <= target[0] < len(self._rows[0])
        if on_map:
            if self._rows[target[1]][target[0]] is F:
                self._pos = target
            return
        # stepping off the edge:
        self.border_attempts += 1
        if self._map == (0, 16) and self._pos == self._warp_cell and d == self._warp_dir:
            self._map = (0, 17)              # warped to the neighbour map
            self._pos = (0, 0)
        # else: blocked, stay put (not a warp)

    # step-back after a transition returns us to the origin map:
    def _step_back_supported(self):
        return True


def test_explore_grid_records_a_border_portal_without_thrash():
    # 3-wide, 2-tall open map; the only warp is UP from (1,0).
    rows = [
        [F, F, F],
        [F, F, F],
    ]
    world = _ExploreWorld(rows, start=(1, 1), warp_cell=(1, 0), warp_dir="up")
    memory = MapMemory()
    result = explore_grid(world, world, memory, target_map=(0, 16))
    # a portal from (0,16) up to (0,17) was recorded
    assert memory.portal((0, 16), (0, 17)) is not None
    # the remembered grid is stored
    assert memory.grid_for((0, 16)) is not None
    # bounded, no infinite re-probing: far fewer than max_steps border attempts
    assert world.border_attempts < 50
    assert result in ("complete", "left_map")
