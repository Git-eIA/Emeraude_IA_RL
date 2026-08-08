from __future__ import annotations

from env.grid_navigator import plan_path_grid
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.map_grid_reader import TileKind as _TK
from env.world_reader import WorldSnapshot

F = TileKind.FREE
W = TileKind.WALL
G = TileKind.GRASS
LU = TileKind.LEDGE_UP
LD = TileKind.LEDGE_DOWN


def _snap(rows: list[list[TileKind]]) -> GridSnapshot:
    tiles = tuple(tuple(r) for r in rows)
    return GridSnapshot(
        map_id=(0, 16), width=len(rows[0]), height=len(rows), tiles=tiles
    )


def test_straight_line_path():
    # 1x3 corridor; walk right from (0,0) to (2,0)
    snap = _snap([[F, F, F]])
    assert plan_path_grid(snap, (0, 0), (2, 0)) == ["right", "right"]


def test_routes_around_a_wall():
    # (1,0) is a wall; go down, right, up
    snap = _snap([
        [F, W, F],
        [F, F, F],
    ])
    path = plan_path_grid(snap, (0, 0), (2, 0))
    assert path is not None
    # ends on the goal
    assert _walk_on(snap, (0, 0), path) == (2, 0)
    # never steps onto the wall
    assert (1, 0) not in _cells_on(snap, (0, 0), path)


def test_one_way_ledge_descend_is_allowed():
    # standing at (0,0); (0,1) is LEDGE_DOWN; (0,2) is the FREE landing.
    # descending the ledge is a single directed jump edge (0,0)->(0,2).
    snap = _snap([
        [F],
        [LD],
        [F],
    ])
    assert plan_path_grid(snap, (0, 0), (0, 2)) == ["down"]


def test_one_way_ledge_climb_is_blocked():
    # same LEDGE_DOWN column, but now going UP from (0,2) to (0,0):
    # the ledge only accepts "down", so there is no path up.
    snap = _snap([
        [F],
        [LD],
        [F],
    ])
    assert plan_path_grid(snap, (0, 2), (0, 0)) is None


def test_routes_around_a_ledge_to_the_right_then_up():
    # A LEDGE_DOWN wall spans the middle row across x=0..1 (cannot climb it).
    # The right column x=2 is open FREE, so the way up is: right along the
    # bottom, up the open right column, then left along the top.
    snap = _snap([
        [F, F, F],
        [LD, LD, F],
        [F, F, F],
    ])
    path = plan_path_grid(snap, (0, 2), (0, 0))
    assert path is not None
    assert _walk_on(snap, (0, 2), path) == (0, 0)
    # the climb must use the open right column, never a ledge upward
    assert "up" in path


def test_unreachable_goal_returns_none():
    # goal (2,0) is walled off entirely
    snap = _snap([
        [F, W, F],
        [F, W, F],
    ])
    assert plan_path_grid(snap, (0, 0), (2, 0)) is None


def test_blocked_edge_forces_a_detour():
    # open 3x2 grid; block the direct (0,0)->right edge so A* detours down.
    snap = _snap([
        [F, F, F],
        [F, F, F],
    ])
    blocked = {((0, 0), "right")}
    path = plan_path_grid(snap, (0, 0), (1, 0), blocked=blocked)
    assert path is not None
    assert _walk_on(snap, (0, 0), path) == (1, 0)
    assert path[0] != "right"


# --- helpers: replay a direction list over the 2-tile jump model ---
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_LEDGE = {
    "up": TileKind.LEDGE_UP,
    "down": TileKind.LEDGE_DOWN,
    "left": TileKind.LEDGE_LEFT,
    "right": TileKind.LEDGE_RIGHT,
}


def _step(snap: GridSnapshot, cell, d):
    dx, dy = _DELTA[d]
    adj = (cell[0] + dx, cell[1] + dy)
    if snap.classify_at(*adj) is _LEDGE[d]:
        return (cell[0] + 2 * dx, cell[1] + 2 * dy)
    return adj


def _walk_on(snap, start, path):
    cell = start
    for d in path:
        cell = _step(snap, cell, d)
    return cell


def _cells_on(snap, start, path):
    cell = start
    out = [cell]
    for d in path:
        cell = _step(snap, cell, d)
        out.append(cell)
    return out


class _FakeGridReader:
    """Serves a fixed classified grid regardless of the loaded map."""

    def __init__(self, rows):
        self._rows = rows

    def grid(self):
        return [list(r) for r in self._rows]


class _LedgeWorld:
    """Emulator + reader double. The player walks a small grid; a LEDGE_DOWN at
    (0,1) may be descended (down from (0,0) lands (0,2)) but never climbed.

    Each 'down'/'up'/... press updates pos per the 2-tile jump model; a press
    into a WALL leaves pos unchanged (a 'blocked' outcome).
    """

    def __init__(self, rows, start):
        self._rows = rows
        self._pos = start
        self._grid = _FakeGridReader(rows)
        self._blocked_npc: set[tuple[tuple[int, int], str]] = set()

    # --- reader surface ---
    def snapshot(self):
        return WorldSnapshot(map_id=(0, 16), pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return False

    def battle_starting(self):
        return False

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    # --- emulator surface ---
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
        self._pos = self._resolve(self._pos, d)

    def _classify(self, x, y):
        if 0 <= y < len(self._rows) and 0 <= x < len(self._rows[0]):
            return self._rows[y][x]
        return _TK.WALL

    def _resolve(self, cell, d):
        if (cell, d) in self._blocked_npc:
            return cell  # a phantom NPC stands here: press does not move
        delta = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        ledge = {
            "up": _TK.LEDGE_UP,
            "down": _TK.LEDGE_DOWN,
            "left": _TK.LEDGE_LEFT,
            "right": _TK.LEDGE_RIGHT,
        }[d]
        adj = (cell[0] + delta[0], cell[1] + delta[1])
        kind = self._classify(*adj)
        if kind in (_TK.FREE, _TK.GRASS):
            return adj
        if kind is ledge:
            land = (cell[0] + 2 * delta[0], cell[1] + 2 * delta[1])
            if self._classify(*land) in (_TK.FREE, _TK.GRASS):
                return land
        return cell  # wall / wrong-arrow ledge: no move


def test_navigate_grid_descends_a_ledge_in_the_correct_direction():
    from env.grid_navigator import navigate_grid

    rows = [
        [_TK.FREE],
        [_TK.LEDGE_DOWN],
        [_TK.FREE],
    ]
    world = _LedgeWorld(rows, start=(0, 0))
    assert navigate_grid(world, world, target=(0, 2)) == "arrived"
    assert world._pos == (0, 2)


def test_navigate_grid_refuses_to_climb_a_one_way_ledge():
    from env.grid_navigator import navigate_grid

    rows = [
        [_TK.FREE],
        [_TK.LEDGE_DOWN],
        [_TK.FREE],
    ]
    world = _LedgeWorld(rows, start=(0, 2))
    # climbing back up is impossible: no path -> unreachable, no hang.
    assert navigate_grid(world, world, target=(0, 0)) == "unreachable"


def test_navigate_grid_detours_around_a_phantom_npc():
    from env.grid_navigator import navigate_grid

    # open 3x2 grid; an NPC blocks the direct (0,0)->right press. navigate_grid
    # must add that edge to its transient set and reroute down/right/up.
    rows = [
        [_TK.FREE, _TK.FREE, _TK.FREE],
        [_TK.FREE, _TK.FREE, _TK.FREE],
    ]
    world = _LedgeWorld(rows, start=(0, 0))
    world._blocked_npc.add(((0, 0), "right"))
    assert navigate_grid(world, world, target=(1, 0)) == "arrived"
    assert world._pos == (1, 0)


def test_navigate_grid_gives_up_on_a_blocked_off_map_border():
    from env.grid_navigator import navigate_grid

    # Target (2,0) is off the 2x1 grid and exactly one step right of the player.
    # The OOB fallback presses right, but the border does not cross (no map
    # change) so the press is 'blocked'. The edge lands in the transient set;
    # a second OOB attempt would repeat the dead press until max_steps, so the
    # navigator must report 'unreachable' immediately instead of hanging.
    rows = [[_TK.FREE, _TK.FREE]]
    world = _LedgeWorld(rows, start=(1, 0))
    assert navigate_grid(world, world, target=(2, 0), max_steps=50) == "unreachable"
    assert world._pos == (1, 0)
