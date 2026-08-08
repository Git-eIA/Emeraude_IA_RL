from __future__ import annotations

import env.grid_navigator as gn
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


# --- handle_battle_interruption unit tests ---


class _ScriptedBattleReader:
    """Emulator+reader double scripting the intro/active/overworld sequence.

    battle_starting() is True from the start; in_battle() flips True only after
    `intro_steps` idle steps (models the opponent populating). A monkeypatched
    play_battle calls end() to clear both, so the fade-wait terminates.
    """

    def __init__(self, intro_steps: int = 2) -> None:
        self._intro_left = intro_steps
        self._battle = True
        self.steps = 0

    def step(self, _key: int, _frames: int) -> None:
        self.steps += 1
        if self._intro_left > 0:
            self._intro_left -= 1

    def battle_starting(self) -> bool:
        return self._battle

    def in_battle(self) -> bool:
        return self._battle and self._intro_left == 0

    def end(self) -> str:
        self._battle = False
        return "won"


def test_handle_waits_out_intro_then_wins(monkeypatch) -> None:
    r = _ScriptedBattleReader(intro_steps=2)
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: r.end())
    out = gn.handle_battle_interruption(r, r, move_type_fn=lambda m: 0, predict=lambda o: 0)
    assert out is None                       # won -> resume
    assert r.in_battle() is False            # overworld control on return
    assert r.battle_starting() is False


def test_handle_returns_none_when_no_battle() -> None:
    class _NoBattle:
        def battle_starting(self) -> bool:
            return False
        def in_battle(self) -> bool:
            return False
    r = _NoBattle()
    assert gn.handle_battle_interruption(r, r, None, None) is None


def test_handle_reports_interrupted_without_fighter() -> None:
    r = _ScriptedBattleReader(intro_steps=0)
    assert gn.handle_battle_interruption(r, r, None, None) == "battle_interrupted"


def test_handle_reports_loss(monkeypatch) -> None:
    r = _ScriptedBattleReader(intro_steps=0)
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: "lost")
    out = gn.handle_battle_interruption(r, r, move_type_fn=lambda m: 0, predict=lambda o: 0)
    assert out == "battle_lost"


def test_handle_reports_timeout(monkeypatch) -> None:
    r = _ScriptedBattleReader(intro_steps=0)
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: "battle_timeout")
    out = gn.handle_battle_interruption(r, r, move_type_fn=lambda m: 0, predict=lambda o: 0)
    assert out == "battle_timeout"


class _FadeBattleReader:
    """After the battle is won, battle_starting()/in_battle() stay True for a
    few fade steps, then clear — exercises handle's fade-wait loop body."""

    def __init__(self, fade_steps: int = 3) -> None:
        self._fade_left = fade_steps
        self._won = False
        self.steps = 0

    def step(self, _key: int, _frames: int) -> None:
        self.steps += 1
        if self._won and self._fade_left > 0:
            self._fade_left -= 1

    def battle_starting(self) -> bool:
        if not self._won:
            return True
        return self._fade_left > 0

    def in_battle(self) -> bool:
        if not self._won:
            return True
        return self._fade_left > 0

    def win(self) -> str:
        self._won = True
        return "won"


def test_handle_idles_through_end_fade(monkeypatch) -> None:
    r = _FadeBattleReader(fade_steps=3)
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: r.win())
    out = gn.handle_battle_interruption(r, r, move_type_fn=lambda m: 0, predict=lambda o: 0)
    assert out is None
    assert r.battle_starting() is False        # fade fully cleared before return
    assert r.in_battle() is False
    assert r.steps >= 3                         # the fade-wait loop idled


class _TrapGrassWorld:
    """3x1 vertical corridor: start (0,2) -> target (0,0), all FREE.

    Stepping from (0,1) to (0,0) is 'frozen' by a wild battle: the first press at
    (0,1) does not move and battle_starting is True. A monkeypatched play_battle
    calls clear() so the retry moves. Tracks pressed directions to prove no poison.
    """

    def __init__(self) -> None:
        self._pos = (0, 2)
        self._battle = False
        self._armed = True          # next northward press at (0,1) triggers a battle
        self._grid = _FakeGridReader([[_TK.FREE], [_TK.FREE], [_TK.FREE]])

    def snapshot(self):
        return WorldSnapshot(map_id=(0, 16), pos=self._pos, tile_behavior=0)

    def in_battle(self):
        return self._battle

    def battle_starting(self):
        return self._battle

    def party_hp(self):
        return [(20, 20)]

    @property
    def grid_reader(self):
        return self._grid

    def clear(self) -> str:
        self._battle = False
        return "won"

    def step(self, key, _frames):
        from emulator import buttons
        if key != buttons.KEY_UP:
            return
        if self._battle:
            return                              # frozen mid-battle
        if self._pos == (0, 1) and self._armed:
            self._battle = True                 # encounter fires, no move
            self._armed = False
            return
        self._pos = (self._pos[0], self._pos[1] - 1)


def test_navigate_grid_recovers_from_battle_frozen_press(monkeypatch) -> None:
    w = _TrapGrassWorld()
    monkeypatch.setattr(gn, "play_battle", lambda *a, **k: w.clear())
    result = gn.navigate_grid(
        w, w, target=(0, 0), max_steps=50,
        move_type_fn=lambda m: 0, predict=lambda o: 0,
    )
    assert result == "arrived"
    assert w._pos == (0, 0)


class _WalledWorld:
    """Target (0,0) is unreachable: (0,1)->(0,0) is a WALL, no battle ever."""

    def __init__(self) -> None:
        self._pos = (0, 2)
        self._grid = _FakeGridReader([[_TK.WALL], [_TK.FREE], [_TK.FREE]])

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

    def step(self, key, _frames):
        from emulator import buttons
        if key == buttons.KEY_UP and self._pos == (0, 2):
            self._pos = (0, 1)      # can reach (0,1); (0,0) is a wall


def test_navigate_grid_still_reports_genuine_wall() -> None:
    w = _WalledWorld()
    result = gn.navigate_grid(w, w, target=(0, 0), max_steps=30)
    assert result == "unreachable"
