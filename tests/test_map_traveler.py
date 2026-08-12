"""map_traveler: door-to-door inter-map travel over a fake multi-map world (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env import map_traveler
from env.grid_navigator import DIRECTIONS
from env.map_grid_reader import TileKind
from env.map_memory import MapMemory, WorldEvent
from env.map_traveler import _cross_border, _cross_in_direction, _cross_up_warp, travel_to
from env.world_reader import WorldSnapshot

_KEY_TO_DIR: dict[int, str] = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}
_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class _AllFreeGridReader:
    """Returns an all-FREE grid of given dimensions; satisfies navigate_grid."""

    def __init__(self, w: int, h: int) -> None:
        self._w, self._h = w, h

    def grid(self):
        from env.map_grid_reader import TileKind
        return [[TileKind.FREE] * self._w for _ in range(self._h)]


class MultiMapWorld:
    """Several hidden grids joined by border crossings, for the traveler.

    Acts as emulator (`step`) and reader (`snapshot`).
    `borders` maps (map_id, cell, direction) -> (next_map, entry_cell): pressing
    `direction` on `cell` of `map_id` drops the player onto `entry_cell` of
    `next_map`. `walls` blocks a (map_id, cell, direction) move. Movement inside a
    map is otherwise free.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        borders: dict[tuple[tuple[int, int], tuple[int, int], str],
                       tuple[tuple[int, int], tuple[int, int]]] | None = None,
        walls: set[tuple[tuple[int, int], tuple[int, int], str]] | None = None,
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._borders = dict(borders or {})
        self._walls = set(walls or ())

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        border = self._borders.get((self.map_id, self.pos, direction))
        if border is not None:
            self.map_id, self.pos = border
            return
        if (self.map_id, self.pos, direction) in self._walls:
            return  # wall: no move
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot | None:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        # Always full: watcher stays quiet, no healing behaviour change.
        return [(1, 1)]

    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False

    def battle_starting(self) -> bool:
        return False

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        # navigate_grid uses a 50x50 all-FREE grid for path planning.
        return _AllFreeGridReader(50, 50)


def test_same_map_delegates_to_navigate() -> None:
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, MapMemory(),
        goal_map=(0, 0), goal_cell=(2, 0),
    )
    assert result == "arrived"
    assert world.pos == (2, 0)


def test_single_hop_crosses_one_known_door() -> None:
    # Map A=(0,0): door at (2,0) pressing 'right' lands on map B=(0,1) cell (0,0).
    borders = {((0, 0), (2, 0), "right"): ((0, 1), (0, 0))}
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1), reversible=True, to_cell=(0, 0))
    result = travel_to(
        world, world, memory,
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "arrived"
    assert world.map_id == (0, 1)
    assert world.pos == (1, 0)


def test_three_map_chain() -> None:
    # A=(0,0) --right@(2,0)--> B=(0,1) --right@(2,0)--> C=(0,2)
    borders = {
        ((0, 0), (2, 0), "right"): ((0, 1), (0, 0)),
        ((0, 1), (2, 0), "right"): ((0, 2), (0, 0)),
    }
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1), reversible=True, to_cell=(0, 0))
    memory.record_portal((0, 1), (2, 0), "right", (0, 2), reversible=True, to_cell=(0, 0))
    result = travel_to(
        world, world, memory,
        goal_map=(0, 2), goal_cell=(1, 0),
    )
    assert result == "arrived"
    assert world.map_id == (0, 2)
    assert world.pos == (1, 0)


def test_unknown_route_when_portal_missing() -> None:
    # Edge A->B exists (from observe) but no portal was ever recorded.
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 0), (0, 0), None), WorldEvent())
    memory.observe(WorldSnapshot((0, 1), (0, 0), None), WorldEvent())
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, memory,
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unknown_route"


def test_unknown_route_when_goal_never_visited() -> None:
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0))
    result = travel_to(
        world, world, MapMemory(),
        goal_map=(9, 9), goal_cell=(0, 0),
    )
    assert result == "unknown_route"


def test_unreachable_when_door_cell_is_walled_off() -> None:
    # Start is sealed on all sides: the door cell can never be reached.
    walls = {((0, 0), (0, 0), d) for d in DIRECTIONS}
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1), reversible=True, to_cell=(0, 0))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), walls=walls)
    result = travel_to(
        world, world, memory,
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unreachable"


def test_lost_when_crossing_lands_on_unexpected_map() -> None:
    # Portal claims the door leads to B=(0,1), but the world sends us to C=(0,5).
    borders = {((0, 0), (2, 0), "right"): ((0, 5), (0, 0))}
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1), reversible=True, to_cell=(0, 0))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), borders=borders)
    result = travel_to(
        world, world, memory,
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "lost"


def test_unreachable_when_the_crossing_press_cannot_reach_the_far_side() -> None:
    # The door cell (2,0) is reachable, but the portal claims it crosses 'right'
    # while the far-side neighbour (3,0) is sealed off and no border fires there.
    # The crossing leg must report "unreachable", not "lost".
    walls = {
        ((0, 0), (2, 0), "right"),
        ((0, 0), (3, 1), "up"),
        ((0, 0), (4, 0), "left"),
        ((0, 0), (3, -1), "down"),
    }
    memory = MapMemory()
    memory.record_portal((0, 0), (2, 0), "right", (0, 1), reversible=True, to_cell=(0, 0))
    world = MultiMapWorld(start_map=(0, 0), start_cell=(0, 0), walls=walls)
    result = travel_to(
        world, world, memory,
        goal_map=(0, 1), goal_cell=(1, 0),
    )
    assert result == "unreachable"


def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class LostBattleWorld:
    """Same-map world where walking treads grass at grass_at; the Fighter loses.

    Acts as emulator (step + read_bytes) and reader (snapshot). Movement is free
    until the battle starts; the battle is a terminal loss (outcome 2).
    """

    def __init__(self, map_id: tuple[int, int], start: tuple[int, int],
                 grass_at: tuple[int, int]) -> None:
        self.map_id = map_id
        self.pos = start
        self._grass_at = grass_at
        self._battle = False
        self._fought = False
        self._outcome = 0
        self._phase = "menu"

    def step(self, keys: int, frames: int) -> None:
        if self._battle:
            if keys == 0:
                return
            if self._phase == "menu" and keys & buttons.KEY_A:
                self._phase = "moves"
            elif self._phase == "moves" and keys & buttons.KEY_A:
                self._outcome = 2   # terminal loss
                self._battle = False
            return
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.pos == self._grass_at and not self._fought:
            self._battle = True
            self._fought = True

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

    def in_battle(self) -> bool:
        return self._battle

    def battle_starting(self) -> bool:
        return self._battle

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        # navigate_grid uses a 50x50 all-FREE grid for path planning.
        return _AllFreeGridReader(50, 50)

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            ACTION_MENU_VALUE,
            BATTLE_MON_SIZE,
            GBATTLE_ACTION_MENU_ADDR,
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_ACTION_MENU_ADDR:
            return bytes([ACTION_MENU_VALUE if self._phase == "menu" else 0])
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16b(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16b(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        for base, hp, mx in ((pbase, 19, 19), (obase, 18, 18)):
            if base <= addr < base + BATTLE_MON_SIZE:
                buf = bytearray(BATTLE_MON_SIZE)
                buf[0x00:0x02] = _u16b(1)
                buf[0x0C:0x0E] = _u16b(1)
                buf[0x24] = 10
                buf[0x21], buf[0x22] = 12, 12
                buf[0x28:0x2A] = _u16b(hp)
                buf[0x2A] = 5
                buf[0x2C:0x2E] = _u16b(mx)
                off = addr - base
                return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def test_battle_loss_propagates_from_the_first_hop() -> None:
    # Same-map goal: travel_to delegates to navigate_to, which treads grass at
    # (1,0), loses the battle, and the loss propagates as battle_lost (not "lost").
    world = LostBattleWorld(map_id=(0, 0), start=(0, 0), grass_at=(1, 0))
    result = travel_to(
        world, world, MapMemory(),
        goal_map=(0, 0), goal_cell=(2, 0),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "battle_lost"


# ---------------------------------------------------------------------------
# Task 2: crossing helpers
# ---------------------------------------------------------------------------


class _OneMapWorld:
    """Single-map fake that can be flipped to another map by a helper stub."""

    def __init__(self, map_id=(0, 16), pos=(10, 17)) -> None:
        self.map_id = map_id
        self.pos = pos
        self.grid_reader = object()   # GridSnapshot.from_reader is monkeypatched, so opaque

    def step(self, keys: int, frames: int) -> None:
        pass

    def snapshot(self):
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def in_battle(self) -> bool:
        return False

    def battle_starting(self) -> bool:
        return False


class _FakeSnap:
    """Fake GridSnapshot: FREE only at the given cells, WALL elsewhere."""

    def __init__(self, free, w=20, h=20) -> None:
        self._free = set(free)
        self.width, self.height = w, h

    def classify_at(self, x, y):
        return TileKind.FREE if (x, y) in self._free else TileKind.WALL


def test_cross_in_direction_dispatches_up_to_warp(monkeypatch):
    seen = []
    monkeypatch.setattr(map_traveler, "_cross_up_warp",
                        lambda *a, **k: seen.append("up") or "crossed")
    monkeypatch.setattr(map_traveler, "_cross_border",
                        lambda *a, **k: seen.append("border") or "crossed")
    world = _OneMapWorld()
    assert _cross_in_direction(world, world, MapMemory(), (0, 9), "up") == "crossed"
    assert seen == ["up"]


def test_cross_in_direction_dispatches_down_to_border(monkeypatch):
    seen = []
    monkeypatch.setattr(map_traveler, "_cross_up_warp",
                        lambda *a, **k: seen.append("up") or "crossed")
    monkeypatch.setattr(map_traveler, "_cross_border",
                        lambda *a, **k: seen.append("border") or "crossed")
    world = _OneMapWorld()
    assert _cross_in_direction(world, world, MapMemory(), (0, 16), "down") == "crossed"
    assert seen == ["border"]


def _stub_border_env(monkeypatch, world):
    """Wire the directional-scan boundary: opaque snap, reachable cells, no battle."""
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19), (10, 17)})))
    monkeypatch.setattr(map_traveler, "plan_path_grid", lambda snap, a, b: ["down"])
    monkeypatch.setattr(map_traveler, "handle_battle_interruption", lambda *a, **k: None)

    def fake_nav(emu, rdr, cell, **kw):
        world.pos = cell
        return "arrived"

    monkeypatch.setattr(map_traveler, "navigate_grid", fake_nav)


def test_cross_border_crosses_the_south_border_and_records(monkeypatch):
    world = _OneMapWorld(map_id=(0, 16), pos=(10, 17))
    memory = MapMemory()
    _stub_border_env(monkeypatch, world)

    def fake_probe(emu, rdr, before, direction):
        world.map_id = (0, 9)   # pressing DOWN on the border cell flips to Littleroot
        return "transition"

    monkeypatch.setattr(map_traveler, "probe_step", fake_probe)
    assert _cross_border(world, world, memory, (0, 16), "down", None, None) == "crossed"
    assert world.map_id == (0, 9)
    assert memory.portal((0, 16), (0, 9)) is not None


def test_cross_border_reports_no_crossing_when_border_never_flips(monkeypatch):
    world = _OneMapWorld(map_id=(0, 16), pos=(10, 17))
    _stub_border_env(monkeypatch, world)
    monkeypatch.setattr(map_traveler, "probe_step", lambda *a: "blocked")   # never crosses
    assert _cross_border(world, world, MapMemory(), (0, 16), "down", None, None) == "no_crossing"


def test_cross_border_passes_through_a_battle_outcome(monkeypatch):
    world = _OneMapWorld(map_id=(0, 16), pos=(10, 17))
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19)})))
    monkeypatch.setattr(map_traveler, "plan_path_grid", lambda snap, a, b: ["down"])
    monkeypatch.setattr(map_traveler, "handle_battle_interruption",
                        lambda *a, **k: "battle_lost")
    assert _cross_border(world, world, MapMemory(), (0, 16), "down", None, None) == "battle_lost"


def test_cross_up_warp_walks_onto_a_warp_tile_and_records(monkeypatch):
    world = _OneMapWorld(map_id=(0, 9), pos=(8, 18))
    memory = MapMemory()
    monkeypatch.setattr(map_traveler, "_warp_cells", lambda rdr, snap: [(8, 17)])
    monkeypatch.setattr(map_traveler, "GridSnapshot",
                        type("_GS", (), {"from_reader": staticmethod(lambda *a: object())}))

    def fake_walk(emu, rdr, snap, cell, from_map):
        world.pos = cell
        world.map_id = (1, 4)  # walking onto the warp tile triggers the transition
        return True

    monkeypatch.setattr(map_traveler, "_precision_walk_to", fake_walk)
    assert _cross_up_warp(world, world, memory, (0, 9), None, None) == "crossed"
    assert memory.portal((0, 9), (1, 4)) is not None


def test_cross_up_warp_reports_no_crossing_without_a_warp_tile(monkeypatch):
    world = _OneMapWorld(map_id=(0, 9))
    monkeypatch.setattr(map_traveler, "_warp_cells", lambda rdr, snap: [])
    monkeypatch.setattr(map_traveler, "GridSnapshot",
                        type("_GS", (), {"from_reader": staticmethod(lambda *a: object())}))
    assert _cross_up_warp(world, world, MapMemory(), (0, 9), None, None) == "no_crossing"
