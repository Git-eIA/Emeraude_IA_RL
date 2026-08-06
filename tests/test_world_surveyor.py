"""world_surveyor: chart a multi-map overworld over a fake WorldGrid (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DELTAS, OPPOSITE
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot
from env.world_surveyor import SurveyReport, survey_world


class _AllFreeGridReader:
    def __init__(self, w: int, h: int) -> None:
        self._w, self._h = w, h

    def grid(self):
        from env.map_grid_reader import TileKind
        return [[TileKind.FREE] * self._w for _ in range(self._h)]

_KEY_TO_DIR = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}


class WorldGrid:
    """Hidden per-map grids joined by borders; answers step + snapshot.

    walls: set[(map_id, cell, direction)] — a blocked directed move.
    borders: dict[(map_id, cell, direction)] -> (to_map, entry_cell).
    Movement inside a map is otherwise free.
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        walls: set[tuple[tuple[int, int], tuple[int, int], str]],
        borders: dict[
            tuple[tuple[int, int], tuple[int, int], str],
            tuple[tuple[int, int], tuple[int, int]],
        ],
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._walls = walls
        self._borders = borders

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # release / noop
        key = (self.map_id, self.pos, direction)
        if key in self._borders:
            self.map_id, self.pos = self._borders[key]
            return
        if key in self._walls:
            return  # wall: stay put
        dx, dy = DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        # Always full: watcher stays quiet, no healing behaviour change.
        return [(1, 1)]

    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        # Derive exact room dimensions from the boundary walls for this map.
        # Wall entries: (map_id, cell, direction). Max cell coords + 1 = size.
        cells = [cell for (mid, cell, _) in self._walls if mid == self.map_id]
        if cells:
            w = max(x for x, _ in cells) + 1
            h = max(y for _, y in cells) + 1
        else:
            w, h = 1, 1
        return _AllFreeGridReader(w, h)


def _sealed_room(
    map_id: tuple[int, int], width: int, height: int
) -> set[tuple[tuple[int, int], tuple[int, int], str]]:
    """Every outward boundary edge of a width x height room, keyed by map_id."""
    walls: set[tuple[tuple[int, int], tuple[int, int], str]] = set()
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if x == 0:
                walls.add((map_id, cell, "left"))
            if x == width - 1:
                walls.add((map_id, cell, "right"))
            if y == 0:
                walls.add((map_id, cell, "up"))
            if y == height - 1:
                walls.add((map_id, cell, "down"))
    return walls


def _reversible_border(
    a_map: tuple[int, int], a_cell: tuple[int, int], direction: str,
    b_map: tuple[int, int], b_cell: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int], str],
          tuple[tuple[int, int], tuple[int, int]]]:
    """A two-way door: a_map@a_cell --direction--> b_map@b_cell and back."""
    return {
        (a_map, a_cell, direction): (b_map, b_cell),
        (b_map, b_cell, OPPOSITE[direction]): (a_map, a_cell),
    }


class _BlindReader:
    """A reader that never settles — snapshot() always returns None."""

    def snapshot(self) -> None:
        return None

    def step(self, keys: int, frames: int) -> None:  # emulator half, unused
        return None

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(1, 1)


def test_no_start_returns_no_start_failure() -> None:
    reader = _BlindReader()
    report = survey_world(reader, reader, MapMemory(), max_maps=10)

    assert report.surveyed == ()
    assert len(report.failed) == 1
    assert report.failed[0][1] == "no_start"


def test_two_maps_linked_by_reversible_border() -> None:
    a, b = (0, 0), (0, 1)
    # Each map is a fully-sealed 2x1 room. The door edges live in `borders`,
    # and WorldGrid.step checks borders BEFORE walls, so a door tile transitions
    # even though it is also a boundary wall — no need to punch holes in `walls`
    # (punching a hole on a NON-door edge would open an infinite void).
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1)
    borders = _reversible_border(a, (1, 0), "right", b, (0, 0))
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), max_maps=10)

    assert isinstance(report, SurveyReport)
    assert set(report.surveyed) == {a, b}
    assert report.failed == ()


def _one_way_border(
    a_map: tuple[int, int], a_cell: tuple[int, int], direction: str,
    b_map: tuple[int, int], b_cell: tuple[int, int],
) -> dict[tuple[tuple[int, int], tuple[int, int], str],
          tuple[tuple[int, int], tuple[int, int]]]:
    """A one-way building warp: a_map@a_cell --direction--> b_map@b_cell, no return."""
    return {(a_map, a_cell, direction): (b_map, b_cell)}


def test_chain_of_three_maps_surveyed_in_bfs_order() -> None:
    a, b, c = (0, 0), (0, 1), (0, 2)
    # Fully-sealed rooms; doors live in `borders` (checked before walls).
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1) | _sealed_room(c, 2, 1)
    borders = {
        **_reversible_border(a, (1, 0), "right", b, (0, 0)),
        **_reversible_border(b, (1, 0), "right", c, (0, 0)),
    }
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), max_maps=10)

    assert report.surveyed == (a, b, c)
    assert report.failed == ()


def test_warp_only_map_is_logged_left_map_and_target_never_enqueued() -> None:
    # Start map A's ONLY exit is a one-way building warp to H (down from (0,0)).
    # map_map(A) probes the warp -> "left_map"; A is still counted surveyed and
    # logged as map:left_map. H is a non-reversible target: never enqueued.
    a, h = (0, 0), (5, 5)
    walls = _sealed_room(a, 1, 1)          # single-cell room, fully sealed
    borders = _one_way_border(a, (0, 0), "down", h, (0, 0))  # warp wins over the wall
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), max_maps=10)

    assert report.surveyed == (a,)                 # A surveyed despite the early exit
    assert (a, "map:left_map") in report.failed    # map-side failure logged
    assert h not in report.surveyed                # warp target never surveyed
    assert all(m != h for (m, _r) in report.failed)  # nor even attempted


def test_max_maps_stops_cleanly_with_partial_report() -> None:
    a, b, c = (0, 0), (0, 1), (0, 2)
    walls = _sealed_room(a, 2, 1) | _sealed_room(b, 2, 1) | _sealed_room(c, 2, 1)
    borders = {
        **_reversible_border(a, (1, 0), "right", b, (0, 0)),
        **_reversible_border(b, (1, 0), "right", c, (0, 0)),
    }
    world = WorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders=borders)

    report = survey_world(world, world, MapMemory(), max_maps=1)

    assert report.surveyed == (a,)        # exactly one map surveyed then stopped


def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class BattleWorldGrid(WorldGrid):
    """WorldGrid pre-armed in a wild battle at the start cell. With a Fighter it
    plays out via play_battle (win clears in_battle so the survey resumes);
    without one, map_map returns "battle_interrupted" and the sweep aborts.
    Serves battle-reader bytes like BattleNavWorld in test_live_navigator."""

    _RESOLVE_PRESSES = 2

    def __init__(self, can_win: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._can_win = can_win
        self._battle = True
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        if self._battle:
            self._battle_step(keys)
            return
        super().step(keys, frames)

    def _battle_step(self, keys: int) -> None:
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            if not self._can_win:
                self._outcome = 2
                self._battle = False
                return
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
                self._battle = False
            self._phase = "resolving"
            self._resolve_left = self._RESOLVE_PRESSES
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def in_battle(self) -> bool:
        return self._battle

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
        for base, hp, mx in ((pbase, self._my_hp, 19), (obase, self._opp_hp, 18)):
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


def test_map_battle_without_a_fighter_aborts_the_sweep() -> None:
    a = (0, 0)
    walls = _sealed_room(a, 1, 1)  # single-cell sealed room
    world = BattleWorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders={})

    report = survey_world(world, world, MapMemory(), max_maps=10)

    # No Fighter: map_map(a) returns battle_interrupted -> sweep aborts before
    # a is counted surveyed, logging the leg.
    assert report.surveyed == ()
    assert report.failed == ((a, "map:battle_interrupted"),)


def test_fighter_win_lets_the_sweep_complete_the_map() -> None:
    a = (0, 0)
    walls = _sealed_room(a, 1, 1)
    world = BattleWorldGrid(start_map=a, start_cell=(0, 0), walls=walls, borders={})

    report = survey_world(
        world, world, MapMemory(), max_maps=10,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )

    # Fighter deps reach map_map: the battle is won, in_battle clears, and the
    # sealed single-cell map completes normally.
    assert report.surveyed == (a,)
    assert report.failed == ()
    assert not world._battle
