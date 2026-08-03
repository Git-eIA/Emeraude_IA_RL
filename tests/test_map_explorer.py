"""Unit tests for map_map with an ExploreWorld fake (plays emulator + reader).

ExploreWorld models one hidden grid keyed by map_id, with `walls` (blocked
directed edges the survey must learn by bumping) and optional `borders`
(reversible/non-reversible map crossings). It exposes both the emulator API
(`step(keys, frames)` decodes the d-pad bit and moves on the hidden grid) and
the reader API (`snapshot()` -> WorldSnapshot). No ROM, no emulator.
"""
from __future__ import annotations

from emulator import buttons
from env.local_navigator import DELTAS, OPPOSITE, WallMap
from env.map_explorer import map_map
from env.map_memory import MapMemory
from env.world_reader import WorldSnapshot

_KEY_TO_DIR = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}


class ExploreWorld:
    """Hidden grid that answers both emulator.step and reader.snapshot."""

    def __init__(
        self,
        map_id: tuple[int, int],
        start: tuple[int, int],
        walls: set[tuple[tuple[int, int], str]],
        borders: dict[
            tuple[tuple[int, int], tuple[int, int], str],
            tuple[tuple[int, int], tuple[int, int]],
        ]
        | None = None,
    ) -> None:
        self.map_id = map_id
        self.pos = start
        self._walls = walls
        self._borders = borders or {}
        self.presses = 0

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # release frame: no movement
        self.presses += 1
        key = (self.map_id, self.pos, direction)
        if key in self._borders:
            to_map, entry = self._borders[key]
            self.map_id = to_map
            self.pos = entry
            return
        if (self.pos, direction) in self._walls:
            return  # wall: stay put
        dx, dy = DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def in_battle(self) -> bool:
        return False


def _sealed_room_walls(
    width: int, height: int
) -> set[tuple[tuple[int, int], str]]:
    """All outward edges on the boundary of a width x height room at origin."""
    walls: set[tuple[tuple[int, int], str]] = set()
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if x == 0:
                walls.add((cell, "left"))
            if x == width - 1:
                walls.add((cell, "right"))
            if y == 0:
                walls.add((cell, "up"))
            if y == height - 1:
                walls.add((cell, "down"))
    return walls


def test_sealed_room_complete_survey():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)  # cells (0,0),(1,0),(0,1),(1,1)
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "complete"
    for (cell, direction) in walls:
        assert wallmap.is_blocked(target, cell, direction)
    assert memory.edges() == set()


def _reversible_border(
    from_map: tuple[int, int],
    from_cell: tuple[int, int],
    direction: str,
    to_map: tuple[int, int],
    entry: tuple[int, int],
) -> dict[
    tuple[tuple[int, int], tuple[int, int], str],
    tuple[tuple[int, int], tuple[int, int]],
]:
    """A two-way border: crossing `direction` lands on to_map@entry, and the
    opposite press from entry returns to from_map@from_cell."""
    return {
        (from_map, from_cell, direction): (to_map, entry),
        (to_map, entry, OPPOSITE[direction]): (from_map, from_cell),
    }


def test_reversible_door_recorded_and_survey_continues():
    target = (3, 3)
    other = (7, 7)
    # 2x1 room: cells (0,0),(1,0). Seal every boundary EXCEPT (1,0)->right,
    # which is a reversible door to `other`.
    walls = _sealed_room_walls(2, 1)
    walls.discard(((1, 0), "right"))
    borders = _reversible_border(target, (1, 0), "right", other, (0, 0))
    world = ExploreWorld(target, start=(0, 0), walls=walls, borders=borders)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "complete"
    portal = memory.portal(target, other)
    assert portal is not None
    assert portal.from_cell == (1, 0)
    assert portal.direction == "right"
    assert portal.to_map == other
    assert portal.reversible is True
    assert portal.to_cell == (0, 0)
    assert wallmap.is_blocked(target, (0, 0), "left")


def test_no_edge_is_reprobed():
    """After 'complete', every (cell, direction) was tried at most once, so the
    press count cannot exceed the total edge count of the reached region."""
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=500)

    assert result == "complete"
    # 4 cells x 4 directions = 16 directed edges is the hard ceiling on probes;
    # repositioning presses are bounded by the same walked region. A runaway
    # re-probe loop would blow past this immediately.
    assert world.presses <= 64


def test_non_reversible_door_ends_run_but_records_portal():
    target = (3, 3)
    other = (7, 7)
    walls = _sealed_room_walls(2, 1)
    walls.discard(((1, 0), "right"))
    # one-way warp: crossing right lands on `other`, but the opposite press
    # from the entry cell does NOT return (no reverse border) — a building warp.
    borders = {(target, (1, 0), "right"): (other, (0, 0))}
    world = ExploreWorld(target, start=(0, 0), walls=walls, borders=borders)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "left_map"
    portal = memory.portal(target, other)
    assert portal is not None
    assert portal.from_cell == (1, 0)
    assert portal.direction == "right"
    assert portal.reversible is False
    assert portal.to_cell == (0, 0)


def test_budget_exhausted_on_large_room_with_tiny_budget():
    target = (3, 3)
    walls = _sealed_room_walls(6, 6)  # 36 cells, far more than the budget
    world = ExploreWorld(target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(world, world, memory, wallmap, target, max_steps=5)

    assert result == "budget_exhausted"


class EncounterExploreWorld(ExploreWorld):
    """ExploreWorld that reports a wild battle while standing on grass_at."""

    def __init__(self, grass_at: tuple[int, int], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._grass_at = grass_at

    def in_battle(self) -> bool:
        return self.pos == self._grass_at


def test_map_map_learns_grass_then_aborts_without_a_fighter():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)  # cells (0,0),(1,0),(0,1),(1,1)
    world = EncounterExploreWorld(grass_at=(1, 0), map_id=target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    # No Fighter deps: stepping onto grass learns it, then aborts on that frame.
    result = map_map(world, world, memory, wallmap, target, max_steps=200)

    assert result == "battle_interrupted"
    assert ((3, 3), (1, 0)) in memory.cells_labeled("has_grass")


def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class BattleExploreWorld(ExploreWorld):
    """ExploreWorld pre-armed in a wild battle at the start cell. A supplied
    Fighter plays it via play_battle; on a win in_battle drops so the survey
    resumes. can_win=False makes the Fighter lose. Serves battle-reader bytes
    exactly like BattleNavWorld in test_live_navigator."""

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
                self._outcome = 2   # terminal loss -> play_battle returns "lost"
                self._battle = False
                return
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
                self._battle = False   # won: resume surveying
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


def test_map_map_fighter_wins_the_battle_and_survey_completes():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)
    world = BattleExploreWorld(map_id=target, start=(0, 0), walls=walls)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(
        world, world, memory, wallmap, target, max_steps=200,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )

    assert result == "complete"
    assert not world._battle  # battle was actually resolved before the survey finished


def test_map_map_fighter_loss_aborts_the_survey():
    target = (3, 3)
    walls = _sealed_room_walls(2, 2)
    world = BattleExploreWorld(map_id=target, start=(0, 0), walls=walls, can_win=False)
    memory = MapMemory()
    wallmap = WallMap()

    result = map_map(
        world, world, memory, wallmap, target, max_steps=200,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )

    assert result == "battle_lost"
