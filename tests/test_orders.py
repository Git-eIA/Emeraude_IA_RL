"""orders: the shared Order language + execute_order (pure, no ROM)."""
from __future__ import annotations

import dataclasses

from emulator import buttons
from env.local_navigator import WallMap
from env.map_memory import MapMemory, WorldEvent
from env.orders import DESTINATIONS, Order, execute_order
from env.world_reader import WorldSnapshot


def test_order_is_a_frozen_dataclass_with_three_fields() -> None:
    order = Order(destination="route_101", mode="advance", combat="win")
    assert order.destination == "route_101"
    assert order.mode == "advance"
    assert order.combat == "win"
    try:
        order.mode = "grind"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Order must be frozen")


def test_destinations_registry_holds_known_places() -> None:
    assert DESTINATIONS["littleroot"] == ((0, 9), (3, 10))
    assert DESTINATIONS["route_101"] == ((0, 16), (5, 12))


def test_unknown_destination_returns_unknown_destination() -> None:
    order = Order(destination="atlantide", mode="advance", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "unknown_destination"



_KEY_TO_DIR: dict[int, str] = {
    buttons.KEY_UP: "up",
    buttons.KEY_DOWN: "down",
    buttons.KEY_LEFT: "left",
    buttons.KEY_RIGHT: "right",
}
_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class NamedWorld:
    """Hidden multi-map grid that plays emulator (step) and reader (snapshot).

    `borders` maps (map_id, cell, direction) -> (next_map, entry_cell). All
    other moves are free (no wall simulation needed for these tests).
    """

    def __init__(
        self,
        start_map: tuple[int, int],
        start_cell: tuple[int, int],
        borders: dict[tuple[tuple[int, int], tuple[int, int], str],
                      tuple[tuple[int, int], tuple[int, int]]] | None = None,
    ) -> None:
        self.map_id = start_map
        self.pos = start_cell
        self._borders = dict(borders or {})

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        border = self._borders.get((self.map_id, self.pos, direction))
        if border is not None:
            self.map_id, self.pos = border
            return
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        # Always full: watcher stays quiet, no healing behaviour change.
        return [(1, 1)]

    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False


def test_advance_to_same_map_destination_arrives() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="littleroot", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "arrived"
    assert world.map_id == (0, 9)
    assert world.pos == (3, 10)


def test_advance_across_one_known_door_arrives() -> None:
    borders = {((0, 9), (2, 10), "right"): ((0, 16), (0, 12))}
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10), borders=borders)
    memory = MapMemory()
    memory.record_portal(
        (0, 9), (2, 10), "right", (0, 16), reversible=True, to_cell=(0, 12)
    )
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "arrived"
    assert world.map_id == (0, 16)
    assert world.pos == (5, 12)


def test_advance_passes_through_unknown_route() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "unknown_route"


# ---------------------------------------------------------------------------
# Heal tests
# ---------------------------------------------------------------------------


class HealWorld:
    """Fake emulator+reader on a single map: party is hurt, refills after N A-presses."""

    def __init__(
        self,
        map_id: tuple[int, int],
        cell: tuple[int, int],
        a_presses_to_full: int = 2,
    ) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_full = a_presses_to_full
        self._a_count = 0

    def step(self, keys: int, frames: int) -> None:
        if keys & buttons.KEY_A:
            self._a_count += 1

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        full = self._a_count >= self._to_full
        return [(5, 5)] if full else [(2, 5)]

    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False


def test_heal_without_known_spot_returns_no_healing_spot_known() -> None:
    world = HealWorld((0, 9), (3, 10))
    order = Order(destination="littleroot", mode="heal", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "no_healing_spot_known"


def test_heal_on_current_map_travels_and_heals() -> None:
    world = HealWorld((0, 9), (3, 10), a_presses_to_full=2)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    order = Order(destination="littleroot", mode="heal", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "healed"


def test_heal_that_never_refills_returns_heal_failed() -> None:
    world = HealWorld((0, 9), (3, 10), a_presses_to_full=10_000)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    order = Order(destination="littleroot", mode="heal", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "heal_failed"


def test_heal_ignores_the_order_destination() -> None:
    # The Strategist gives a pure "heal" intention; destination is not a real place.
    world = HealWorld((0, 9), (3, 10), a_presses_to_full=2)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    order = Order(destination="not_a_registered_place", mode="heal", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "healed"


# ---------------------------------------------------------------------------
# Grind tests
# ---------------------------------------------------------------------------


class GrassWorld:
    """Single-map fake: treading triggers a wild battle after N steps."""

    def __init__(
        self,
        map_id: tuple[int, int],
        cell: tuple[int, int],
        steps_to_encounter: int = 3,
    ) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_enc = steps_to_encounter
        self._steps = 0

    def step(self, keys: int, frames: int) -> None:
        if _KEY_TO_DIR.get(keys) is not None:
            self._steps += 1  # count d-pad presses; releases (keys=0) do not count

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]  # full: heal watcher stays quiet

    def in_battle(self) -> bool:
        return self._steps >= self._to_enc


def test_grind_without_known_grass_returns_no_grass_spot_known() -> None:
    world = GrassWorld((0, 16), (5, 12))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "no_grass_spot_known"


def test_grind_on_known_grass_starts_an_encounter() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "encounter_started"


def test_grind_that_never_battles_returns_no_encounter() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=10_000)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "no_encounter"


def test_grind_passes_through_travel_failure() -> None:
    # Grass is remembered on a map with no known route from here -> unknown_route.
    world = GrassWorld((0, 9), (3, 10))
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 99), (1, 1), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "unknown_route"


def test_grind_ignores_the_order_destination() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="not_a_registered_place", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "encounter_started"


# ---------------------------------------------------------------------------
# Grind + Fighter hookup tests
# ---------------------------------------------------------------------------


def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class GrassBattleWorld:
    """Treads to a battle, then plays a scripted battle the Fighter wins in 3 turns."""

    _RESOLVE_PRESSES = 2

    def __init__(self, map_id: tuple[int, int], cell: tuple[int, int],
                 steps_to_encounter: int = 3) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_enc = steps_to_encounter
        self._steps = 0
        self._battle = False
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        from emulator import buttons

        if not self._battle:
            if _KEY_TO_DIR.get(keys) is not None:
                self._steps += 1
                if self._steps >= self._to_enc:
                    self._battle = True
            return
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
            self._phase = "resolving"
            self._resolve_left = self._RESOLVE_PRESSES
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

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


def test_grind_with_fighter_wins_the_battle() -> None:
    world = GrassBattleWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(
        order, world, world, memory, WallMap(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "won"


def test_grind_without_fighter_deps_still_returns_encounter_started() -> None:
    world = GrassBattleWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory, WallMap())
    assert result == "encounter_started"
