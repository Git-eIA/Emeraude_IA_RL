"""orders: the shared Order language + execute_order (pure, no ROM)."""
from __future__ import annotations

import dataclasses

from emulator import buttons
from env.map_memory import MapMemory, WorldEvent
from env.orders import DESTINATIONS, Order, _advance_story_dialogue, execute_order
from env.world_reader import WorldSnapshot


class _AllFreeGridReader:
    """Returns an all-FREE grid of given dimensions; satisfies navigate_grid."""

    def __init__(self, w: int, h: int) -> None:
        self._w, self._h = w, h

    def grid(self):
        from env.map_grid_reader import TileKind
        return [[TileKind.FREE] * self._w for _ in range(self._h)]


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
    result = execute_order(order, None, None, MapMemory())
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

    def battle_starting(self) -> bool:
        return False

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def test_advance_to_same_map_destination_arrives() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="littleroot", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory())
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
    result = execute_order(order, world, world, memory)
    assert result == "arrived"
    assert world.map_id == (0, 16)
    assert world.pos == (5, 12)


def test_advance_passes_through_unknown_route() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory())
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

    def battle_starting(self) -> bool:
        return False

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def test_heal_without_known_spot_returns_no_healing_spot_known() -> None:
    world = HealWorld((0, 9), (3, 10))
    order = Order(destination="littleroot", mode="heal", combat="win")
    result = execute_order(order, world, world, MapMemory())
    assert result == "no_healing_spot_known"


def test_heal_on_current_map_travels_and_heals() -> None:
    world = HealWorld((0, 9), (3, 10), a_presses_to_full=2)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    order = Order(destination="littleroot", mode="heal", combat="win")
    result = execute_order(order, world, world, memory)
    assert result == "healed"


def test_heal_that_never_refills_returns_heal_failed() -> None:
    world = HealWorld((0, 9), (3, 10), a_presses_to_full=10_000)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    order = Order(destination="littleroot", mode="heal", combat="win")
    result = execute_order(order, world, world, memory)
    assert result == "heal_failed"


def test_heal_ignores_the_order_destination() -> None:
    # The Strategist gives a pure "heal" intention; destination is not a real place.
    world = HealWorld((0, 9), (3, 10), a_presses_to_full=2)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 9), (3, 10), None), WorldEvent(healed=True))
    order = Order(destination="not_a_registered_place", mode="heal", combat="win")
    result = execute_order(order, world, world, memory)
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

    def battle_starting(self) -> bool:
        return self._steps >= self._to_enc

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def test_grind_without_known_grass_returns_no_grass_spot_known() -> None:
    world = GrassWorld((0, 16), (5, 12))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, MapMemory())
    assert result == "no_grass_spot_known"


def test_grind_on_known_grass_starts_an_encounter() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory)
    assert result == "encounter_started"


def test_grind_that_never_battles_returns_no_encounter() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=10_000)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory)
    assert result == "no_encounter"


def test_grind_passes_through_travel_failure() -> None:
    # Grass is remembered on a map with no known route from here -> unknown_route.
    world = GrassWorld((0, 9), (3, 10))
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 99), (1, 1), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory)
    assert result == "unknown_route"


def test_grind_ignores_the_order_destination() -> None:
    world = GrassWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="not_a_registered_place", mode="grind", combat="win")
    result = execute_order(order, world, world, memory)
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

    def battle_starting(self) -> bool:
        return self._battle

    @property
    def grid_reader(self) -> _AllFreeGridReader:
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
        order, world, world, memory,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "won"


def test_grind_without_fighter_deps_still_returns_encounter_started() -> None:
    world = GrassBattleWorld((0, 16), (5, 12), steps_to_encounter=3)
    memory = MapMemory()
    memory.observe(WorldSnapshot((0, 16), (5, 12), None), WorldEvent(encounter_started=True))
    order = Order(destination="route_101", mode="grind", combat="win")
    result = execute_order(order, world, world, memory)
    assert result == "encounter_started"


# ---------------------------------------------------------------------------
# Level-up loop tests
# ---------------------------------------------------------------------------


class FarmWorld:
    """Full grind-loop fake: treads -> wins a scripted 1-turn battle -> levels up
    and takes damage; A-presses out of battle (the Center) refill the party.

    Snapshots at a fixed cell that is BOTH the grass spot and the healing spot,
    so travel_to always arrives immediately. `party_levels()` reports the same
    level for every member, so the average equals `self._level`.
    """

    MAP = (0, 16)
    CELL = (5, 12)

    def __init__(
        self,
        start_level: int,
        target_hp_after: list[tuple[int, int]],
        levels_per_win: int = 1,
        steps_to_encounter: int = 3,
        party_size: int = 1,
        can_win: bool = True,
    ) -> None:
        self.map_id = self.MAP
        self.pos = self.CELL
        self._level = start_level
        self._levels_per_win = levels_per_win
        self._to_enc = steps_to_encounter
        self._party_size = party_size
        self._can_win = can_win
        self._hp_after = list(target_hp_after)
        self._hp: list[tuple[int, int]] = [(5, 5)] * party_size  # start full
        self.battles_won = 0
        self.heals = 0
        self._steps = 0
        self._battle = False
        self._phase = "menu"
        self._opp_hp = 6
        self._my_hp = 19
        self._outcome = 0

    def _start_battle(self) -> None:
        self._battle = True
        self._phase = "menu"
        self._opp_hp = 6
        self._outcome = 0
        self._steps = 0

    def _end_battle(self) -> None:
        if self._can_win:
            self._opp_hp = 0
            self._outcome = 1
            self._level += self._levels_per_win
            self.battles_won += 1
        else:
            self._outcome = 2   # terminal, not won -> play_battle returns "lost"
        self._battle = False
        self._hp = list(self._hp_after)

    def step(self, keys: int, frames: int) -> None:
        from emulator import buttons

        if not self._battle:
            if keys & buttons.KEY_A:                 # nurse dialog: refill to full
                self._hp = [(mx, mx) for _, mx in self._hp]
                self.heals += 1
                return
            if _KEY_TO_DIR.get(keys) is not None:    # treading
                self._steps += 1
                if self._steps >= self._to_enc:
                    self._start_battle()
            return
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            self._end_battle()

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_levels(self) -> list[int]:
        return [self._level] * self._party_size

    def party_hp(self) -> list[tuple[int, int]]:
        return list(self._hp)

    def in_battle(self) -> bool:
        return self._battle

    def battle_starting(self) -> bool:
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

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def _farm_memory(*, with_healing_spot: bool = True) -> MapMemory:
    memory = MapMemory()
    snap = WorldSnapshot(FarmWorld.MAP, FarmWorld.CELL, None)
    memory.observe(snap, WorldEvent(encounter_started=True))   # grass here
    if with_healing_spot:
        memory.observe(snap, WorldEvent(healed=True))          # Center here
    return memory


_FIGHTER = {"move_type_fn": (lambda mid: 12), "predict": (lambda obs: 0)}


def test_level_up_reaches_target_after_several_battles() -> None:
    world = FarmWorld(start_level=7, target_hp_after=[(5, 5)])  # full: never heals
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(),
        target_level=10, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.battles_won == 3   # 7 -> 8 -> 9 -> 10


def test_level_up_already_at_target_fights_nothing() -> None:
    world = FarmWorld(start_level=10, target_hp_after=[(5, 5)])
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(),
        target_level=10, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.battles_won == 0


def test_level_up_detours_to_heal_when_hp_is_low() -> None:
    # After each win the single member drops to 1/5 = 0.2 < 0.4 -> heal, then resume.
    world = FarmWorld(start_level=9, target_hp_after=[(1, 5)])
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(),
        target_level=10, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.battles_won == 1
    assert world.heals >= 1
    assert all(cur == mx for cur, mx in world.party_hp())   # healed to full


def test_level_up_heals_on_ko_even_when_totals_are_fine() -> None:
    # 5/10 total is above threshold, but a KO'd member forces the heal.
    world = FarmWorld(start_level=9, target_hp_after=[(0, 5), (5, 5)], party_size=2)
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(),
        target_level=10, heal_threshold=0.4, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.heals >= 1


def test_level_up_aborts_when_heal_needed_but_no_spot_known() -> None:
    world = FarmWorld(start_level=1, target_hp_after=[(1, 5)])
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(with_healing_spot=False),
        target_level=10, **_FIGHTER,
    )
    assert result == "no_healing_spot_known"


def test_level_up_exhausts_budget_without_reaching_target() -> None:
    world = FarmWorld(start_level=5, target_hp_after=[(5, 5)])  # full: never heals
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(),
        target_level=20, max_cycles=2, **_FIGHTER,
    )
    assert result == "grind_exhausted"
    assert world.battles_won == 2   # bounded by max_cycles


def test_level_up_aborts_on_a_lost_battle() -> None:
    world = FarmWorld(start_level=5, target_hp_after=[(5, 5)], can_win=False)
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(),
        target_level=10, **_FIGHTER,
    )
    assert result == "lost"


# ---------------------------------------------------------------------------
# Advance + Fighter (interruptible nav) tests
# ---------------------------------------------------------------------------


class AdvanceBattleWorld:
    """Walks toward the route_101 destination cell, treading grass on the way.

    Acts as emulator (step + read_bytes) and reader (snapshot). The Fighter (if
    wired) wins the battle and walking resumes to the goal cell.
    """

    _RESOLVE_PRESSES = 2

    def __init__(self, map_id: tuple[int, int], start: tuple[int, int],
                 grass_at: tuple[int, int]) -> None:
        self.map_id = map_id
        self.pos = start
        self._grass_at = grass_at
        self._battle = False
        self._fought = False
        self._opp_hp = 18
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        from emulator import buttons

        if self._battle:
            if keys == 0:
                return
            if self._phase == "menu" and keys & buttons.KEY_A:
                self._phase = "moves"
            elif self._phase == "moves" and keys & buttons.KEY_A:
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
        for base, hp, mx in ((pbase, 19, 19), (obase, self._opp_hp, 18)):
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

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def test_advance_wins_a_grass_battle_and_arrives() -> None:
    # route_101 destination is ((0,16),(5,12)); start one grass cell short.
    world = AdvanceBattleWorld(map_id=(0, 16), start=(3, 12), grass_at=(4, 12))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(
        order, world, world, MapMemory(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "arrived"
    assert world.pos == (5, 12)


def test_advance_without_fighter_reports_battle_interrupted() -> None:
    world = AdvanceBattleWorld(map_id=(0, 16), start=(3, 12), grass_at=(4, 12))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory())
    assert result == "battle_interrupted"


# ---------------------------------------------------------------------------
# battle_trainer mode tests
# ---------------------------------------------------------------------------


def test_route_103_is_a_known_destination() -> None:
    assert "route_103" in DESTINATIONS


def test_battle_trainer_wins_at_the_destination() -> None:
    map_id, cell = DESTINATIONS["route_103"]
    world = GrassBattleWorld(map_id, cell, steps_to_encounter=3)
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(
        order, world, world, MapMemory(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "won"


def test_battle_trainer_without_fighter_reports_encounter_started() -> None:
    map_id, cell = DESTINATIONS["route_103"]
    world = GrassBattleWorld(map_id, cell, steps_to_encounter=3)
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(order, world, world, MapMemory())
    assert result == "encounter_started"


def test_battle_trainer_without_a_trainer_returns_no_trainer() -> None:
    map_id, cell = DESTINATIONS["route_103"]
    world = GrassBattleWorld(map_id, cell, steps_to_encounter=10_000)
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(
        order, world, world, MapMemory(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "no_trainer"


def test_battle_trainer_unknown_destination() -> None:
    order = Order(destination="atlantide", mode="battle_trainer", combat="win")
    result = execute_order(order, None, None, MapMemory())
    assert result == "unknown_destination"


def test_battle_trainer_passes_through_travel_failure() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(order, world, world, MapMemory())
    assert result == "unknown_route"


# ---------------------------------------------------------------------------
# Story tests
# ---------------------------------------------------------------------------


class StoryWorld:
    """Fake emulator+reader on a single map: a scripted cutscene predicate flips
    true after N A-presses (models a lab/shoes A-spam until control/flag)."""

    def __init__(
        self,
        map_id: tuple[int, int],
        cell: tuple[int, int],
        a_presses_to_done: int = 2,
    ) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_done = a_presses_to_done
        self._a_count = 0

    def step(self, keys: int, frames: int) -> None:
        if keys & buttons.KEY_A:
            self._a_count += 1

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

    def in_battle(self) -> bool:
        return False

    def battle_starting(self) -> bool:
        return False

    def event_done(self) -> bool:
        return self._a_count >= self._to_done

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def test_story_unknown_destination_returns_unknown_destination() -> None:
    world = StoryWorld((1, 4), (6, 12))
    order = Order(destination="atlantide", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "unknown_destination"


def test_story_arrives_then_a_spams_until_target_holds() -> None:
    world = StoryWorld((1, 4), (6, 12), a_presses_to_done=2)
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "story_done"


def test_story_target_already_true_returns_immediately() -> None:
    world = StoryWorld((1, 4), (6, 12), a_presses_to_done=0)
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "story_done"


def test_story_that_never_completes_returns_story_timeout() -> None:
    world = StoryWorld((1, 4), (6, 12), a_presses_to_done=10_000)
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "story_timeout"


def test_story_passes_through_travel_failure() -> None:
    # route_101_shoes is on a different map with no seeded route -> unknown_route,
    # surfaced verbatim before any A-spam.
    world = StoryWorld((0, 9), (3, 10))
    order = Order(destination="route_101_shoes", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "unknown_route"


def test_story_without_target_returns_story_target_required() -> None:
    world = StoryWorld((1, 4), (6, 12))
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(order, world, world, MapMemory())
    assert result == "story_target_required"


def test_story_early_exit_skips_release_once_target_holds() -> None:
    # I11: the predicate is re-checked right after the A press; once it holds,
    # the release/settle step must not run (halves worst-case cutscene time).
    class _ReleaseCountingWorld(StoryWorld):
        def __init__(self) -> None:
            super().__init__((1, 4), (6, 12), a_presses_to_done=1)
            self.release_steps = 0

        def step(self, keys: int, frames: int) -> None:
            super().step(keys, frames)
            if keys == 0:
                self.release_steps += 1

    world = _ReleaseCountingWorld()
    result = _advance_story_dialogue(world, world, lambda r: r.event_done())
    assert result == "story_done"
    assert world.release_steps == 0
