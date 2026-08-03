"""live_navigator: control-loop tests over a fake grid world (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.live_navigator import navigate_to
from env.local_navigator import WallMap
from env.map_memory import MapMemory, Portal
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


class FakeWorld:
    """A hidden grid the navigator must traverse blind.

    Acts as emulator (`step`) and reader (`snapshot`). `walls` is a set of
    (cell, direction) edges that block movement. `turn_first` models Emerald's
    turn-in-place: the first press in a new direction only rotates. `map_flips`
    are cells that, once entered, change map_id (a map transition). `none_frames`
    emits that many None snapshots first (SaveBlock relocation).
    """

    def __init__(
        self,
        start: tuple[int, int],
        walls: set[tuple[tuple[int, int], str]] | None = None,
        map_id: tuple[int, int] = (0, 0),
        turn_first: bool = False,
        map_flips: set[tuple[int, int]] | None = None,
        none_frames: int = 0,
    ) -> None:
        self.pos = start
        self.map_id = map_id
        self._walls = set(walls or ())
        self._turn_first = turn_first
        self._facing: str | None = None
        self._map_flips = set(map_flips or ())
        self._none_frames = none_frames
        self.presses = 0

    def step(self, keys: int, frames: int) -> None:
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return  # noop / release
        self.presses += 1
        if self._turn_first and self._facing != direction:
            self._facing = direction
            return  # first press only turns
        self._facing = direction
        if (self.pos, direction) in self._walls:
            return  # wall: no move
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.pos in self._map_flips:
            self.map_id = (self.map_id[0], self.map_id[1] + 1)

    def snapshot(self) -> WorldSnapshot | None:
        if self._none_frames > 0:
            self._none_frames -= 1
            return None
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        # Always full: watcher starts _was_full=True, stays quiet — no spurious heal.
        return [(1, 1)]

    def in_battle(self) -> bool:
        # No battle: EncounterWatcher stays quiet — no spurious grass learned.
        return False


class HealingFakeWorld(FakeWorld):
    """Extends FakeWorld with party_hp that refills to full upon reaching heal_at."""

    def __init__(self, heal_at: tuple[int, int], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._heal_at = heal_at

    def party_hp(self) -> list[tuple[int, int]]:
        # Hurt until the player stands on heal_at, then full.
        if self.pos == self._heal_at:
            return [(5, 5)]
        return [(2, 5)]


def test_walks_straight_corridor_to_target() -> None:
    world = FakeWorld(start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(3, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (3, 0)
    assert world.presses == 3


def test_times_out_when_budget_too_small() -> None:
    world = FakeWorld(start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(10, 0), max_steps=3)
    assert result == "timeout"
    assert world.pos != (10, 0)


def test_records_wall_and_reroutes() -> None:
    # Stepping right from (0,0) is walled; a down/right/up detour reaches (1,0).
    world = FakeWorld(start=(0, 0), walls={((0, 0), "right")})
    wallmap = WallMap()
    result = navigate_to(world, world, wallmap, target=(1, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (1, 0)
    assert wallmap.is_blocked((0, 0), (0, 0), "right")


def test_first_press_turns_without_recording_wall() -> None:
    # In Emerald the first press only rotates the character; it must not be read
    # as a wall.
    world = FakeWorld(start=(0, 0), turn_first=True)
    wallmap = WallMap()
    result = navigate_to(world, world, wallmap, target=(2, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (2, 0)
    assert not wallmap.is_blocked((0, 0), (0, 0), "right")


def test_unreachable_when_goal_is_sealed_off() -> None:
    walls = {((0, 0), d) for d in ("up", "down", "left", "right")}
    world = FakeWorld(start=(0, 0), walls=walls)
    result = navigate_to(world, world, WallMap(), target=(5, 5), max_steps=50)
    assert result == "unreachable"


def test_left_map_when_stepping_onto_a_transition_cell() -> None:
    world = FakeWorld(start=(0, 0), map_flips={(0, 1)})
    result = navigate_to(world, world, WallMap(), target=(0, 3), max_steps=50)
    assert result == "left_map"


def test_tolerates_none_snapshots_at_loop_top() -> None:
    world = FakeWorld(start=(0, 0), none_frames=2)
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50)
    assert result == "arrived"
    assert world.pos == (2, 0)


def test_left_map_records_portal_when_memory_given() -> None:
    # Stepping down from (0,0) onto the transition cell (0,1) crosses to a new map.
    world = FakeWorld(start=(0, 0), map_flips={(0, 1)})
    memory = MapMemory()
    result = navigate_to(
        world, world, WallMap(), target=(0, 3), max_steps=50, memory=memory
    )
    assert result == "left_map"
    assert memory.portal((0, 0), (0, 1)) == Portal(
        from_cell=(0, 0), direction="down", to_map=(0, 1),
        reversible=False, to_cell=(0, 1),
    )


def test_probe_step_and_snapshot_settled_are_public() -> None:
    """map_map imports these two by their public names; guard the rename."""
    from env.live_navigator import probe_step, snapshot_settled

    world = FakeWorld(start=(0, 0))  # open world, player at (0, 0)
    before = snapshot_settled(world)
    assert before is not None
    assert before.pos == (0, 0)

    outcome = probe_step(world, world, before, "right")
    assert outcome == "moved"
    assert snapshot_settled(world).pos == (1, 0)


def test_navigate_learns_healing_spot_on_hp_refill() -> None:
    # Straight corridor; HP refills when the player arrives at target (3, 0).
    # Party starts hurt (2/5), becomes full (5/5) only at heal_at=(3, 0).
    # The watcher must fire on that tick and the spot must be recorded.
    target = (3, 0)
    world = HealingFakeWorld(start=(0, 0), heal_at=target)
    memory = MapMemory()
    result = navigate_to(world, world, WallMap(), target=target, memory=memory)
    assert result == "arrived"
    assert memory.healing_spots() == [(world.map_id, target)]


def test_navigate_without_memory_ignores_hp() -> None:
    # memory=None: no party_hp calls should cause a crash; behaviour is unchanged.
    target = (3, 0)
    world = HealingFakeWorld(start=(0, 0), heal_at=target)
    result = navigate_to(world, world, WallMap(), target=target)
    assert result == "arrived"


class EncounterFakeWorld(FakeWorld):
    """Extends FakeWorld: a wild battle starts once the player reaches grass_at."""

    def __init__(self, grass_at: tuple[int, int], **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._grass_at = grass_at

    def in_battle(self) -> bool:
        return self.pos == self._grass_at


def test_no_fighter_learns_grass_then_returns_battle_interrupted() -> None:
    # Walking (0,0)->(2,0); a battle fires on (1,0). With no Fighter wired the
    # cell is still tagged has_grass (recording runs first), then navigate_to
    # aborts with battle_interrupted.
    world = EncounterFakeWorld(grass_at=(1, 0), start=(0, 0))
    memory = MapMemory()
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50, memory=memory)
    assert result == "battle_interrupted"
    assert memory.cells_labeled("has_grass") == [((0, 0), (1, 0))]


def test_no_fighter_returns_battle_interrupted_even_without_memory() -> None:
    # memory=None: the battle is still detected and aborts with battle_interrupted.
    world = EncounterFakeWorld(grass_at=(1, 0), start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50)
    assert result == "battle_interrupted"


def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class BattleNavWorld(FakeWorld):
    """FakeWorld that starts a wild battle on grass_at. The injected Fighter
    plays it out via play_battle; on a win the battle clears (in_battle drops to
    False) so walking resumes. can_win=False makes the Fighter lose.

    Serves the battle-reader bytes exactly like GrassBattleWorld in test_orders.
    """

    _RESOLVE_PRESSES = 2

    def __init__(self, grass_at: tuple[int, int], can_win: bool = True,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._grass_at = grass_at
        self._can_win = can_win
        self._battle = False
        self._fought = False
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
        if self.pos == self._grass_at and not self._fought:
            self._battle = True
            self._fought = True

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
                self._battle = False   # won: resume walking
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


def test_fighter_wins_the_interruption_and_navigation_resumes() -> None:
    # Battle fires on grass cell (1,0); the Fighter wins, walking resumes to (2,0).
    # The false-wall bug must NOT trigger: no wall is recorded and it arrives.
    world = BattleNavWorld(grass_at=(1, 0), start=(0, 0))
    memory = MapMemory()
    wallmap = WallMap()
    result = navigate_to(
        world, world, wallmap, target=(2, 0), max_steps=50, memory=memory,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "arrived"
    assert world.pos == (2, 0)
    assert memory.cells_labeled("has_grass") == [((0, 0), (1, 0))]
    assert not wallmap.is_blocked((0, 0), (1, 0), "right")


def test_fighter_loss_aborts_navigation() -> None:
    world = BattleNavWorld(grass_at=(1, 0), start=(0, 0), can_win=False)
    wallmap = WallMap()
    result = navigate_to(
        world, world, wallmap, target=(2, 0), max_steps=50,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "battle_lost"
    assert not wallmap.is_blocked((0, 0), (1, 0), "right")  # no false wall on abort


def test_handle_battle_interruption_is_public_and_quiet_off_battle() -> None:
    from env.live_navigator import handle_battle_interruption

    class _NoBattle:
        def in_battle(self) -> bool:
            return False

    # No battle -> None, and no Fighter needed to reach that branch.
    assert handle_battle_interruption(None, _NoBattle(), None, None) is None
