"""live_navigator: control-loop tests over a fake grid world (no ROM)."""
from __future__ import annotations

from emulator import buttons
from env.live_navigator import navigate_to
from env.local_navigator import WallMap
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
