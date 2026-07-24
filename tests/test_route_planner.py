"""plan_route: BFS shortest path over the MapMemory graph (no ROM, no emulator)."""
from __future__ import annotations

from env.map_memory import MapMemory, WorldEvent
from env.route_planner import plan_route
from env.world_reader import WorldSnapshot


def _snap(map_id: tuple[int, int]) -> WorldSnapshot:
    return WorldSnapshot(map_id=map_id, pos=(0, 0), tile_behavior=None)


def _walk(memory: MapMemory, *map_ids: tuple[int, int]) -> None:
    """Feed a sequence of maps so MapMemory records the walked edges."""
    for map_id in map_ids:
        memory.observe(_snap(map_id), WorldEvent())


def test_start_equals_goal_returns_single_map() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9))
    assert plan_route(memory, (0, 9), (0, 9)) == [(0, 9)]


def test_direct_edge() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16))  # (0,9) -> (0,16)
    assert plan_route(memory, (0, 9), (0, 16)) == [(0, 9), (0, 16)]


def test_multi_hop_shortest_path() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16), (0, 17), (0, 18))
    assert plan_route(memory, (0, 9), (0, 18)) == [(0, 9), (0, 16), (0, 17), (0, 18)]


def test_none_when_goal_unknown() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16))
    assert plan_route(memory, (0, 9), (5, 5)) is None


def test_none_when_disconnected() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16))   # component A
    memory._prev_map_id = None       # break the chain so no edge (0,16)->(1,1)
    _walk(memory, (1, 1), (1, 2))    # component B, unreachable from (0,9)
    assert plan_route(memory, (0, 9), (1, 2)) is None


def test_bfs_prefers_fewer_hops() -> None:
    memory = MapMemory()
    # long way: A -> B -> C -> D
    _walk(memory, (0, 0), (0, 1), (0, 2), (0, 3))
    # shortcut: A -> D
    memory._prev_map_id = (0, 0)
    _walk(memory, (0, 3))
    assert plan_route(memory, (0, 0), (0, 3)) == [(0, 0), (0, 3)]
