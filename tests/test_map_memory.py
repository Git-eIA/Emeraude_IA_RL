"""MapMemory: self-built observational graph of discovered places."""
from __future__ import annotations

from env.map_memory import KNOWN_PLACES, MapMemory, PlaceNode, WorldEvent
from env.world_reader import WorldSnapshot


def _snap(map_id: tuple[int, int], pos: tuple[int, int] = (0, 0)) -> WorldSnapshot:
    return WorldSnapshot(map_id=map_id, pos=pos, tile_behavior=None)


def test_first_sight_creates_a_node() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent())
    node = mem.node((0, 16))
    assert isinstance(node, PlaceNode)
    assert node.map_id == (0, 16)
    assert node.labels == set()


def test_unknown_map_defaults_to_unknown_place_type() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent())
    assert mem.node((0, 16)).place_type == "unknown"


def test_catalogued_map_resolves_its_place_type() -> None:
    if not KNOWN_PLACES:
        return
    known_id, known_type = next(iter(KNOWN_PLACES.items()))
    mem = MapMemory()
    mem.observe(_snap(known_id), WorldEvent())
    assert mem.node(known_id).place_type == known_type


def test_revisiting_a_map_does_not_duplicate_the_node() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16), (1, 1)), WorldEvent())
    mem.observe(_snap((0, 16), (2, 2)), WorldEvent())
    assert mem.node((0, 16)) is not None
    assert len(mem.nodes) == 1


def test_node_returns_none_for_unseen_map() -> None:
    assert MapMemory().node((9, 9)) is None
