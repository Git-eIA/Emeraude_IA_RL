"""MapMemory: self-built observational graph of discovered places."""
from __future__ import annotations

import pytest

from env.map_memory import KNOWN_PLACES, MapMemory, PlaceNode, Portal, WorldEvent
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
        pytest.skip("KNOWN_PLACES is empty — add a catalog entry to enable this test")
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


def test_transition_between_maps_adds_a_directed_edge() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9)), WorldEvent())    # Littleroot
    mem.observe(_snap((0, 16)), WorldEvent())   # step onto Route 101
    assert ((0, 9), (0, 16)) in mem.edges()


def test_staying_on_the_same_map_adds_no_edge() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9), (1, 1)), WorldEvent())
    mem.observe(_snap((0, 9), (1, 2)), WorldEvent())
    assert mem.edges() == set()


def test_edges_are_directional() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9)), WorldEvent())
    mem.observe(_snap((0, 16)), WorldEvent())
    assert ((0, 16), (0, 9)) not in mem.edges()


def test_first_observation_adds_no_edge() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9)), WorldEvent())
    assert mem.edges() == set()


def test_heal_event_labels_the_current_place_as_healing_spot() -> None:
    mem = MapMemory()
    mem.observe(_snap((1, 5)), WorldEvent(healed=True))
    assert "healing_spot" in mem.node((1, 5)).labels


def test_encounter_event_labels_the_current_place_as_has_grass() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent(encounter_started=True))
    assert "has_grass" in mem.node((0, 16)).labels


def test_labels_are_additive_across_observations() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent(encounter_started=True))
    mem.observe(_snap((0, 16)), WorldEvent(healed=True))
    assert mem.node((0, 16)).labels == {"has_grass", "healing_spot"}


def test_no_event_adds_no_label() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent())
    assert mem.node((0, 16)).labels == set()


def test_record_and_read_portal() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    p = mem.portal((0, 9), (0, 16))
    assert p == Portal(
        from_cell=(5, 0), direction="up", to_map=(0, 16),
        reversible=True, to_cell=(5, 12),
    )


def test_portal_is_none_for_unrecorded_pair() -> None:
    assert MapMemory().portal((0, 9), (0, 16)) is None


def test_record_portal_also_creates_the_edge() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    assert ((0, 9), (0, 16)) in mem.edges()


def test_record_portal_last_write_wins() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    mem.record_portal((0, 9), (4, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    assert mem.portal((0, 9), (0, 16)) == Portal((4, 0), "up", (0, 16), True, (5, 12))


def test_record_portal_creates_both_nodes() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    assert mem.node((0, 9)) is not None
    assert mem.node((0, 16)) is not None


def test_record_portal_round_trips_reversible_and_to_cell() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    p = mem.portal((0, 9), (0, 16))
    assert p == Portal(
        from_cell=(5, 0), direction="up", to_map=(0, 16),
        reversible=True, to_cell=(5, 12),
    )


def test_outgoing_and_incoming_portals() -> None:
    mem = MapMemory()
    mem.record_portal((0, 9), (5, 0), "up", (0, 16), reversible=True, to_cell=(5, 12))
    mem.record_portal((0, 16), (5, 12), "down", (0, 9), reversible=True, to_cell=(5, 0))
    mem.record_portal((0, 16), (2, 2), "right", (1, 0), reversible=False, to_cell=(0, 0))
    out = mem.outgoing_portals((0, 16))
    assert {p.to_map for p in out} == {(0, 9), (1, 0)}
    inc = mem.incoming_portals((0, 16))
    assert [p.from_cell for p in inc] == [(5, 0)]
