"""Pin: env.maps is the single source of the shared map-ID constants (I7)."""
from __future__ import annotations

from env import campaign, maps, milestones


def test_campaign_reexports_maps_constants():
    assert campaign.LITTLEROOT is maps.LITTLEROOT
    assert campaign.ROUTE_101 is maps.ROUTE_101
    assert campaign.OLDALE is maps.OLDALE
    assert campaign.ROUTE_103 is maps.ROUTE_103
    assert campaign.LAB is maps.LAB


def test_milestones_reexports_maps_constants():
    assert milestones.LITTLEROOT is maps.LITTLEROOT
    assert milestones.ROUTE_101 is maps.ROUTE_101
    assert milestones.OLDALE is maps.OLDALE
    assert milestones.ROUTE_103 is maps.ROUTE_103


def test_map_ids_match_pret_map_groups():
    assert maps.LITTLEROOT == (0, 9)
    assert maps.ROUTE_101 == (0, 16)
    assert maps.OLDALE == (0, 10)
    assert maps.ROUTE_103 == (0, 18)
    assert maps.LAB == (1, 4)
