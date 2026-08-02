"""Gated ROM smoke: run_campaign skips level_up and drives a real advance.

Load-bearing when states/post_starter.state exists (captured by
tools/capture_post_starter.py): a level-5 party on the route_101 map lets
run_campaign skip level_up by construction and walk a real advance on the ROM.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROM = os.environ.get("POKEMON_EMERALD_ROM")
# states/ lives in the main repo, not the worktree; resolve it absolutely.
_STATE = Path(__file__).resolve().parents[2] / "Emu" / "states" / "post_starter.state"


@pytest.mark.skipif(not _ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _STATE.exists(), reason="states/post_starter.state missing")
def test_run_campaign_skips_level_up_and_advances_on_real_rom() -> None:
    from emulator.gba import GbaEmulator
    from env.campaign import Milestone, run_campaign
    from env.local_navigator import WallMap
    from env.map_memory import MapMemory
    from env.orders import DESTINATIONS, reached
    from env.world_reader import WorldReader

    emu = GbaEmulator(_ROM)
    emu.load_state(_STATE.read_bytes())
    emu.step(0, 4)  # settle after load_state
    reader = WorldReader(emu.read_bytes)

    start = reader.snapshot()
    assert start is not None
    # Same-map precondition: advance takes travel_to's same-map branch (no portals).
    assert start.map_id == DESTINATIONS["route_101"][0]
    # A >=5 party means run_campaign skips level_up by construction.
    assert reached(reader.party_levels(), 5)

    outcome = run_campaign(
        emu, reader, MapMemory(), WallMap(),
        curriculum=(Milestone("route_101", 5),),
    )

    assert outcome in {"campaign_complete", "unreachable", "left_map", "timeout"}
    assert outcome == "campaign_complete" or reader.snapshot().pos != start.pos
