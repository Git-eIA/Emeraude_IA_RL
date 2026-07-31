"""ROM-gated smoke test: navigate_to wired to the real mGBA emulator."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from emulator.gba import GbaEmulator
from env.live_navigator import navigate_to
from env.local_navigator import WallMap
from env.world_reader import WorldReader

ROM = os.environ.get("POKEMON_EMERALD_ROM")

# states/ lives in the main repo, not this worktree — resolve via the ROM path
# (both sit under the same main repo root) or fall back to a sibling path.
_MAIN_REPO = Path(__file__).resolve().parents[2] / "Emu"
_STATE_FILE = _MAIN_REPO / "states" / "initial.state"
if not _STATE_FILE.is_file():
    # When run from the main repo cwd (unlikely but possible)
    _STATE_FILE = Path("states/initial.state")


@pytest.mark.skipif(not ROM, reason="requires POKEMON_EMERALD_ROM")
def test_navigate_to_wires_to_real_emulator() -> None:
    emu = GbaEmulator(ROM)
    emu.load_state(_STATE_FILE.read_bytes())
    reader = WorldReader(emu.read_bytes)

    start = reader.snapshot()
    assert start is not None  # live position read works

    # Already on target -> arrives immediately.
    assert navigate_to(emu, reader, WallMap(), target=start.pos, max_steps=5) == "arrived"

    # Nearby same-map target: the loop runs live and returns a known outcome.
    target = (start.pos[0], start.pos[1] + 2)
    result = navigate_to(emu, reader, WallMap(), target=target, max_steps=40)
    assert result in {"arrived", "unreachable", "left_map", "timeout"}
