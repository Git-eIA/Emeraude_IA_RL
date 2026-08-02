"""Capture states/post_starter.state: level-5 party, free-roam on route_101.

Loads states/lab_entry.state (post forced-battle, in Birch's lab (1,4)) and
scripts the lab intro to route_101, using the layout discovered by
tools/probe_lab_intro.py:
  Phase 2: A-spam to clear the lab Pokedex/rival auto-dialogue (a test move that
           changes pos proves control returned), then walk DOWN out of the lab
           until the map becomes Littleroot (0,9).
  Phase 3: press RIGHT to clear the x=7 lab-door warp column (walking UP from x=7
           re-enters the lab), then walk UP until the map becomes route_101 (0,16),
           confirm free-roam with a real pos change, and save the artifact.

in_battle() is a permanent false-positive in this post-starter state (known
SaveBlock quirk); it is never gated on -- only pos/map_id changes are trusted.

Constants below are from tools/probe_lab_intro.py (Task 2).

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_route101_freeroam.py
"""
from __future__ import annotations

import os
from pathlib import Path

from emulator import buttons
from emulator.gba import GbaEmulator
from env.live_navigator import snapshot_settled
from env.orders import DESTINATIONS
from env.world_reader import WorldReader

# NOTE: all from tools/probe_lab_intro.py, over states/lab_entry.state.
LAB_MAP = (1, 4)                # confirmed lab map-id
LAB_EXIT_DIR = "down"          # direction that warps out of the lab
LITTLEROOT = (0, 9)            # map the lab exit lands on
LITTLEROOT_RIGHT_STEPS = 3     # RIGHT presses to clear the x=7 lab-door warp column
GATE_CLEAR_BUDGET = 2000       # A-presses; the lab intro dialogue is long (~600)
GATE_TEST_EVERY = 200          # test a move every N A-presses to detect regained control
ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16)

STATE = Path("states/lab_entry.state")
OUT_PATH = Path("states/post_starter.state")

_DIRS = {"up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
         "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT}


def _press(emu, key, hold=24, release=8):
    emu.step(key, hold)
    emu.step(0, release)


def _clear_gate(emu, reader):
    """A-spam the lab auto-dialogue; a test move that changes pos proves control."""
    for i in range(GATE_CLEAR_BUDGET):
        _press(emu, buttons.KEY_A, hold=6, release=10)
        if (i + 1) % GATE_TEST_EVERY == 0:
            before = snapshot_settled(reader)
            _press(emu, _DIRS[LAB_EXIT_DIR])   # test move also heads toward the exit
            after = snapshot_settled(reader)
            if before is not None and after is not None and after.pos != before.pos:
                return True
    return False


def _walk_until_map(emu, reader, direction, target_map, max_steps=50):
    """Press `direction` until snapshot().map_id == target_map."""
    for _ in range(max_steps):
        snap = snapshot_settled(reader)
        if snap is not None and snap.map_id == target_map:
            return True
        _press(emu, _DIRS[direction])
    snap = snapshot_settled(reader)
    return snap is not None and snap.map_id == target_map


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    if not _clear_gate(emu, reader):
        print("Phase 2: gate not cleared", flush=True)
        return

    # Leave the lab (the test move in _clear_gate already stepped toward the door).
    if not _walk_until_map(emu, reader, LAB_EXIT_DIR, LITTLEROOT):
        print("Phase 2: never exited the lab", flush=True)
        return

    # Phase 3: clear the x=7 lab-door warp column before heading north.
    for _ in range(LITTLEROOT_RIGHT_STEPS):
        _press(emu, buttons.KEY_RIGHT)

    if not _walk_until_map(emu, reader, "up", ROUTE_101):
        print("Phase 3: never entered route_101", flush=True)
        return

    # Confirm free-roam with a real pos change, then save.
    ref = snapshot_settled(reader)
    for _ in range(8):
        _press(emu, buttons.KEY_UP)
        snap = snapshot_settled(reader)
        if snap is not None and ref is not None and snap.map_id == ROUTE_101 \
                and snap.pos != ref.pos:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(emu.save_state())
            print(
                f"POST-STARTER saved: map {snap.map_id} pos {snap.pos} "
                f"levels {reader.party_levels()} -> {OUT_PATH.resolve()}",
                flush=True,
            )
            return
    print("Phase 3: no free-roam movement confirmed on route_101", flush=True)


if __name__ == "__main__":
    main()
