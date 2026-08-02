"""Discover the lab -> Littleroot -> route_101 layout for the capture driver.

Loads states/lab_entry.state (from tools/capture_lab_entry.py) and:
  Phase A: press A (hold=6, release=10) + test DOWN (hold=24, release=8) every 200
           presses; log when the test move changes pos (= script gate cleared).
           NOTE: in_battle stays True as a permanent false-positive after
           starter_obtained (known SaveBlock quirk); only pos change is reliable.
  Phase B: once in control, walk DOWN until map_id changes (lab exit warp), then
           press RIGHT 3 times to reach x=10 (avoids lab-door re-entry warp at
           x=7), then walk UP until map_id becomes route_101 (0,16).

Discovered constants (from this probe, states/lab_entry.state):
  LAB_MAP               = (1, 4)
  GATE_CLEARED          : yes, at A-press 600 (16-frame A cycle)
  LAB_EXIT_CELL         = (6, 12)   exit direction = down
  Mexit                 = (0, 9)    (Littleroot town), landing pos (7, 16)
  LITTLEROOT_NORTH_CELL = (10, 1)   -> UP enters route_101 (0, 16) at (10, 19)

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/probe_lab_intro.py
"""
from __future__ import annotations

import os
from pathlib import Path

from emulator import buttons
from emulator.gba import GbaEmulator
from env.live_navigator import snapshot_settled
from env.orders import DESTINATIONS
from env.world_reader import WorldReader

ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16)
STATE = Path("states/lab_entry.state")

_DIRS = {"up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
         "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT}


def _press(emu, key, hold=24, release=8):
    emu.step(key, hold)
    emu.step(0, release)


def _log(reader, tag):
    snap = snapshot_settled(reader)
    where = None if snap is None else (snap.map_id, snap.pos)
    # NOTE: in_battle is a known false-positive in this state; logged for record only
    print(f"{tag}: {where} in_battle={reader.in_battle()}", flush=True)
    return snap


def _walk_until_map_change(emu, reader, direction, tag, max_steps=25):
    """Press `direction` until map_id changes; log each pos/map change."""
    start = snapshot_settled(reader)
    last = None if start is None else (start.map_id, start.pos)
    before_change_pos = start.pos if start is not None else None
    for i in range(max_steps):
        _press(emu, _DIRS[direction])
        snap = snapshot_settled(reader)
        cur = None if snap is None else (snap.map_id, snap.pos)
        if cur != last:
            print(f"{tag} step {i} {direction}: {cur}", flush=True)
            last = cur
        if snap is not None and start is not None and snap.map_id != start.map_id:
            print(f"{tag}: MAP CHANGE to {snap.map_id} at pos {snap.pos}", flush=True)
            return snap
        if snap is not None and snap.map_id == start.map_id:
            before_change_pos = snap.pos
    print(f"{tag}: no map change after {max_steps} {direction} presses "
          f"(last pos={before_change_pos})", flush=True)
    return snapshot_settled(reader)


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    _log(reader, "loaded")

    # Phase A: clear auto-dialogue via A-spam; detect control via test DOWN every 200.
    # in_battle stays True throughout — only pos change is reliable.
    cleared = False
    for i in range(2000):
        _press(emu, buttons.KEY_A, hold=6, release=10)
        if (i + 1) % 200 == 0:
            before = snapshot_settled(reader)
            _press(emu, buttons.KEY_DOWN)
            after = snapshot_settled(reader)
            if before is not None and after is not None and after.pos != before.pos:
                print(f"GATE CLEARED at A-press {i + 1}: moved {before.pos} -> {after.pos}",
                      flush=True)
                cleared = True
                break
    if not cleared:
        print("gate NOT cleared by A-spam+test-move; inspect the log", flush=True)

    _log(reader, "post-gate")

    # Phase B1: exit the lab going south.
    # Track the last lab pos before the warp fires — that is LAB_EXIT_CELL.
    s = snapshot_settled(reader)
    lab_map = s.map_id if s is not None else None
    lab_exit_cell = s.pos if s is not None else None
    print(f"Phase B1 start: map={lab_map} pos={lab_exit_cell}", flush=True)
    landed = _walk_until_map_change(emu, reader, "down", "EXIT-LAB", max_steps=50)

    if landed is None or landed.map_id == lab_map:
        print("BLOCKED: could not exit lab; aborting Phase B2", flush=True)
        return

    print(f"Littleroot landing: {landed.map_id} {landed.pos}", flush=True)

    # Phase B2: press RIGHT 3 times to reach x=10 column, then walk UP to route_101.
    # x=7 is the lab door warp column — walking UP from there re-enters the lab.
    # 3 RIGHT presses from (7,16) land at (10,17) which clears the warp column.
    print("Phase B2: pressing RIGHT 3 times to reach x=10 column", flush=True)
    for _ in range(3):
        _press(emu, buttons.KEY_RIGHT)
    s = snapshot_settled(reader)
    print(f"After right moves: {s.map_id if s else None} {s.pos if s else None}", flush=True)

    # Walk UP to route_101, logging every map/pos change.
    print("Phase B2: walking UP toward route_101", flush=True)
    _walk_until_map_change(emu, reader, "up", "TO-ROUTE101", max_steps=40)


if __name__ == "__main__":
    main()
