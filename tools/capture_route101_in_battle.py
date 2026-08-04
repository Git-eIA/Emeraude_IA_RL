"""Walk post_starter into route_101 grass until a wild battle fires, then cache
states/route101_in_battle.state.

Loads states/post_starter.state (a level-5 party free-roaming on route_101).
The entrance tile (10, 17) is bare path; the tall-grass band sits on the west
side (column x~2). So we first walk west to the wall, then sweep up/down across
the band with a release-frame debounce (grind's _walk_until_encounter pattern)
until reader.in_battle() flips true, then save. Cycling in place at the entrance
never reaches grass, and a fixed heading risks leaving the band before the
stochastic encounter roll fires.

One-shot scaffolding: run once locally where the ROM + post_starter.state exist.
The output feeds tests/test_battle_proof_survey_rom.py (deterministic mid-battle
smoke). Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_route101_in_battle.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from emulator.buttons import KEY_DOWN, KEY_LEFT, KEY_UP
from emulator.gba import GbaEmulator
from env.orders import GRIND_RELEASE_FRAMES, GRIND_STEP_FRAMES
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/route101_in_battle.state")
_WEST_MAX_PRESSES = 20  # LEFT presses to reach the far-west grass band
_SWEEP_HALF_PERIOD = 3  # steps per UP/DOWN half-cycle when sweeping the band


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="states/post_starter.state")
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    start = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [start], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()

    def press(key: int) -> bool:
        env.emulator.step(key, GRIND_STEP_FRAMES)
        env.emulator.step(0, GRIND_RELEASE_FRAMES)  # release (GBA debounce)
        return reader.in_battle()

    def _x() -> int | None:
        snap = reader.snapshot()
        return None if snap is None else snap.pos[0]

    # Walk west to the wall so the sweep happens inside the grass band.
    moved_west = False
    prev = _x()
    for _ in range(_WEST_MAX_PRESSES):
        press(KEY_LEFT)
        now = _x()
        if now is not None and prev is not None:
            if now < prev:
                moved_west = True
            elif moved_west:  # stopped after moving -> hit the west wall
                break
        prev = now

    # Sweep up/down across the band until the stochastic encounter fires.
    for i in range(args.max_steps):
        key = KEY_UP if (i // _SWEEP_HALF_PERIOD) % 2 == 0 else KEY_DOWN
        if press(key):
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            snap = reader.snapshot()
            print(
                f"IN-BATTLE state saved after {i} sweep steps "
                f"(map {None if snap is None else snap.map_id} "
                f"pos {None if snap is None else snap.pos}) -> {OUT_PATH.resolve()}",
                flush=True,
            )
            return

    print(f"no wild battle in {args.max_steps} sweep steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
