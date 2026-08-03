"""Walk post_starter into route_101 grass until a wild battle fires, then cache
states/route101_in_battle.state.

Loads states/post_starter.state (a level-5 party free-roaming on route_101) and
cycles the four d-pad directions with a release-frame debounce (grind's
_walk_until_encounter pattern) until reader.in_battle() flips true, then saves.
A fixed heading would risk hitting a wall or leaving the grass patch before the
stochastic encounter roll fires, so we wander.

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

from emulator.buttons import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP
from emulator.gba import GbaEmulator
from env.orders import GRIND_RELEASE_FRAMES, GRIND_STEP_FRAMES
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/route101_in_battle.state")
_DIRECTIONS = (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT)


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

    for i in range(args.max_steps):
        env.emulator.step(_DIRECTIONS[i % len(_DIRECTIONS)], GRIND_STEP_FRAMES)
        env.emulator.step(0, GRIND_RELEASE_FRAMES)  # release (GBA debounce)
        if reader.in_battle():
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            snap = reader.snapshot()
            print(
                f"IN-BATTLE state saved after {i} steps "
                f"(map {None if snap is None else snap.map_id}) -> {OUT_PATH.resolve()}",
                flush=True,
            )
            return

    print(f"no wild battle in {args.max_steps} steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
