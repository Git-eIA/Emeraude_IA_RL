"""Walk post_starter toward route_103 until the rival trainer battle fires, then
cache states/trainer_battle.state.

Unlike tools/capture_trainer_battle.py (which saves on ANY in_battle), this
verifies the battle is on the target map (route_103) before saving, so a wild
battle in route_101 grass on the way does not get captured by mistake. If a
battle fires on a non-target map, the character is stuck in it and the tool
aborts with guidance (start from a state already past the grass).

The captured states/trainer_battle.state is exactly the artifact that Brique 3
part 1's gated smoke (tests/test_battle_player_rom.py::
test_fighter_wins_a_real_trainer_battle) needs — so running this makes THAT
smoke load-bearing. This run also reveals the real route_103 map_id, approach
cell, and engage heading to backfill into env/orders.DESTINATIONS and
TRAINER_APPROACH (both currently unverified).

IMPORTANT: reader.in_battle() can be a false-positive in some post-cutscene
states. The tool trusts it but prints the map_id; the operator must eyeball the
saved state to confirm it is the genuine rival battle.

One-shot scaffolding: run once locally where the ROM + post_starter.state exist.
Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_rival_battle.py
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_rival_battle.py \\
      --target-map 0 18 --heading up --max-steps 800
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

OUT_PATH = Path("states/trainer_battle.state")
_DIRECTIONS = (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT)
_HEADING_MAP = {"up": KEY_UP, "down": KEY_DOWN, "left": KEY_LEFT, "right": KEY_RIGHT}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capture the route_103 rival battle.")
    ap.add_argument("--state", default="states/post_starter.state")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument(
        "--heading",
        choices=list(_HEADING_MAP),
        default="up",
        help="Walk direction toward route_103 (default: up).",
    )
    ap.add_argument(
        "--target-map",
        type=int,
        nargs=2,
        default=[0, 18],
        metavar=("GROUP", "NUM"),
        help="route_103 map id to accept the battle on (default: 0 18, unverified).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    target_map = (args.target_map[0], args.target_map[1])

    rom = os.environ["POKEMON_EMERALD_ROM"]
    start = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [start], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()

    key = _HEADING_MAP[args.heading]
    for i in range(args.max_steps):
        env.emulator.step(key, GRIND_STEP_FRAMES)
        env.emulator.step(0, GRIND_RELEASE_FRAMES)   # release (GBA debounce)
        if reader.in_battle():
            snap = reader.snapshot()
            map_id = None if snap is None else snap.map_id
            if map_id != target_map:
                print(
                    f"battle fired on map {map_id} (not target {target_map}) after "
                    f"{i + 1} steps — likely a wild encounter blocking the path. "
                    f"Start from a state past the grass, or set --target-map.",
                    flush=True,
                )
                raise SystemExit(1)
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            print(
                f"RIVAL-BATTLE state saved after {i + 1} steps "
                f"(map {map_id}, pos {None if snap is None else snap.pos}) "
                f"-> {OUT_PATH.resolve()}",
                flush=True,
            )
            print(
                "NOTE: verify this is the genuine rival battle (eyeball the state) — "
                "in_battle() can be a false-positive. Backfill the real map/cell/"
                "heading into env/orders.DESTINATIONS + TRAINER_APPROACH.",
                flush=True,
            )
            return

    print(f"no battle in {args.max_steps} steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
