"""Walk from a known savestate until a trainer battle fires, then cache
states/trainer_battle.state.

Loads a start savestate (default: states/post_starter.state) and cycles the
four d-pad directions with a release-frame debounce until reader.in_battle()
flips True, then saves the emulator state.

Whether a trainer battle is actually reachable depends on the start state.
If no battle fires within --max-steps, the tool exits 1 and the ROM smoke
(tests/test_battle_player_rom.py::test_fighter_wins_a_real_trainer_battle)
stays a documented skip — mirrors the Brique 2 precedent for route101_in_battle.

IMPORTANT: reader.in_battle() can be a false-positive in some post-cutscene
states (documented in project history, e.g. Birch's lab). The tool trusts it,
but the operator must eyeball the printed map_id and the saved state to confirm
it is a genuine trainer battle (not a wild encounter, not a false positive).
This is acceptable for a disposable capture tool.

One-shot scaffolding: run once locally where the ROM + start state exist.
Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_trainer_battle.py
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_trainer_battle.py \\
      --state states/post_starter.state --max-steps 800 --heading up
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
_HEADING_MAP = {
    "up": KEY_UP,
    "down": KEY_DOWN,
    "left": KEY_LEFT,
    "right": KEY_RIGHT,
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capture a trainer-battle savestate.")
    ap.add_argument("--state", default="states/post_starter.state")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument(
        "--heading",
        choices=list(_HEADING_MAP),
        default=None,
        help="Fix a single walk direction instead of cycling all four.",
    )
    return ap.parse_args()


def _walk_one(env: PokemonEmeraldEnv, key: int) -> None:
    """Press one d-pad direction with the GBA debounce release frame."""
    env.emulator.step(key, GRIND_STEP_FRAMES)
    env.emulator.step(0, GRIND_RELEASE_FRAMES)


def main() -> None:
    args = _parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    start = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [start], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()

    fixed_key: int | None = _HEADING_MAP.get(args.heading) if args.heading else None

    for i in range(args.max_steps):
        key = fixed_key if fixed_key is not None else _DIRECTIONS[i % len(_DIRECTIONS)]
        _walk_one(env, key)
        if reader.in_battle():
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            snap = reader.snapshot()
            map_id = None if snap is None else snap.map_id
            print(
                f"IN-BATTLE state saved after {i + 1} steps "
                f"(map {map_id}) -> {OUT_PATH.resolve()}",
                flush=True,
            )
            print(
                "NOTE: verify this is a genuine trainer battle (check map_id, "
                "eyeball the state) — in_battle() can be a false-positive.",
                flush=True,
            )
            return

    print(f"no battle in {args.max_steps} steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
