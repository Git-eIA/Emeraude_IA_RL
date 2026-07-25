"""Capture an open-map savestate by letting the trained Explorer walk out.

Loads the Explorer PPO policy, plays from states/initial.state, and the moment
WorldReader reports the player has left the truck interior onto a different map
it writes states/open_map.state. One-time artifact generation: run once locally
where checkpoints/ppo_emerald_final.zip exists; the output is committed so the
map-explorer ROM smoke does not need the checkpoint afterwards.

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_open_map.py \
      --max-steps 6000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/open_map.state")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), initial, max_steps=10_000_000)
    reader = WorldReader(EmeraldReader(env.emulator.read_bytes))
    model = PPO.load(args.model, device="cpu")

    obs, _ = env.reset()
    start = reader.snapshot()
    start_map = start.map_id if start is not None else None
    print(f"start map: {start_map}", flush=True)

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, _ = env.step(int(action))
        snap = reader.snapshot()
        if snap is not None and snap.map_id != start_map:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            print(
                f"OPEN MAP at step {step}: map {snap.map_id} pos {snap.pos} "
                f"-> {OUT_PATH.resolve()}",
                flush=True,
            )
            return

    print(
        f"no open map reached in {args.max_steps} steps "
        f"(still on {start_map}); try more steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
