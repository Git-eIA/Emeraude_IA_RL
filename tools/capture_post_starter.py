"""Capture a post-starter overworld savestate by driving the trained Explorer.

Loads the Explorer PPO policy, plays from states/initial.state until the
"starter_obtained" milestone fires, then keeps stepping until the forced wild
battle (Poochyena attacking Birch) clears, and writes states/post_starter.state.
One-time artifact generation: run once locally where checkpoints/ppo_emerald_final.zip
and the ROM exist. The output is gitignored; the run_campaign ROM smoke skips when it
is absent.

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_post_starter.py --max-steps 8000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/post_starter.state")
SETTLE_FRAMES = 4   # consecutive out-of-battle snapshots before we trust the overworld


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    model = PPO.load(args.model, device="cpu")

    obs, _ = env.reset()
    got_starter = False
    settled = 0

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, info = env.step(int(action))

        if not got_starter and "starter_obtained" in info["milestones"]:
            got_starter = True
            print(f"starter_obtained at step {step}", flush=True)

        if got_starter:
            # Wait for the forced battle to clear: a run of out-of-battle frames.
            settled = settled + 1 if not reader.in_battle() else 0
            if settled >= SETTLE_FRAMES:
                snap = reader.snapshot()
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUT_PATH.write_bytes(env.emulator.save_state())
                print(
                    f"POST-STARTER at step {step}: "
                    f"map {snap.map_id if snap else None} "
                    f"pos {snap.pos if snap else None} "
                    f"in_battle {reader.in_battle()} "
                    f"levels {reader.party_levels()} "
                    f"-> {OUT_PATH.resolve()}",
                    flush=True,
                )
                return

    print(
        f"starter not obtained / battle not cleared in {args.max_steps} steps; "
        f"try more steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
