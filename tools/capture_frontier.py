"""Capture frontier savestates by letting the trained Explorer play.

Drives the Explorer PPO policy from states/initial.state and, the first time a
frontier milestone appears in the env info, writes states/explorer/<name>.state.
These become extra reset points for Go-Explore Palier 0 training. One-shot,
run locally where checkpoints/ppo_emerald_final.zip exists (outputs gitignored).

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_frontier.py \
      --episodes 5 --max-steps 4096
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv

# Milestones in the detachment zone worth restarting from. exit_truck/enter_house/
# clock_set are too early to matter; starter_obtained is terminal.
FRONTIER_MILESTONES = frozenset(
    {"enter_rival_house", "rival_upstairs", "meet_rival", "north_littleroot", "reach_route_101"}
)
OUT_DIR = Path("states/explorer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=args.max_steps)
    model = PPO.load(args.model, device="cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captured: set[str] = set()

    for episode in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(int(action))
            for name in info["milestones"]:
                if name in FRONTIER_MILESTONES and name not in captured:
                    out = OUT_DIR / f"{name}.state"
                    out.write_bytes(env.emulator.save_state())
                    captured.add(name)
                    print(f"captured {name} -> {out.resolve()}", flush=True)
            if terminated or truncated:
                break
        if captured >= FRONTIER_MILESTONES:
            break

    missing = sorted(FRONTIER_MILESTONES - captured)
    print(f"done: {len(captured)} captured, missing={missing}", flush=True)


if __name__ == "__main__":
    main()
