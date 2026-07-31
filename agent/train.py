"""Train PPO on Pokémon Emerald. Requires POKEMON_EMERALD_ROM and states/initial.state."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv

log = logging.getLogger("agent.train")

STATE_PATH = Path("states/initial.state")
EXPLORER_DIR = Path("states/explorer")


def load_initial_states(truck: Path, frontier_dir: Path) -> list[bytes]:
    """Truck state first, then every frontier savestate (Go-Explore Palier 0)."""
    if not truck.is_file():
        raise FileNotFoundError(truck)
    states = [truck.read_bytes()]
    if frontier_dir.is_dir():
        states += [p.read_bytes() for p in sorted(frontier_dir.glob("*.state"))]
    return states


def make_env(rom_path: str, initial_states: list[bytes], max_steps: int):
    def _init() -> Monitor:
        # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean.
        env = PokemonEmeraldEnv(GbaEmulator(rom_path), initial_states, max_steps=max_steps)
        return Monitor(env)

    return _init


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint .zip to resume")
    args = parser.parse_args()

    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        log.error("Set POKEMON_EMERALD_ROM")
        return 1
    if not STATE_PATH.is_file():
        log.error("Missing %s — create it with tools/play_interactive.py", STATE_PATH)
        return 1

    initial_states = load_initial_states(STATE_PATH, EXPLORER_DIR)
    log.info("Reset pool: %d state(s) (truck + %d frontier)", len(initial_states), len(initial_states) - 1)
    vec = SubprocVecEnv(
        [make_env(rom, initial_states, args.max_steps) for _ in range(args.envs)]
    )
    device = pick_device()
    log.info("Training on device=%s with %d envs", device, args.envs)

    if args.resume:
        model = PPO.load(args.resume, env=vec, device=device)
    else:
        model = PPO(
            "CnnPolicy",
            vec,
            n_steps=512,
            batch_size=512,
            ent_coef=0.01,
            learning_rate=3e-4,
            device=device,
            verbose=1,
            tensorboard_log="runs",
        )
    checkpoints = CheckpointCallback(
        save_freq=max(50_000 // args.envs, 1), save_path="checkpoints", name_prefix="ppo_emerald"
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoints, reset_num_timesteps=False)
    model.save("checkpoints/ppo_emerald_final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
