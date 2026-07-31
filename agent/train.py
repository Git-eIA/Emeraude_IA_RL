"""Train PPO on Pokémon Emerald. Requires POKEMON_EMERALD_ROM and states/initial.state."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from emulator.gba import GbaEmulator
from env.capture.recorder import RecorderCallback
from env.pokemon_env import PokemonEmeraldEnv

log = logging.getLogger("agent.train")

STATE_PATH = Path("states/initial.state")


def make_env(rom_path: str, initial_state: bytes, max_steps: int):
    def _init() -> Monitor:
        # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean.
        env = PokemonEmeraldEnv(GbaEmulator(rom_path), initial_state, max_steps=max_steps)
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
    parser.add_argument("--run-id", default=None, help="capture run id (default: timestamp)")
    parser.add_argument("--capture", dest="capture", action="store_true", default=True)
    parser.add_argument("--no-capture", dest="capture", action="store_false")
    parser.add_argument("--capture-every", type=int, default=200)
    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--max-frame-gb", type=float, default=20.0)
    args = parser.parse_args()

    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        log.error("Set POKEMON_EMERALD_ROM")
        return 1
    if not STATE_PATH.is_file():
        log.error("Missing %s — create it with tools/play_interactive.py", STATE_PATH)
        return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("captures") / run_id

    initial_state = STATE_PATH.read_bytes()
    vec = SubprocVecEnv(
        [make_env(rom, initial_state, args.max_steps) for _ in range(args.envs)]
    )
    device = pick_device()
    log.info("Training on device=%s with %d envs (run=%s)", device, args.envs, run_id)

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

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_emerald",
        )
    ]
    if args.capture:
        meta = {
            "argv": sys.argv,
            "total_timesteps": args.timesteps,
            "rom": rom,
            "initial_state": str(STATE_PATH),
            "max_steps": args.max_steps,
        }
        callbacks.append(
            RecorderCallback(
                run_dir=run_dir,
                capture_every=args.capture_every,
                clip_len=args.clip_len,
                max_frame_gb=args.max_frame_gb,
                meta=meta,
            )
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=run_id,
        reset_num_timesteps=False,
    )
    model.save(str(run_dir / "checkpoints" / "ppo_emerald_final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
