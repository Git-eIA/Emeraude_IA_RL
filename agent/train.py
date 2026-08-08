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

from agent.train_fighter import make_move_type_fn
from emulator.gba import GbaEmulator
from env.capture.recorder import RecorderCallback
from env.milestones import route103_milestones, starter_milestones
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


def select_milestones(name: str) -> tuple:
    """Map a CLI name to a milestone chain (reward curriculum)."""
    if name == "route103":
        return route103_milestones()
    return starter_milestones()


def make_env(
    rom_path: str,
    initial_states: list[bytes],
    max_steps: int,
    milestones_name: str = "starter",
    fighter_path: Path | None = None,
):
    def _init() -> Monitor:
        emu = GbaEmulator(rom_path)
        move_type_fn = None
        predict = None
        if fighter_path is not None:
            # SubprocVecEnv can't ship a live SB3 model across processes, so each
            # env subprocess loads the (tiny, CPU) Fighter itself and wraps it as
            # the in-episode auto-battler that PokemonEmeraldEnv.step calls.
            fighter = PPO.load(fighter_path, device="cpu")
            move_type_fn = make_move_type_fn(emu)

            def predict(obs):
                return int(fighter.predict(obs, deterministic=True)[0])

        # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean.
        env = PokemonEmeraldEnv(
            emu,
            initial_states,
            max_steps=max_steps,
            milestones=select_milestones(milestones_name),
            move_type_fn=move_type_fn,
            predict=predict,
        )
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
    parser.add_argument(
        "--milestones", choices=["starter", "route103"], default="starter",
        help="reward curriculum",
    )
    parser.add_argument(
        "--fighter", type=Path, default=None,
        help="Fighter checkpoint to auto-battle wild/trainer encounters in-episode",
    )
    parser.add_argument(
        "--start", type=Path, default=None,
        help="single start savestate (overrides the truck+frontier reset pool)",
    )
    args = parser.parse_args()

    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        log.error("Set POKEMON_EMERALD_ROM")
        return 1
    if args.start is None and not STATE_PATH.is_file():
        log.error("Missing %s — create it with tools/play_interactive.py", STATE_PATH)
        return 1
    if args.start is not None and not args.start.is_file():
        log.error("Missing --start %s", args.start)
        return 1
    if args.fighter is not None and not args.fighter.is_file():
        log.error("Missing --fighter %s", args.fighter)
        return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("captures") / run_id
    # Guarantee the checkpoint dir exists for the final save, even with --no-capture
    # on a run too short to trigger CheckpointCallback's own mkdir.
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.start is not None:
        initial_states = [args.start.read_bytes()]
        log.info("Reset pool: single start state %s", args.start)
    else:
        initial_states = load_initial_states(STATE_PATH, EXPLORER_DIR)
        log.info(
            "Reset pool: %d state(s) (truck + %d frontier)",
            len(initial_states), len(initial_states) - 1,
        )
    vec = SubprocVecEnv(
        [
            make_env(rom, initial_states, args.max_steps, args.milestones, args.fighter)
            for _ in range(args.envs)
        ]
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
            "initial_state": str(args.start) if args.start is not None else str(STATE_PATH),
            "milestones": args.milestones,
            "fighter": str(args.fighter) if args.fighter is not None else None,
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
