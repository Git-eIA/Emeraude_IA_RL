"""Train the Fighter: PPO + MlpPolicy on the battle env (numbers, not pixels)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from emulator.gba import GbaEmulator
from env.battle_env import BattleEmeraldEnv

# gBattleMoves ROM table: BattleMove struct, size 0xC, type at offset 0x02.
# TODO(probe): confirm GBATTLE_MOVES_ADDR on BPEF (a ROM address, 0x08......).
GBATTLE_MOVES_ADDR = 0x0831C898
BATTLE_MOVE_SIZE = 0xC
BATTLE_MOVE_TYPE_OFF = 0x02


def make_move_type_fn(emu: GbaEmulator):
    def move_type(move_id: int) -> int:
        if move_id <= 0:
            return 0
        addr = GBATTLE_MOVES_ADDR + move_id * BATTLE_MOVE_SIZE + BATTLE_MOVE_TYPE_OFF
        return emu.read_bytes(addr, 1)[0]

    return move_type


def load_states(directory: str) -> list[bytes]:
    states = [
        p.read_bytes()
        for p in sorted(Path(directory).glob("*.state"))
        if p.name != "probe.state"
    ]
    if not states:
        raise SystemExit(f"no savestates in {directory}")
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--states-dir", default="states/battles")
    ap.add_argument("--max-turns", type=int, default=64)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    states = load_states(args.states_dir)

    def make_env():
        emu = GbaEmulator(rom)
        return BattleEmeraldEnv(
            emu, states, make_move_type_fn(emu), max_turns=args.max_turns
        )

    env = DummyVecEnv([make_env])
    ckpt = CheckpointCallback(
        save_freq=50_000, save_path="checkpoints/fighter", name_prefix="ppo_fighter"
    )
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save("checkpoints/fighter/ppo_fighter_final")


if __name__ == "__main__":
    main()
