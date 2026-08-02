"""Reach Birch's lab once and cache states/lab_entry.state.

Phase 0: run the trained Explorer from states/initial.state until the
"starter_obtained" milestone, then keep stepping until the forced wild battle
starts (reader.in_battle() True). Phase 1: hand the battle to the trained Fighter
via play_battle and win it. Then settle and save the post-battle state (in Birch's
lab). Bounded retry over the whole thing because Phase 0 is stochastic
(deterministic=False, ~9-10/10 reaches the starter).

One-time scaffolding: run once locally where the ROM + checkpoints exist. The
output feeds tools/probe_lab_intro.py and tools/capture_route101_freeroam.py so
they skip the slow intro. Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_lab_entry.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator.buttons import KEY_A
from emulator.gba import GbaEmulator
from env.battle_player import play_battle
from env.orders import DESTINATIONS
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/lab_entry.state")
ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16)


def _reach_starter_then_battle(env, model, reader, max_steps):
    """Step the Explorer to the starter, then on until the forced battle starts."""
    obs, _ = env.reset()
    got_starter = False
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, info = env.step(int(action))
        if not got_starter and "starter_obtained" in info["milestones"]:
            got_starter = True
        if got_starter and reader.in_battle():
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--explorer", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--fighter", default="checkpoints/fighter/ppo_fighter_final.zip")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    explorer = PPO.load(args.explorer, device="cpu")
    fighter = PPO.load(args.fighter, device="cpu")

    def predict(obs) -> int:
        return int(fighter.predict(obs, deterministic=True)[0])

    for attempt in range(args.attempts):
        if not _reach_starter_then_battle(env, explorer, reader, args.max_steps):
            print(f"attempt {attempt}: never reached starter+battle; retrying", flush=True)
            continue
        result = play_battle(env.emulator, make_move_type_fn(env.emulator), predict)
        print(f"attempt {attempt}: battle -> {result}", flush=True)
        if result != "won":
            continue
        # After play_battle the game is frozen on the victory frame waiting for
        # button presses to advance the post-battle dialogue and trigger the warp
        # to Birch's lab.  Spam A to clear every dialogue/script screen; keep
        # going until the map changes away from Route 101 (up to ~600 A-presses
        # ≈ 10 s at 60 fps).
        snap = None
        for settle_tick in range(600):
            env.emulator.step(KEY_A, 6)
            env.emulator.step(0, 10)
            snap = reader.snapshot()
            if snap is not None and snap.map_id != ROUTE_101:
                print(f"attempt {attempt}: map changed after {settle_tick} A-presses -> {snap.map_id}", flush=True)
                break
        if snap is None or snap.map_id == ROUTE_101:
            print(f"attempt {attempt}: still on route_101 after settling ({snap}); retrying", flush=True)
            continue
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_bytes(env.emulator.save_state())
        print(
            f"LAB ENTRY saved: map {snap.map_id} pos {snap.pos} "
            f"levels {reader.party_levels()} -> {OUT_PATH.resolve()}",
            flush=True,
        )
        return

    print(f"failed to reach the lab in {args.attempts} attempts", flush=True)


if __name__ == "__main__":
    main()
