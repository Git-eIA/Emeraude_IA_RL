"""Watch the trained agent play one episode in a pygame window.

Keys: Space = pause, Up/Down = speed, Esc = quit.
Run: POKEMON_EMERALD_ROM=... .venv/bin/python tools/watch.py [--model PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv

SCALE = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/ppo_emerald_final")
    parser.add_argument("--state", default="states/initial.state")
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sps", type=int, default=15, help="agent steps per second")
    args = parser.parse_args()

    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM")
        return 1

    env = PokemonEmeraldEnv(
        GbaEmulator(rom), [Path(args.state).read_bytes()], max_steps=args.max_steps
    )
    model = PPO.load(args.model, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((240 * SCALE, 160 * SCALE))
    clock = pygame.time.Clock()

    obs, info = env.reset(seed=args.seed)
    seen: set[str] = set(info["milestones"])
    total = 0.0
    step = 0
    sps = args.sps
    paused = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 0
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    sps = min(sps + 5, 60)
                if event.key == pygame.K_DOWN:
                    sps = max(sps - 5, 1)

        if not paused:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, term, trunc, info = env.step(int(action))
            total += reward
            step += 1
            for name in set(info["milestones"]) - seen:
                print(f"step {step}: milestone {name} (total={total:.1f})")
                seen.add(name)
            if term or trunc:
                print(f"episode over: {'starter!' if term else 'time out'} "
                      f"reward={total:.1f} steps={step}")
                paused = True

        frame = np.transpose(env.render(), (1, 0, 2))
        surface = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.set_caption(
            f"Agent — step {step}  reward {total:.1f}  {sps} sps"
            f"{'  [PAUSE]' if paused else ''}"
        )
        pygame.display.flip()
        clock.tick(sps)


if __name__ == "__main__":
    sys.exit(main())
