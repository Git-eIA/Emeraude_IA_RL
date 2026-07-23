"""Capture battle-opening savestates into states/battles/ (human-in-loop).

Play until the FIGHT menu is visible (first turn, must-choose-a-move prompt),
then press F5 to write states/battles/<name>.state. Capture several wild
encounters (Routes 101/103) and the Route 103 rival battle to form the
Fighter's curriculum, plus a probe.state mid-battle for tools/probe_battle.py.

Keys: arrows = D-pad, X = A, Z = B, Enter = Start, Backspace = Select,
A = L, S = R. F5 = save, Esc = quit.

Usage: python tools/make_battle_states.py <name>   (default name: "battle")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pygame

from emulator import buttons
from emulator.gba import GbaEmulator

MAPPING = {
    pygame.K_UP: buttons.KEY_UP,
    pygame.K_DOWN: buttons.KEY_DOWN,
    pygame.K_LEFT: buttons.KEY_LEFT,
    pygame.K_RIGHT: buttons.KEY_RIGHT,
    pygame.K_x: buttons.KEY_A,
    pygame.K_z: buttons.KEY_B,
    pygame.K_RETURN: buttons.KEY_START,
    pygame.K_BACKSPACE: buttons.KEY_SELECT,
    pygame.K_a: buttons.KEY_L,
    pygame.K_s: buttons.KEY_R,
}
SCALE = 3
OUT_DIR = Path("states/battles")


def main() -> int:
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM")
        return 1
    name = sys.argv[1] if len(sys.argv) > 1 else "battle"
    state_path = OUT_DIR / f"{name}.state"

    emu = GbaEmulator(rom)
    # Resume from the standard start so the human plays into a battle from there.
    initial = Path("states/initial.state")
    if initial.exists():
        emu.load_state(initial.read_bytes())

    pygame.init()
    screen = pygame.display.set_mode((240 * SCALE, 160 * SCALE))
    pygame.display.set_caption(f"Emerald — F5 saves {state_path}")
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 0
                if event.key == pygame.K_F5:
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    state_path.write_bytes(emu.save_state())
                    print(f"Saved {state_path}")
        pressed = pygame.key.get_pressed()
        keys = 0
        for pg_key, gba_key in MAPPING.items():
            if pressed[pg_key]:
                keys |= gba_key
        emu.step(keys, frames=1)
        frame = np.transpose(emu.screenshot(), (1, 0, 2))  # pygame wants (w, h)
        surface = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    sys.exit(main())
