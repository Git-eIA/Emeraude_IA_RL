"""Play the ROM with the keyboard; press S to save states/initial.state.

Keys: arrows = D-pad, X = A, Z = B, Enter = Start, Backspace = Select,
A = L, S(hift) keys per MAPPING below. Press F5 to save state, Esc to quit.
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
STATE_PATH = Path("states/initial.state")


def main() -> int:
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM")
        return 1
    emu = GbaEmulator(rom)
    pygame.init()
    screen = pygame.display.set_mode((240 * SCALE, 160 * SCALE))
    pygame.display.set_caption("Emerald — F5 saves states/initial.state")
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 0
                if event.key == pygame.K_F5:
                    STATE_PATH.parent.mkdir(exist_ok=True)
                    STATE_PATH.write_bytes(emu.save_state())
                    print(f"Saved {STATE_PATH}")
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
