"""M1 acceptance: run a scripted input sequence, dump numbered screenshots."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

from emulator import buttons
from emulator.gba import GbaEmulator

# (keys, frames) pairs: mash through intro, then walk around
SCRIPT: list[tuple[int, int]] = [
    (0, 600),
    (buttons.KEY_A, 10), (0, 60),
    (buttons.KEY_A, 10), (0, 60),
    (buttons.KEY_START, 10), (0, 60),
    (buttons.KEY_A, 10), (0, 120),
    (buttons.KEY_DOWN, 32), (buttons.KEY_LEFT, 32),
    (buttons.KEY_UP, 32), (buttons.KEY_RIGHT, 32),
]


def main() -> int:
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM")
        return 1
    out_dir = Path("scripted_frames")
    out_dir.mkdir(exist_ok=True)
    emu = GbaEmulator(rom)
    for i, (keys, frames) in enumerate(SCRIPT):
        emu.step(keys, frames)
        Image.fromarray(emu.screenshot()).save(out_dir / f"{i:03d}.png")
    print(f"Wrote {len(SCRIPT)} frames to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
