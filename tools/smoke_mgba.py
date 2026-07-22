"""Boot a ROM headless, run 300 frames, dump a screenshot. Proves the core works."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mgba.core
import mgba.image
import mgba.log


def main() -> int:
    mgba.log.silence()
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM to a .gba file")
        return 1
    core = mgba.core.load_path(rom)
    if core is None:
        print(f"mGBA could not load: {rom}")
        return 1
    width, height = core.desired_video_dimensions()
    image = mgba.image.Image(width, height)
    core.set_video_buffer(image)
    core.reset()
    for _ in range(300):
        core.run_frame()
    out = Path("smoke_frame.png")
    image.to_pil().convert("RGB").save(out)
    # Game code at 0x080000AC identifies the ROM (Emerald FR = BPEF)
    game_code = bytes(core.memory.u8[0x080000AC:0x080000B0]).decode("ascii", "replace")
    print(f"OK — game code {game_code} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
