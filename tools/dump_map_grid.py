"""Throwaway ASCII dump of the loaded map grid (visual sanity check)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.map_grid_reader import MapGridReader, TileKind

_GLYPH = {
    TileKind.FREE: ".",
    TileKind.WALL: "#",
    TileKind.GRASS: '"',
    TileKind.LEDGE_UP: "^",
    TileKind.LEDGE_DOWN: "v",
    TileKind.LEDGE_LEFT: "<",
    TileKind.LEDGE_RIGHT: ">",
}


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state_path = sys.argv[1] if len(sys.argv) > 1 else "states/post_starter.state"
    emu = GbaEmulator(rom)
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    grid = MapGridReader(emu.read_bytes).grid()
    if grid is None:
        print("no map")
        sys.exit(1)
    ps = EmeraldReader(emu.read_bytes).player_state()
    for y, row in enumerate(grid):
        line = []
        for x, kind in enumerate(row):
            if ps is not None and (x, y) == (ps.x, ps.y):
                line.append("@")
            else:
                line.append(_GLYPH[kind])
        print("".join(line))


if __name__ == "__main__":
    main()
