"""Throwaway BPEF map-grid discovery/validation probe.

Loads a mid-overworld savestate, prints the gBackupMapLayout struct + the
gMapHeader tileset chain, then exercises MapGridReader at the player's tile and
sweeps the grid for LEDGE_*/GRASS tiles. Exits 0 only if the reader decodes a
sane map (dimensions resolve AND the player's own tile classifies FREE). A
non-zero exit means the seeded BPEE addresses/padding are wrong for BPEF ->
escalate to the user before trusting the reader.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.map_grid_reader import MapGridReader, TileKind


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state_path = sys.argv[1] if len(sys.argv) > 1 else "states/post_starter.state"
    emu = GbaEmulator(rom)
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)  # let one frame settle so the map layout is coherent

    grid = MapGridReader(emu.read_bytes)
    player = EmeraldReader(emu.read_bytes).player_state()
    dims = grid.dimensions()
    print(f"dimensions() = {dims}")
    if player is not None:
        px, py = player.x, player.y
        print(f"player at ({px},{py})")
        print(f"  classify_at = {grid.classify_at(px, py)}")
        print(f"  tile_behavior_at = {grid.tile_behavior_at(px, py)}")
    else:
        px = py = None
        print("player_state() is None")

    if dims is not None:
        w, h = dims
        ledges = []
        grasses = []
        for y in range(h):
            for x in range(w):
                kind = grid.classify_at(x, y)
                if kind in (TileKind.LEDGE_UP, TileKind.LEDGE_DOWN,
                            TileKind.LEDGE_LEFT, TileKind.LEDGE_RIGHT):
                    ledges.append((x, y, kind.name))
                elif kind is TileKind.GRASS:
                    grasses.append((x, y))
        print(f"ledges found: {ledges[:20]}{' ...' if len(ledges) > 20 else ''}")
        print(f"grass tiles: {len(grasses)} (e.g. {grasses[:10]})")

    ok = (
        dims is not None
        and player is not None
        and grid.classify_at(px, py) is TileKind.FREE
    )
    print("VALIDATION:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
