"""Empirically validate battle RAM addresses on BPEF from a mid-battle savestate.

Run against a savestate captured DURING a battle (see tools/make_battle_states.py,
F5 while the FIGHT menu is visible). Prints candidate addresses and the parsed
values so we can confirm the pret-documented offsets hold on the FR ROM before
any dependent code trusts them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator

# Candidate symbol addresses to validate. These mirror the constants added to
# env/game_state.py (BPEE/US values); the FR (BPEF) build may differ. The probe
# confirms or corrects them.
CANDIDATES = {
    "gBattleMons": 0x02024084,
    "gBattleTypeFlags": 0x02022FEC,
    "gBattleOutcome": 0x0202433A,
    "gMoveResultFlags": 0x0202427C,
}
MON_SIZE = 0x58


def _u8(read, addr: int) -> int:
    return read(addr, 1)[0]


def _u16(read, addr: int) -> int:
    b = read(addr, 2)
    return b[0] | (b[1] << 8)


def _u32(read, addr: int) -> int:
    b = read(addr, 4)
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state_path = sys.argv[1] if len(sys.argv) > 1 else "states/battles/probe.state"
    emu = GbaEmulator(rom)
    with open(state_path, "rb") as fh:
        emu.load_state(fh.read())
    read = emu.read_bytes

    flags = _u32(read, CANDIDATES["gBattleTypeFlags"])
    outcome = _u8(read, CANDIDATES["gBattleOutcome"])
    print(f"gBattleTypeFlags = 0x{flags:08X} (nonzero => in battle)")
    print(f"gBattleOutcome   = {outcome} (0 => ongoing)")
    print(f"gMoveResultFlags = 0x{_u16(read, CANDIDATES['gMoveResultFlags']):04X}")

    base = CANDIDATES["gBattleMons"]
    for slot in range(2):
        addr = base + slot * MON_SIZE
        species = _u16(read, addr + 0x00)
        hp = _u16(read, addr + 0x28)
        max_hp = _u16(read, addr + 0x2C)
        level = _u8(read, addr + 0x2A)
        t1 = _u8(read, addr + 0x21)
        t2 = _u8(read, addr + 0x22)
        moves = [_u16(read, addr + 0x0C + 2 * i) for i in range(4)]
        pp = [_u8(read, addr + 0x24 + i) for i in range(4)]
        who = "player" if slot == 0 else "opponent"
        print(
            f"[{who}] species={species} lvl={level} hp={hp}/{max_hp} "
            f"types=({t1},{t2}) moves={moves} pp={pp}"
        )


if __name__ == "__main__":
    main()
