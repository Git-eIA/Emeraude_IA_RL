"""Auto-capture wild-battle openings by letting the Explorer walk into grass.

Loads the trained Explorer (pixel policy), plays from states/initial.state, and
the moment BattleReader reports in_battle it writes states/battles/<prefix>_<i>.state.
Repeats --count times with different seeds. Prints the parsed BattleState on each
capture so the run doubles as a RAM-address sanity check (like probe_battle.py):
if the printed HP/level/species are plausible, the BPEF addresses are correct.

Only WILD battles are reachable this way (the Explorer wanders into grass). The
Route 103 rival is a scripted trainer fight and must be captured separately.

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/auto_capture_battles.py \
      --count 5 --max-steps 4000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator import buttons
from emulator.gba import GbaEmulator
from env.game_state import GBATTLE_TYPE_FLAGS_ADDR, BattleReader
from env.pokemon_env import PokemonEmeraldEnv

OUT_DIR = Path("states/battles")

# Advance the fresh battle past its intro dialogue to the first action menu so
# the saved state loads straight into a move choice (cheap env resets). Mirrors
# BattleEmeraldEnv's press/advance choreography.
_ADVANCE_PRESSES = 120


def _press(emu, key: int) -> None:
    emu.step(key, 6)
    emu.step(0, 10)


def _advance_to_menu(emu, reader: BattleReader) -> bool:
    """Press A until the action menu appears; True if reached, False if stuck."""
    for _ in range(_ADVANCE_PRESSES):
        if reader.at_action_menu():
            emu.step(0, 8)  # settle so the menu accepts input
            return True
        _press(emu, buttons.KEY_A)
    return False


def _u32(read, addr: int) -> int:
    b = read(addr, 4)
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    ap.add_argument("--prefix", default="wild")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    # Give the env a large step budget so its own truncation never interrupts us.
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
    reader = BattleReader(env.emulator.read_bytes)
    model = PPO.load(args.model, device="cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captured = 0
    for i in range(args.count):
        obs, _ = env.reset(seed=1000 + i)
        max_flags = 0
        hit = False
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, _, _, _ = env.step(int(action))
            max_flags = max(max_flags, _u32(env.emulator.read_bytes, GBATTLE_TYPE_FLAGS_ADDR))
            bs = reader.battle_state()
            if bs.in_battle:
                reached = _advance_to_menu(env.emulator, reader)
                bs = reader.battle_state()
                path = OUT_DIR / f"{args.prefix}_{i}.state"
                path.write_bytes(env.emulator.save_state())
                captured += 1
                hit = True
                print(
                    f"run {i}: BATTLE at step {step} "
                    f"(action menu {'reached' if reached else 'NOT reached'}) -> {path}\n"
                    f"   me lvl{bs.my_level} hp {bs.my_hp}/{bs.my_max_hp} types {bs.my_types}\n"
                    f"   opp species {bs.opp_species} lvl{bs.opp_level} "
                    f"hp {bs.opp_hp}/{bs.opp_max_hp} types {bs.opp_types}",
                    flush=True,
                )
                break
        if not hit:
            print(
                f"run {i}: no battle in {args.max_steps} steps "
                f"(max gBattleTypeFlags seen = 0x{max_flags:08X})",
                flush=True,
            )

    print(f"\ncaptured {captured}/{args.count} wild-battle openings in {OUT_DIR}")
    if captured == 0:
        print(
            "0 captures. Either the Explorer never entered grass, or the battle "
            "RAM addresses are wrong for BPEF. If max gBattleTypeFlags stayed "
            "0x00000000 across runs, validate the address with tools/probe_battle.py."
        )


if __name__ == "__main__":
    main()
