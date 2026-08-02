"""Capture a post-starter overworld savestate by driving the trained Explorer.

Loads the Explorer PPO policy, plays from states/initial.state until the
"starter_obtained" milestone fires, then keeps stepping until the player has
real free-roam control on route_101, and writes states/post_starter.state.
One-time artifact generation: run once locally where checkpoints/ppo_emerald_final.zip
and the ROM exist. The output is gitignored; the run_campaign ROM smoke skips when it
is absent.

"starter_obtained" fires the instant party_count>=1 (grabbing the starter from
Birch's bag) — DURING the bag-grab cutscene, before the forced wild Poochyena
battle and its post-battle dialogue. The player cannot walk through any of that,
so a mere out-of-battle frame is not enough: a state captured there is stuck
(every d-pad press is swallowed). The only proof of free-roam control is a real
overworld position change, so we anchor on the first out-of-battle pos on
route_101 after the starter and capture once it actually moves.

KNOWN LIMITATION (why the run_campaign ROM smoke is currently a documented skip):
the trained Explorer's episode ENDS at starter_obtained (terminal milestone).
After the forced battle the player is warped to Birch's lab (map (1,4)); getting
back to route_101 in free-roam requires clearing the lab intro (talk to Birch for
the Pokédex, rival scene, exit) — script gates a wandering policy will not clear.
So this tool does not produce the artifact with the current checkpoint: it writes
nothing and reports "no free-roam movement". A route_101 free-roam capture needs
a scripted lab-intro walkthrough or a further-progressed Explorer (follow-up).

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_post_starter.py --max-steps 8000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.orders import DESTINATIONS
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/post_starter.state")
ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16), the map the smoke advances on


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--model", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    model = PPO.load(args.model, device="cpu")

    obs, _ = env.reset()
    got_starter = False
    ref_pos: tuple[int, int] | None = None   # first out-of-battle route_101 pos post-starter

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, info = env.step(int(action))

        if not got_starter and "starter_obtained" in info["milestones"]:
            got_starter = True
            print(f"starter_obtained at step {step}", flush=True)

        if not got_starter or reader.in_battle():
            continue
        snap = reader.snapshot()
        if snap is None or snap.map_id != ROUTE_101:
            continue
        if ref_pos is None:
            ref_pos = snap.pos
            continue
        if snap.pos != ref_pos:   # the player actually walked: free-roam confirmed
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            print(
                f"POST-STARTER at step {step}: "
                f"map {snap.map_id} "
                f"pos {snap.pos} "
                f"in_battle {reader.in_battle()} "
                f"levels {reader.party_levels()} "
                f"-> {OUT_PATH.resolve()}",
                flush=True,
            )
            return

    print(
        f"starter not obtained / no free-roam movement in {args.max_steps} steps; "
        f"try more steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
