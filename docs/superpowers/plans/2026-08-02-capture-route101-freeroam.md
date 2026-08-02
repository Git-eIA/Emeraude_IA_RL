# Capture route_101 free-roam savestate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `states/post_starter.state` (level-5 party, free-roam on route_101 `(0,16)`) via a scripted lab-intro walkthrough, so the pre-existing gated smoke `tests/test_campaign_rom.py` turns green.

**Architecture:** Three disposable `tools/` scripts driving the raw emulator + `WorldReader` (NOT `PokemonEmeraldEnv`, which terminates at the starter milestone). `capture_lab_entry.py` reaches Birch's lab once (Explorer → starter → Fighter wins the forced battle) and caches `states/lab_entry.state`. `probe_lab_intro.py` loads that cache and logs the lab/Littleroot/route_101 layout to discover the coordinates the driver needs. `capture_route101_freeroam.py` loads the cache, clears the lab script-gate, crosses lab → Littleroot → route_101, and saves the artifact. The acceptance test is the existing gated smoke going from skip to pass.

**Tech Stack:** Python 3.12, Stable-Baselines3 PPO (Explorer + Fighter checkpoints), the project emulator (`emulator.gba.GbaEmulator`), `env.world_reader.WorldReader`, `env.battle_player.play_battle`, `env.live_navigator.navigate_to`, `env.local_navigator.WallMap`.

**Why no unit tests on the tools:** these are one-shot ROM-driving scripts (same convention as `tools/capture_open_map.py`, `capture_frontier.py`, `capture_post_starter.py` — none are unit-tested). Correctness is proven by running them against the ROM and by the gated smoke passing (Task 4).

**Runtime path convention (all ROM commands):** the ROM, checkpoints, and `states/` live in the MAIN repo `/Users/_eloi/Projets/Emu`, not this worktree. Run every tool with `cwd` = the main repo, pointing Python at the worktree's tool file. The env var `POKEMON_EMERALD_ROM` names the ROM:

```bash
cd /Users/_eloi/Projets/Emu && \
POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
/Users/_eloi/Projets/Emu/.venv/bin/python \
/Users/_eloi/Projets/Emu-p4-capture-route101/tools/<script>.py <args>
```

**Task data-dependency (read before starting):** Task 2's probe DISCOVERS four numeric constants (lab map-id, lab exit cell + door direction, Littleroot north-edge cell). Task 3 hardcodes them. The seed values in Task 3 come from an earlier trajectory trace (lab `(1,4)`); the probe confirms or corrects them. Do Task 2 before finalizing Task 3, and substitute the probe's logged values into Task 3's `CONSTANTS` block.

---

## File Structure

- **Create** `tools/capture_lab_entry.py` — Phase 0 (Explorer → `starter_obtained`, bounded retry) + Phase 1 (`play_battle` wins the forced Poochyena) → writes `states/lab_entry.state`. Fast, reproducible entry point so Tasks 2-3 don't re-pay the slow, stochastic intro each run.
- **Create** `tools/probe_lab_intro.py` — loads `states/lab_entry.state`, presses A to clear the auto-dialogue then walks cardinally, logging `map_id`/`pos`/`in_battle` on every change. Discovers the four constants and whether an A-spam-then-test-move clears the gate.
- **Create** `tools/capture_route101_freeroam.py` — loads `states/lab_entry.state`, Phase 2 (clear gate) + Phase 3 (cross lab → Littleroot → route_101), writes `states/post_starter.state`. Consumes Task 2's constants.
- **Delete** `tools/capture_post_starter.py` — obsolete (its KNOWN LIMITATION is what this plan resolves).
- **Delete** `tools/probe_lab_intro.py` at the end — pure discovery scaffolding.
- **Unchanged** `tests/test_campaign_rom.py` — the acceptance test.
- **Modify** `docs/superpowers/specs/2026-08-02-capture-route101-freeroam-design.md` — status → implemented + record the discovered constants.

---

## Task 1: Reach the lab once and cache `states/lab_entry.state`

**Files:**
- Create: `tools/capture_lab_entry.py`

- [ ] **Step 1: Write the tool**

```python
"""Reach Birch's lab once and cache states/lab_entry.state.

Phase 0: run the trained Explorer from states/initial.state until the
"starter_obtained" milestone, then keep stepping until the forced wild battle
starts (reader.in_battle() True). Phase 1: hand the battle to the trained Fighter
via play_battle and win it. Then settle and save the post-battle state (in Birch's
lab). Bounded retry over the whole thing because Phase 0 is stochastic
(deterministic=False, ~9-10/10 reaches the starter).

One-time scaffolding: run once locally where the ROM + checkpoints exist. The
output feeds tools/probe_lab_intro.py and tools/capture_route101_freeroam.py so
they skip the slow intro. Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_lab_entry.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator import buttons
from emulator.gba import GbaEmulator
from env.orders import DESTINATIONS
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/lab_entry.state")
ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16)


def _reach_starter_then_battle(env, model, reader, max_steps):
    """Step the Explorer to the starter, then on until the forced battle starts."""
    obs, _ = env.reset()
    got_starter = False
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, info = env.step(int(action))
        if not got_starter and "starter_obtained" in info["milestones"]:
            got_starter = True
        if got_starter and reader.in_battle():
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--explorer", default="checkpoints/ppo_emerald_final")
    ap.add_argument("--fighter", default="checkpoints/fighter/ppo_fighter_final.zip")
    ap.add_argument("--state", default="states/initial.state")
    args = ap.parse_args()

    rom = os.environ["POKEMON_EMERALD_ROM"]
    initial = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [initial], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    explorer = PPO.load(args.explorer, device="cpu")
    fighter = PPO.load(args.fighter, device="cpu")

    def predict(obs) -> int:
        return int(fighter.predict(obs, deterministic=True)[0])

    for attempt in range(args.attempts):
        if not _reach_starter_then_battle(env, explorer, reader, args.max_steps):
            print(f"attempt {attempt}: never reached starter+battle; retrying", flush=True)
            continue
        result = play_battle(env.emulator, make_move_type_fn(env.emulator), predict)
        print(f"attempt {attempt}: battle -> {result}", flush=True)
        if result != "won":
            continue
        env.emulator.step(0, 30)   # settle through the post-battle warp
        snap = reader.snapshot()
        if snap is None or snap.map_id == ROUTE_101:
            print(f"attempt {attempt}: unexpected post-battle map {snap}; retrying", flush=True)
            continue
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_bytes(env.emulator.save_state())
        print(
            f"LAB ENTRY saved: map {snap.map_id} pos {snap.pos} "
            f"levels {reader.party_levels()} -> {OUT_PATH.resolve()}",
            flush=True,
        )
        return

    print(f"failed to reach the lab in {args.attempts} attempts", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the missing import**

The body above uses `play_battle` but does not import it. Add to the import block:

```python
from env.battle_player import play_battle
```

- [ ] **Step 3: Run it against the ROM**

Run (background-capable, slow — minutes):
```bash
cd /Users/_eloi/Projets/Emu && \
POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
/Users/_eloi/Projets/Emu/.venv/bin/python \
/Users/_eloi/Projets/Emu-p4-capture-route101/tools/capture_lab_entry.py 2>&1 | tail -20
```
Expected: a line `LAB ENTRY saved: map (M) pos (X,Y) levels [5,...] -> .../states/lab_entry.state`, where `map` is NOT `(0, 16)` (route_101) and NOT a battle. Record the printed `map` — it is the lab map-id used in Task 2/3.

If it reports "failed to reach the lab", re-run with `--attempts 10 --max-steps 12000`.

- [ ] **Step 4: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101
git add tools/capture_lab_entry.py
git commit -m "feat: capture_lab_entry tool — cache post-starter lab savestate"
```

---

## Task 2: Probe the lab intro to discover the layout

**Files:**
- Create: `tools/probe_lab_intro.py`

- [ ] **Step 1: Write the probe**

```python
"""Discover the lab -> Littleroot -> route_101 layout for the capture driver.

Loads states/lab_entry.state (from tools/capture_lab_entry.py) and:
  Phase A: press A + a test DOWN each iteration; log when a test move actually
           changes pos (= the script gate cleared and control returned).
  Phase B: once in control, walk DOWN until the map_id changes (the lab exit warp),
           logging every map_id/pos change; then walk UP until map_id becomes
           route_101 (0,16), logging every change.

Reads off, from the log: the lab map-id, the last lab cell before the exit warp
(LAB_EXIT_CELL) + the exit direction, the map the exit lands on (expected
Littleroot (0,9)), and the Littleroot cell whose UP press enters route_101
(LITTLEROOT_NORTH_CELL). Disposable — deleted after the driver is written.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/probe_lab_intro.py
"""
from __future__ import annotations

import os
from pathlib import Path

from emulator import buttons
from emulator.gba import GbaEmulator
from env.live_navigator import snapshot_settled
from env.orders import DESTINATIONS
from env.world_reader import WorldReader

ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16)
STATE = Path("states/lab_entry.state")

_DIRS = {"up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
         "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT}


def _press(emu, key, hold=24, release=8):
    emu.step(key, hold)
    emu.step(0, release)


def _log(reader, tag):
    snap = snapshot_settled(reader)
    where = None if snap is None else (snap.map_id, snap.pos)
    print(f"{tag}: {where} in_battle={reader.in_battle()}", flush=True)
    return snap


def _walk_until_map_change(emu, reader, direction, tag, max_steps=25):
    """Press `direction` until map_id changes; log each pos/map change."""
    start = snapshot_settled(reader)
    last = None if start is None else (start.map_id, start.pos)
    for i in range(max_steps):
        _press(emu, _DIRS[direction])
        snap = snapshot_settled(reader)
        cur = None if snap is None else (snap.map_id, snap.pos)
        if cur != last:
            print(f"{tag} step {i} {direction}: {cur}", flush=True)
            last = cur
        if snap is not None and start is not None and snap.map_id != start.map_id:
            print(f"{tag}: MAP CHANGE to {snap.map_id} at pos {snap.pos}", flush=True)
            return snap
    print(f"{tag}: no map change after {max_steps} {direction} presses", flush=True)
    return snapshot_settled(reader)


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    _log(reader, "loaded")

    # Phase A: clear the auto-dialogue, detect regained control via a test move.
    cleared = False
    for i in range(40):
        _press(emu, buttons.KEY_A, hold=6, release=10)
        before = snapshot_settled(reader)
        _press(emu, buttons.KEY_DOWN)
        after = snapshot_settled(reader)
        if before is not None and after is not None and after.pos != before.pos:
            print(f"GATE CLEARED at A-press {i}: moved {before.pos} -> {after.pos}", flush=True)
            cleared = True
            break
    if not cleared:
        print("gate NOT cleared by A-spam+test-move; inspect the log", flush=True)

    # Phase B: exit the lab (south), then head north toward route_101.
    landed = _walk_until_map_change(emu, reader, "down", "EXIT-LAB")
    if landed is not None and landed.map_id != ROUTE_101:
        _walk_until_map_change(emu, reader, "up", "TO-ROUTE101")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and capture the log**

Run:
```bash
cd /Users/_eloi/Projets/Emu && \
POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
/Users/_eloi/Projets/Emu/.venv/bin/python \
/Users/_eloi/Projets/Emu-p4-capture-route101/tools/probe_lab_intro.py 2>&1 | tee /tmp/probe_lab.log
```
Expected log lines to extract:
- `GATE CLEARED at A-press N: moved (x,y) -> (x,y')` → confirms Phase 2 is A-spam-then-test-move (no walk-to-Birch needed). If instead "gate NOT cleared", the intro needs a different clear (e.g. a rival scene) — STOP and report; Task 3 will need adjustment.
- `EXIT-LAB: MAP CHANGE to (Mexit) at pos (X,Y)` → `Mexit` is the map the lab exits to (expected Littleroot `(0,9)`); the lab cell one step before this is `LAB_EXIT_CELL`, the door direction is `down` (or whichever direction produced the change — the probe only tries `down`; if no change, re-run editing the direction).
- `TO-ROUTE101: MAP CHANGE to (0, 16) at pos (X,Y)` → the Littleroot cell one step before is `LITTLEROOT_NORTH_CELL`.

Write the extracted values into the commit message so Task 3 can consume them.

- [ ] **Step 3: Commit the probe + recorded findings**

```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101
git add tools/probe_lab_intro.py
git commit -m "$(cat <<'EOF'
chore: probe_lab_intro tool + recorded lab layout

Discovered (from states/lab_entry.state):
  LAB_MAP = <from Task 1/2>
  gate clears via A-spam + test move: <yes/no, at press N>
  LAB_EXIT_CELL = <x,y>, exit direction = <down/...> -> lands on <Mexit>
  LITTLEROOT_NORTH_CELL = <x,y> -> UP enters route_101 (0,16)
EOF
)"
```

---

## Task 3: Write the capture driver and produce the artifact

**Files:**
- Create: `tools/capture_route101_freeroam.py`

- [ ] **Step 1: Write the driver (substitute Task 2's constants)**

```python
"""Capture states/post_starter.state: level-5 party, free-roam on route_101.

Loads states/lab_entry.state (post forced-battle, in Birch's lab) and scripts the
lab intro to route_101:
  Phase 2: clear the lab script-gate (advance the Pokedex auto-dialogue with A;
           a test move that changes pos proves control returned), then leave the
           lab (walk the exit direction until the map changes to Littleroot).
  Phase 3: navigate_to the Littleroot north-edge cell, press UP until the map
           becomes route_101 (0,16), confirm free-roam with a real pos change,
           and save the artifact.

navigate_to is called WITHOUT `memory`, so the (lab) in_battle() false-positive
never reaches the has_grass-learning branch and cannot affect pathfinding.

All CONSTANTS below are from tools/probe_lab_intro.py (Task 2). The seeds are an
earlier trajectory trace; replace with the probe's logged values.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_route101_freeroam.py
"""
from __future__ import annotations

import os
from pathlib import Path

from emulator import buttons
from emulator.gba import GbaEmulator
from env.live_navigator import navigate_to, snapshot_settled
from env.local_navigator import WallMap
from env.orders import DESTINATIONS
from env.world_reader import WorldReader

# NOTE: all four filled from tools/probe_lab_intro.py (Task 2).
LAB_MAP = (1, 4)                 # confirmed by Task 1's "LAB ENTRY saved: map ..."
LAB_EXIT_DIR = "down"           # direction that warps out of the lab
LITTLEROOT = (0, 9)             # map the lab exit lands on
LITTLEROOT_NORTH_CELL = (10, 1) # cell whose UP press enters route_101
ROUTE_101 = DESTINATIONS["route_101"][0]   # (0, 16)

STATE = Path("states/lab_entry.state")
OUT_PATH = Path("states/post_starter.state")

_DIRS = {"up": buttons.KEY_UP, "down": buttons.KEY_DOWN,
         "left": buttons.KEY_LEFT, "right": buttons.KEY_RIGHT}


def _press(emu, key, hold=24, release=8):
    emu.step(key, hold)
    emu.step(0, release)


def _clear_gate(emu, reader, max_presses=40):
    """Advance the auto-dialogue; a test move that changes pos proves control."""
    for _ in range(max_presses):
        _press(emu, buttons.KEY_A, hold=6, release=10)
        before = snapshot_settled(reader)
        _press(emu, _DIRS[LAB_EXIT_DIR])   # test move also heads toward the exit
        after = snapshot_settled(reader)
        if before is not None and after is not None and after.pos != before.pos:
            return True
    return False


def _walk_until_map(emu, reader, direction, target_map, max_steps=25):
    """Press `direction` until snapshot().map_id == target_map."""
    for _ in range(max_steps):
        snap = snapshot_settled(reader)
        if snap is not None and snap.map_id == target_map:
            return True
        _press(emu, _DIRS[direction])
    snap = snapshot_settled(reader)
    return snap is not None and snap.map_id == target_map


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    emu.load_state(STATE.read_bytes())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    if not _clear_gate(emu, reader):
        print("Phase 2: gate not cleared", flush=True)
        return

    # Leave the lab (the test move in _clear_gate already stepped toward the door).
    if not _walk_until_map(emu, reader, LAB_EXIT_DIR, LITTLEROOT):
        print("Phase 2: never exited the lab", flush=True)
        return

    # Cross Littleroot to route_101.
    navigate_to(emu, reader, WallMap(), LITTLEROOT_NORTH_CELL, max_steps=200)
    if not _walk_until_map(emu, reader, "up", ROUTE_101):
        print("Phase 3: never entered route_101", flush=True)
        return

    # Confirm free-roam with a real pos change, then save.
    ref = snapshot_settled(reader)
    for _ in range(8):
        _press(emu, buttons.KEY_UP)
        snap = snapshot_settled(reader)
        if snap is not None and ref is not None and snap.map_id == ROUTE_101 \
                and snap.pos != ref.pos:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(emu.save_state())
            print(
                f"POST-STARTER saved: map {snap.map_id} pos {snap.pos} "
                f"levels {reader.party_levels()} -> {OUT_PATH.resolve()}",
                flush=True,
            )
            return
    print("Phase 3: no free-roam movement confirmed on route_101", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and produce the artifact**

Run:
```bash
cd /Users/_eloi/Projets/Emu && \
POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
/Users/_eloi/Projets/Emu/.venv/bin/python \
/Users/_eloi/Projets/Emu-p4-capture-route101/tools/capture_route101_freeroam.py 2>&1 | tail -10
```
Expected: `POST-STARTER saved: map (0, 16) pos (X,Y) levels [5,...] -> .../states/post_starter.state`.

If a phase prints a failure, re-run `probe_lab_intro.py` to re-read the layout and correct the `CONSTANTS` block (most likely `LAB_EXIT_DIR` or `LITTLEROOT_NORTH_CELL`), then re-run.

- [ ] **Step 3: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101
git add tools/capture_route101_freeroam.py
git commit -m "feat: capture_route101_freeroam — scripted lab-intro walkthrough to the artifact"
```

---

## Task 4: Verify the acceptance test goes green

**Files:**
- Unchanged: `tests/test_campaign_rom.py`

- [ ] **Step 1: Run the gated smoke with the artifact present**

Run:
```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101 && \
POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba \
/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest \
tests/test_campaign_rom.py -v
```
Expected: `test_run_campaign_skips_level_up_and_advances_on_real_rom PASSED` (NOT skipped). This proves the artifact is a valid route_101 free-roam level-5 state and that `run_campaign` drives a real advance on it — the whole point of the follow-up.

If it still SKIPS with "states/post_starter.state missing", the artifact path is wrong: the test resolves `parents[2]/"Emu"/states/post_starter.state`; confirm Task 3 wrote it under the MAIN repo's `states/`.

- [ ] **Step 2: Run the full suite without the ROM to confirm no regressions**

Run:
```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101 && \
/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```
Expected: the same green baseline as `main` (`251 passed, 12 skipped` or with the ROM smoke now un-skipped when the env is set), ruff-clean tools. No failures.

---

## Task 5: Cleanup and record the finding

**Files:**
- Delete: `tools/capture_post_starter.py`
- Delete: `tools/probe_lab_intro.py`
- Modify: `docs/superpowers/specs/2026-08-02-capture-route101-freeroam-design.md`

- [ ] **Step 1: Delete the obsolete + scaffolding tools**

```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101
git rm tools/capture_post_starter.py tools/probe_lab_intro.py
```
Keep `tools/capture_lab_entry.py` and `tools/capture_route101_freeroam.py` — together they document/reproduce the artifact.

- [ ] **Step 2: Record the outcome in the spec**

Append a `## Finding after implementation (2026-08-02)` section to the spec with: the discovered constants (LAB_MAP, LAB_EXIT_DIR, LITTLEROOT_NORTH_CELL), confirmation that the A-spam gate-clear worked (or what replaced it), and that `tests/test_campaign_rom.py` now passes with the artifact. Set the header `Status:` to `implemented`.

- [ ] **Step 3: Lint and commit**

```bash
cd /Users/_eloi/Projets/Emu-p4-capture-route101
/Users/_eloi/Projets/Emu/.venv/bin/ruff check tools/ && \
git add -A && \
git commit -m "$(cat <<'EOF'
chore: drop obsolete capture_post_starter + probe scaffolding, record finding

capture_route101_freeroam produces states/post_starter.state; the run_campaign
ROM smoke is now load-bearing (passes with the artifact).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Finish the branch**

Use the superpowers:finishing-a-development-branch skill to merge back to `main` (the user's standing choice is a local no-ff merge).
