# Scripted Rival Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the throwaway tool `tools/capture_trainer_battle.py` so it spawns the
flag-gated route_103 rival and, on talk, mints `states/trainer_battle.state` — making two
skipped ROM smokes load-bearing.

**Architecture:** A single self-contained driver that (per attempt) loads
`states/route_103_reached.state`, clears hide-flag `0x0382` in RAM, forces a map reload so the
rival object respawns at `(7,3)`, navigates sand to a cell adjacent to it (Fighter clears
wilds), and spams A through dialogue until a trainer battle starts. A bounded retry loop
re-runs the whole attempt with a fresh state load + RNG perturbation to absorb wild-encounter
attrition. Success writes the artifact and exits 0; exhausting attempts exits 1 with
diagnostics (→ escalate to the deferred Option B story-flag path).

**Tech Stack:** Python 3.12, mGBA (`emulator.gba.GbaEmulator`), stable-baselines3 PPO
(Fighter), the ledge-aware grid nav (`env.grid_navigator`, `env.grid_snapshot`).

**How this is verified:** This is a live-ROM integration driver, not unit-testable with
fakes. Each task's verification runs the tool (or the partial script) and observes its prints.
The terminal proof is the produced artifact + the two smokes going load-bearing (Task 6).

**CRITICAL — worktree vs artifacts.** The edited tool lives in the worktree
(`/Users/_eloi/Projets/Emu-scripted-rival-capture`), but the ROM, states, checkpoints, and
`.venv` are gitignored and exist ONLY in the main repo (`/Users/_eloi/Projets/Emu`). To run
the worktree's edited tool against the real artifacts, symlink the artifact dirs into the
worktree ONCE before any verification, then run everything from the worktree:

```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
for d in roms states checkpoints .venv; do ln -sfn /Users/_eloi/Projets/Emu/$d $d; done
```

These symlinks are inside the worktree; `roms/ states/ checkpoints/ .venv/` are already
gitignored so they will not be committed. Do this symlink setup as the first action of Task 1.

**Runtime prerequisites (real files in the main repo, reached via the symlinks above):**
- `roms/pokemon_emerald_fr.gba` (env var `POKEMON_EMERALD_ROM`)
- `states/route_103_reached.state`
- `checkpoints/fighter/ppo_fighter_final.zip`

**Canonical run command (used by every task's verification, from the worktree):**
```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture \
  && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
     .venv/bin/python tools/capture_trainer_battle.py
```

Every `cd /Users/_eloi/Projets/Emu && ...` run/pytest command in the tasks below should be run
from the worktree `/Users/_eloi/Projets/Emu-scripted-rival-capture` instead (the symlinks make
`roms/ states/ checkpoints/ .venv/` resolve to the main repo's real files).

**Design decision — fallback (supersedes spec's teleport wording):** The spec permitted the
plan to replace the fragile RAM-teleport fallback with "retry reload+nav with fresh RNG". This
plan does exactly that. RAM-teleporting the player (writing `gObjectEvents[0].currentCoords`)
desyncs camera/collision/`SaveBlock1.pos`; a whiteout also faints the party, so the state must
be reloaded fresh regardless. The retry loop (Task 5) subsumes both attrition and transient
failure. No teleport is implemented.

---

## File Structure

- **Rewrite (single file):** `tools/capture_trainer_battle.py` — the whole driver. The file is
  built up across Tasks 1-5 (helpers → reload → nav → talk → retry wrapper), then verified
  end-to-end in Task 6.
- **Produces:** `states/trainer_battle.state` (gitignored).
- **Unblocks (no edit):** `tests/test_battle_player_rom.py::test_fighter_wins_a_real_trainer_battle`,
  `tests/test_north_rival_milestones_rom.py`.

The existing `tools/capture_trainer_battle.py` (naive d-pad cycling, never worked because the
rival is flag-gated) is fully replaced. Its two consumers reference only the output path
`states/trainer_battle.state`, which is unchanged.

---

## Task 1: Scaffold — imports, constants, RAM flag-clear, one-attempt skeleton

**Files:**
- Rewrite: `tools/capture_trainer_battle.py`

- [ ] **Step 1: Replace the whole file with the scaffold**

```python
"""Throwaway capture tool: spawn the flag-gated route_103 rival and mint trainer_battle.state.

The rival (gObjectEvents template obj[10]: gfx 0x40, tile (7,3), trainerType 0) is HIDDEN by
SaveBlock1 hide-flag 0x0382. This tool clears that flag in RAM (cheat-spawn, for the ARTIFACT
only), forces a map reload so the object respawns, navigates the sand to a cell adjacent to
(7,3), and spams A through any dialogue until a trainer battle starts -- then saves
states/trainer_battle.state. A bounded retry loop reloads the state fresh (party restored) and
perturbs RNG to absorb wild-encounter attrition on the northward trip.

The legit path (Birch's post-lab dialog clearing 0x0382) is the deferred scripted campaign
(Option B); this tool only produces the artifact so the two trainer-battle ROM smokes become
load-bearing.

Run in the MAIN repo:
  POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
    .venv/bin/python tools/capture_trainer_battle.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.train_fighter import make_move_type_fn
from emulator import buttons
from emulator.gba import GbaEmulator
from env.game_state import SAVE_BLOCK1_PTR, BattleReader
from env.grid_navigator import (
    BATTLE_TRANSITION_SETTLE,
    DELTAS,
    RELEASE_FRAMES,
    handle_battle_interruption,
    plan_path_grid,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind
from env.world_reader import WorldReader

INPUT_STATE = "states/route_103_reached.state"
OUTPUT_STATE = "states/trainer_battle.state"
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"

ROUTE_103 = (0, 18)
RIVAL_FLAG = 0x0382
RIVAL_TILE = (7, 3)

_FLAGS_OFF = 0x1270           # SaveBlock1.flags[] offset (Emerald)
_STANDABLE = {TileKind.FREE, TileKind.GRASS}
_LOST = ("battle_lost", "battle_timeout", "battle_interrupted")

_RELOAD_BUDGET = 30           # bounded map-crossing loop
_NAV_MAX = 250               # bounded nav loop
_TALK_A_PRESSES = 30         # A-spam budget through pre-battle dialogue
_MAX_ATTEMPTS = 5            # retry-RNG budget
_RNG_PERTURB_FRAMES = 17     # idle frames * attempt to vary wild rolls


def _clear_flag(emu, flag_id: int) -> int:
    """Clear a SaveBlock1 flag bit in RAM; return the bit value read back (0 = cleared)."""
    ptr = int.from_bytes(emu.read_bytes(SAVE_BLOCK1_PTR, 4), "little")
    addr = ptr + _FLAGS_OFF + flag_id // 8
    val = emu.read_bytes(addr, 1)[0] & ~(1 << (flag_id % 8))
    emu._core._core.rawWrite8(emu._core._core, addr, -1, val)
    return (emu.read_bytes(addr, 1)[0] >> (flag_id % 8)) & 1


def main() -> int:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    emu = GbaEmulator(rom)
    with open(INPUT_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = WorldReader(emu.read_bytes)

    here = reader.snapshot()
    print(f"start map={here.map_id} pos={here.pos}")
    bit = _clear_flag(emu, RIVAL_FLAG)
    print(f"cleared flag 0x{RIVAL_FLAG:04x} -> bit now {bit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run to verify load + flag-clear**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python tools/capture_trainer_battle.py`
Expected: prints `start map=(0, 18) pos=(10, 21)` then `cleared flag 0x0382 -> bit now 0`.

- [ ] **Step 3: Commit**

```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
git add tools/capture_trainer_battle.py
git commit -m "feat: capture tool scaffold — load state, clear rival hide-flag in RAM"
```

---

## Task 2: Map reload + spawn scan

**Files:**
- Modify: `tools/capture_trainer_battle.py`

- [ ] **Step 1: Add the reload helper and object scan above `main`**

```python
def _step_until_map(emu, reader, key, *, want_equal=None, want_change_from=None):
    """Hold `key` frame-stepping until map_id reaches/leaves the target. Bounded."""
    last = snapshot_settled(reader)
    for _ in range(_RELOAD_BUDGET):
        emu.step(key, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES)
        here = snapshot_settled(reader)
        if here is None:
            continue
        last = here
        if want_equal is not None and here.map_id == want_equal:
            return here
        if want_equal is None and here.map_id != want_change_from:
            return here
    return last


def _reload_route_103(emu, reader):
    """Leave route_103 south into Oldale then re-enter north; respawns object events."""
    off = _step_until_map(emu, reader, buttons.KEY_DOWN, want_change_from=ROUTE_103)
    print(f"  stepped south -> map={off.map_id if off else None}")
    if off is None or off.map_id == ROUTE_103:
        return None
    back = _step_until_map(emu, reader, buttons.KEY_UP, want_equal=ROUTE_103)
    print(f"  stepped north -> map={back.map_id if back else None} pos={back.pos if back else None}")
    return back if (back is not None and back.map_id == ROUTE_103) else None


def _scan_objects(emu):
    """Print live gObjectEvents slots (gfx + tile) so the rival spawn can be eyeballed."""
    blob = emu.read_bytes(0x02037350, 0x24 * 16)
    for i in range(16):
        o = i * 0x24
        if not blob[o] & 1:
            continue
        gx = int.from_bytes(blob[o + 0x10:o + 0x12], "little", signed=True)
        gy = int.from_bytes(blob[o + 0x12:o + 0x14], "little", signed=True)
        tag = " <-- RIVAL GFX" if blob[o + 5] == 0x40 else ""
        print(f"    live obj slot={i} gfx=0x{blob[o+5]:02x} tile=({gx-7},{gy-7}){tag}")
```

- [ ] **Step 2: Call the reload + scan from `main` (replace the `return 0` block)**

```python
    back = _reload_route_103(emu, reader)
    if back is None:
        print("FAILED to reload route_103; aborting")
        return 1
    _scan_objects(emu)
    return 0
```

- [ ] **Step 3: Run to verify the reload + rival spawn**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python tools/capture_trainer_battle.py`
Expected: prints a south then north crossing landing back on `map=(0, 18)`. NOTE: the rival at
`(7,3)` may NOT appear in the scan yet — objects only populate near the camera and the landing
cell is the south edge (y~21). This is expected; the scan is diagnostic. The pass condition
here is only that `_reload_route_103` returns route_103 (no "FAILED to reload" line).

- [ ] **Step 4: Commit**

```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
git add tools/capture_trainer_battle.py
git commit -m "feat: force map reload (south then north) so the rival object respawns"
```

---

## Task 3: Navigate the sand to a cell adjacent to (7,3)

**Files:**
- Modify: `tools/capture_trainer_battle.py`

- [ ] **Step 1: Add nav helpers above `main`**

```python
def _grass_blocked(snap):
    """Directed edges whose target tile is GRASS (grass allowed but discouraged via cost)."""
    blocked = set()
    for y in range(snap.height):
        for x in range(snap.width):
            if snap.classify_at(x, y) not in _STANDABLE:
                continue
            for direction, (dx, dy) in DELTAS.items():
                if snap.classify_at(x + dx, y + dy) is TileKind.GRASS:
                    blocked.add(((x, y), direction))
    return blocked


def _adjacent_targets(snap, tile):
    """Standable cells 4-adjacent to `tile`, each with the heading that faces `tile`."""
    out = []
    for direction, (dx, dy) in DELTAS.items():
        cell = (tile[0] - dx, tile[1] - dy)
        if snap.classify_at(*cell) in _STANDABLE:
            out.append((cell, direction))
    return out


def _pick_stand_cell(reader, here):
    """Shortest grass-avoiding path to a cell adjacent to the rival. Returns (cell, facing)."""
    snap = GridSnapshot.from_reader(reader.grid_reader, here.map_id)
    grass = _grass_blocked(snap)
    best = None
    for cell, facing in _adjacent_targets(snap, RIVAL_TILE):
        path = plan_path_grid(snap, here.pos, cell, blocked=grass)
        if path is not None and (best is None or len(path) < best[2]):
            best = (cell, facing, len(path))
    if best is None:
        for cell, facing in _adjacent_targets(snap, RIVAL_TILE):   # grass-allowed fallback
            path = plan_path_grid(snap, here.pos, cell)
            if path is not None and (best is None or len(path) < best[2]):
                best = (cell, facing, len(path))
    return (best[0], best[1]) if best else (None, None)


def _navigate(emu, reader, battle, mtf, predict, stand_cell):
    """Walk to stand_cell, letting the Fighter clear wilds. Returns None on arrival else a
    losing outcome string."""
    presses = 0
    while presses < _NAV_MAX:
        here = snapshot_settled(reader)
        if here is None:
            emu.step(0, RELEASE_FRAMES)
            presses += 1
            continue
        if battle.battle_starting():
            for _ in range(BATTLE_TRANSITION_SETTLE):
                if reader.in_battle():
                    break
                emu.step(0, RELEASE_FRAMES)
            wild = handle_battle_interruption(emu, reader, mtf, predict)
            print(f"    wild at {here.pos} -> {wild}")
            if wild in _LOST:
                return wild
            continue
        if here.pos == stand_cell:
            return None
        snap = GridSnapshot.from_reader(reader.grid_reader, here.map_id)
        path = plan_path_grid(snap, here.pos, stand_cell)
        if not path:
            print(f"    lost path from {here.pos} to {stand_cell}")
            return "battle_interrupted"
        probe_step(emu, reader, here, path[0])
        presses += 1
    return "battle_timeout"
```

- [ ] **Step 2: Wire Fighter + BattleReader + nav into `main` (replace the scan/return tail)**

```python
    battle = BattleReader(emu.read_bytes)
    from stable_baselines3 import PPO
    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])
    mtf = make_move_type_fn(emu)

    _scan_objects(emu)
    here = snapshot_settled(reader)
    stand_cell, facing = _pick_stand_cell(reader, here)
    if stand_cell is None:
        print(f"no path to a cell adjacent to {RIVAL_TILE}; aborting")
        return 1
    print(f"navigate {here.pos} -> stand {stand_cell} face {facing}")
    lost = _navigate(emu, reader, battle, mtf, predict, stand_cell)
    if lost is not None:
        print(f"END nav {lost}")
        return 1
    print(f"arrived at {stand_cell}")
    return 0
```

- [ ] **Step 3: Run to verify arrival adjacent to the rival**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python tools/capture_trainer_battle.py`
Expected: after reload, prints `navigate ... -> stand (x,y) face <dir>` then `arrived at (x,y)`
where `(x,y)` is 4-adjacent to `(7,3)`. A wild or two may be cleared en route (`wild ... -> won`).
If it ends `END nav battle_lost/timeout`, that is the attrition case Task 5 retries — acceptable
for this task as long as the nav machinery ran.

- [ ] **Step 4: Commit**

```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
git add tools/capture_trainer_battle.py
git commit -m "feat: navigate sand to a cell adjacent to the rival, Fighter clears wilds"
```

---

## Task 4: Talk through dialogue + save artifact + exit codes

**Files:**
- Modify: `tools/capture_trainer_battle.py`

- [ ] **Step 1: Add the talk helper above `main`**

```python
def _talk_until_battle(emu, reader, battle, facing):
    """Face the rival, then spam A through pre-battle dialogue until a battle starts."""
    here = snapshot_settled(reader)
    probe_step(emu, reader, here, facing)   # turn to face (7,3)
    for _ in range(_TALK_A_PRESSES):
        emu.step(buttons.KEY_A, RELEASE_FRAMES)
        emu.step(0, RELEASE_FRAMES)
        if battle.battle_starting():
            return True
    return battle.battle_starting()
```

- [ ] **Step 2: Replace the `arrived` tail of `main` with talk + save**

```python
    print(f"arrived at {stand_cell}; facing {facing}, spamming A")
    if _talk_until_battle(emu, reader, battle, facing) and battle.is_trainer_battle():
        Path(OUTPUT_STATE).write_bytes(emu.save_state())
        print(f"END rival_confirmed -> saved {OUTPUT_STATE}")
        return 0
    print(f"END no_trainer_battle starting={battle.battle_starting()} "
          f"trainer={battle.is_trainer_battle()} in_battle={reader.in_battle()}")
    return 1
```

- [ ] **Step 3: Run to verify the artifact is produced (single-attempt)**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python tools/capture_trainer_battle.py; ls -la states/trainer_battle.state`
Expected (happy path): `END rival_confirmed -> saved states/trainer_battle.state` and the file
exists. **If instead** `END no_trainer_battle starting=False ...`: the sprite spawned but talking
did not start a battle — this is the spec's named risk (spawn ≠ battle-script active). Record
the exact diagnostic line and STOP; escalate per the spec's Risks section (Option B), do not
patch further. **If** `END nav battle_lost/timeout`: attrition — Task 5's retry loop addresses it.

- [ ] **Step 4: Commit**

```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
git add tools/capture_trainer_battle.py
git commit -m "feat: talk through dialogue, confirm trainer battle, save trainer_battle.state"
```

---

## Task 5: Retry-RNG wrapper (fresh state load per attempt)

**Files:**
- Modify: `tools/capture_trainer_battle.py`

- [ ] **Step 1: Extract one attempt into `_attempt`, add the retry loop in `main`**

Refactor: move everything currently in `main` after the ROM read into a new function
`_attempt(rom, attempt) -> int` that builds a FRESH emulator each call (load `INPUT_STATE`,
perturb RNG by `attempt * _RNG_PERTURB_FRAMES` idle frames before clearing the flag), and
returns 0 on success, 1 otherwise. `main` becomes the retry loop.

```python
def _attempt(rom: str, attempt: int) -> int:
    emu = GbaEmulator(rom)
    with open(INPUT_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4 + attempt * _RNG_PERTURB_FRAMES)   # vary wild rolls per attempt
    reader = WorldReader(emu.read_bytes)

    here = reader.snapshot()
    print(f"[attempt {attempt}] start map={here.map_id} pos={here.pos}")
    bit = _clear_flag(emu, RIVAL_FLAG)
    print(f"  cleared flag 0x{RIVAL_FLAG:04x} -> bit now {bit}")

    back = _reload_route_103(emu, reader)
    if back is None:
        print("  FAILED to reload route_103")
        return 1

    battle = BattleReader(emu.read_bytes)
    from stable_baselines3 import PPO
    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs) -> int:
        return int(model.predict(obs, deterministic=True)[0])
    mtf = make_move_type_fn(emu)

    _scan_objects(emu)
    here = snapshot_settled(reader)
    stand_cell, facing = _pick_stand_cell(reader, here)
    if stand_cell is None:
        print(f"  no path to a cell adjacent to {RIVAL_TILE}")
        return 1
    print(f"  navigate {here.pos} -> stand {stand_cell} face {facing}")
    lost = _navigate(emu, reader, battle, mtf, predict, stand_cell)
    if lost is not None:
        print(f"  nav {lost}")
        return 1

    print(f"  arrived at {stand_cell}; facing {facing}, spamming A")
    if _talk_until_battle(emu, reader, battle, facing) and battle.is_trainer_battle():
        Path(OUTPUT_STATE).write_bytes(emu.save_state())
        print(f"  rival_confirmed -> saved {OUTPUT_STATE}")
        return 0
    print(f"  no_trainer_battle starting={battle.battle_starting()} "
          f"trainer={battle.is_trainer_battle()} in_battle={reader.in_battle()}")
    return 1


def main() -> int:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    for attempt in range(_MAX_ATTEMPTS):
        if _attempt(rom, attempt) == 0:
            print(f"END success on attempt {attempt}")
            return 0
    print(f"END failed after {_MAX_ATTEMPTS} attempts (attrition or spawn-not-battling)")
    return 1
```

Delete the now-unused single-attempt body from the old `main` and the module-level PPO import
if any remains (PPO is imported inside `_attempt`).

- [ ] **Step 2: Run the full retry loop**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python tools/capture_trainer_battle.py; echo "exit=$?"; ls -la states/trainer_battle.state`
Expected: `END success on attempt N` (N in 0.._MAX_ATTEMPTS-1), `exit=0`, and
`states/trainer_battle.state` exists. If every attempt prints `no_trainer_battle starting=False`,
the spawn-not-battling risk is realized → escalate to Option B (do not loop-tune further).

- [ ] **Step 3: Commit**

```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
git add tools/capture_trainer_battle.py
git commit -m "feat: bounded retry-RNG loop — fresh state load per attempt absorbs attrition"
```

---

## Task 6: Make the two ROM smokes load-bearing

**Files:**
- Verify (no edit): `tests/test_battle_player_rom.py`, `tests/test_north_rival_milestones_rom.py`

- [ ] **Step 1: Confirm the artifact exists (from Task 5)**

Run: `cd /Users/_eloi/Projets/Emu && ls -la states/trainer_battle.state`
Expected: the file exists (produced by the successful tool run).

- [ ] **Step 2: Run the previously-skipped trainer-battle smoke**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python -m pytest tests/test_battle_player_rom.py::test_fighter_wins_a_real_trainer_battle -v`
Expected: PASS (no longer skipped) — the Fighter beats the real rival, `in_battle` False after.

- [ ] **Step 3: Run the north-rival milestones smoke**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python -m pytest tests/test_north_rival_milestones_rom.py -v`
Expected: PASS — the route_103 latch / beat_rival half now runs with the artifact present.

- [ ] **Step 4: Run the full suite for regressions**

Run: `cd /Users/_eloi/Projets/Emu && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python -m pytest -q`
Expected: all pass (the two smokes now execute instead of skipping); ruff clean:
`cd /Users/_eloi/Projets/Emu && .venv/bin/ruff check tools/capture_trainer_battle.py`.

- [ ] **Step 5: Commit any test-status notes (if the plan touched tests) — otherwise skip**

No test edits are expected; the smokes were already written to become load-bearing when the
artifact appears. If a smoke needed a trivial fix (e.g. a stale skip guard), commit it:
```bash
cd /Users/_eloi/Projets/Emu-scripted-rival-capture
git add tests/ && git commit -m "test: unskip trainer-battle smokes now that the artifact exists"
```

---

## Self-Review

- **Spec coverage:** flag-clear (T1) · reload/respawn (T2) · sand nav w/ Fighter (T3) · talk
  through dialogue + save + exit codes (T4) · attrition fallback as retry-RNG (T5, per spec's
  permitted substitution) · both smokes load-bearing (T6) · spawn-not-battling risk handled as
  an explicit STOP→Option B in T4/T5. All spec sections mapped.
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `_attempt(rom, attempt)->int`, `_navigate(...)->str|None`,
  `_pick_stand_cell(...)->tuple(cell,facing)|(None,None)`, `_talk_until_battle(...)->bool`,
  `_clear_flag(...)->int` used consistently; `_LOST` naming stable.
