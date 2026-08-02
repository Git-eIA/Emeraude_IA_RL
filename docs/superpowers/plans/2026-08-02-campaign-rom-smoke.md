# run_campaign ROM smoke (plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First load-bearing ROM wiring for `run_campaign` — capture a post-starter overworld savestate, then a gated smoke drives a real `advance` from it (skipping `level_up`).

**Architecture:** Two new files, zero production change. `tools/capture_post_starter.py` (one-shot artifact generator, modeled on `capture_open_map.py`) produces `states/post_starter.state`. `tests/test_campaign_rom.py` (double-skip gated) loads it and asserts `run_campaign` skips `level_up` and advances on the real ROM.

**Tech Stack:** Python 3.12, Stable-Baselines3 PPO, Gymnasium, mGBA (`GbaEmulator`), pytest.

**Reference spec:** `docs/superpowers/specs/2026-08-02-campaign-rom-smoke-design.md`

**Note on TDD:** Both files are emulator/ROM-bound — there is no pure logic to red/green in isolation. Verification is therefore: `ruff` clean, pytest *collects* the new test, the gated smoke *skips cleanly* without the artifact (Tasks 1–2), then a manual capture makes it *pass* load-bearing (Task 3).

---

### Task 1: Capture tool

**Files:**
- Create: `tools/capture_post_starter.py`

- [ ] **Step 1: Write the capture tool**

```python
"""Capture a post-starter overworld savestate by driving the trained Explorer.

Loads the Explorer PPO policy, plays from states/initial.state until the
"starter_obtained" milestone fires, then keeps stepping until the forced wild
battle (Poochyena attacking Birch) clears, and writes states/post_starter.state.
One-time artifact generation: run once locally where checkpoints/ppo_emerald_final.zip
and the ROM exist. The output is gitignored; the run_campaign ROM smoke skips when it
is absent.

Usage:
  POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_post_starter.py --max-steps 8000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/post_starter.state")
SETTLE_FRAMES = 4   # consecutive out-of-battle snapshots before we trust the overworld


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
    settled = 0

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, _, info = env.step(int(action))

        if not got_starter and "starter_obtained" in info["milestones"]:
            got_starter = True
            print(f"starter_obtained at step {step}", flush=True)

        if got_starter:
            # Wait for the forced battle to clear: a run of out-of-battle frames.
            settled = settled + 1 if not reader.in_battle() else 0
            if settled >= SETTLE_FRAMES:
                snap = reader.snapshot()
                OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                OUT_PATH.write_bytes(env.emulator.save_state())
                print(
                    f"POST-STARTER at step {step}: "
                    f"map {snap.map_id if snap else None} "
                    f"pos {snap.pos if snap else None} "
                    f"in_battle {reader.in_battle()} "
                    f"levels {reader.party_levels()} "
                    f"-> {OUT_PATH.resolve()}",
                    flush=True,
                )
                return

    print(
        f"starter not obtained / battle not cleared in {args.max_steps} steps; "
        f"try more steps",
        flush=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `ruff check tools/capture_post_starter.py`
Expected: no errors.

- [ ] **Step 3: Verify it imports and shows help without a ROM**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python tools/capture_post_starter.py --help`
Expected: argparse usage printed, exit 0 (no `KeyError: POKEMON_EMERALD_ROM`, since `--help` short-circuits before env access).

- [ ] **Step 4: Commit**

```bash
git add tools/capture_post_starter.py
git commit -m "feat: capture_post_starter tool — dump a post-starter overworld savestate"
```

---

### Task 2: Gated ROM smoke

**Files:**
- Create: `tests/test_campaign_rom.py`

- [ ] **Step 1: Write the gated smoke**

```python
"""Gated ROM smoke: run_campaign skips level_up and drives a real advance.

Load-bearing when states/post_starter.state exists (captured by
tools/capture_post_starter.py): a level-5 party on the route_101 map lets
run_campaign skip level_up by construction and walk a real advance on the ROM.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_ROM = os.environ.get("POKEMON_EMERALD_ROM")
# states/ lives in the main repo, not the worktree; resolve it absolutely.
_STATE = Path(__file__).resolve().parents[2] / "Emu" / "states" / "post_starter.state"


@pytest.mark.skipif(not _ROM, reason="POKEMON_EMERALD_ROM not set")
@pytest.mark.skipif(not _STATE.exists(), reason="states/post_starter.state missing")
def test_run_campaign_skips_level_up_and_advances_on_real_rom() -> None:
    from emulator.gba import GbaEmulator
    from env.campaign import Milestone, run_campaign
    from env.local_navigator import WallMap
    from env.map_memory import MapMemory
    from env.orders import DESTINATIONS, reached
    from env.world_reader import WorldReader

    emu = GbaEmulator(_ROM)
    emu.load_state(_STATE.read_bytes())
    emu.step(0, 4)  # settle after load_state
    reader = WorldReader(emu.read_bytes)

    start = reader.snapshot()
    assert start is not None
    # Same-map precondition: advance takes travel_to's same-map branch (no portals).
    assert start.map_id == DESTINATIONS["route_101"][0]
    # A >=5 party means run_campaign skips level_up by construction.
    assert reached(reader.party_levels(), 5)

    outcome = run_campaign(
        emu, reader, MapMemory(), WallMap(),
        curriculum=(Milestone("route_101", 5),),
    )

    assert outcome in {"campaign_complete", "unreachable", "left_map", "timeout"}
    assert outcome == "campaign_complete" or reader.snapshot().pos != start.pos
```

- [ ] **Step 2: Lint**

Run: `ruff check tests/test_campaign_rom.py`
Expected: no errors.

- [ ] **Step 3: Verify it collects and skips cleanly (no ROM, no artifact in worktree)**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign_rom.py -v`
Expected: `1 skipped` (reason: `POKEMON_EMERALD_ROM not set`), collection succeeds — no import/collection errors.

- [ ] **Step 4: Verify the pure suite is unchanged**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: same pure-suite count as before plus this skip (e.g. `... passed, N skipped`), no failures.

- [ ] **Step 5: Commit**

```bash
git add tests/test_campaign_rom.py
git commit -m "test: gated ROM smoke — run_campaign skips level_up and advances"
```

---

### Task 3: Produce the artifact and make the smoke load-bearing (manual, run in the main repo)

**Files:** none (produces the gitignored `states/post_starter.state`).

This runs in the **main repo** `/Users/_eloi/Projets/Emu` where the ROM, checkpoint, and `states/` live (they are not in the worktree). The worktree test resolves the state via an absolute path, so a state captured in the main repo is visible to the worktree smoke.

- [ ] **Step 1: Capture the savestate**

From `/Users/_eloi/Projets/Emu`, using the tool file from the worktree (or after merge):

```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
  /Users/_eloi/Projets/Emu/.venv/bin/python \
  /Users/_eloi/Projets/Emu-p4-campaign-rom-smoke/tools/capture_post_starter.py --max-steps 8000
```

Expected: a `POST-STARTER at step N:` line printing `map (0, 16)`, a `pos`, `in_battle False`, and `levels [5]` (or `[5, ...]`), and the written path. If it prints "starter not obtained / battle not cleared", re-run with a larger `--max-steps`.

- [ ] **Step 2: Confirm the captured state is sane**

Verify from the printed line: `map (0, 16)` (route_101), `in_battle False` (overworld, not mid-battle), and the party mean level is ≥ 5. If `map` is not `(0, 16)` or `in_battle` is `True`, re-run (increase `--max-steps`); do not proceed with a bad artifact.

- [ ] **Step 3: Run the smoke load-bearing (from the worktree)**

```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest \
  /Users/_eloi/Projets/Emu-p4-campaign-rom-smoke/tests/test_campaign_rom.py -v
```

Expected: `1 passed` (NOT skipped) — the smoke is now load-bearing.

- [ ] **Step 4: Full suite with ROM (from the worktree)**

```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
  /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```

Expected: all prior tests pass, plus this one now passing, no regressions.

---

## Self-Review

- **Spec coverage:** capture tool (Task 1) ↔ spec §Architecture 1; gated smoke with all 8 assertion steps (Task 2) ↔ spec §Architecture 2 + §Why load-bearing; manual capture + load-bearing verification (Task 3) ↔ spec §Files + §Testing acceptance. Covered.
- **Placeholders:** none — both files are complete.
- **Type/name consistency:** `WallMap` from `env.local_navigator`, `MapMemory` from `env.map_memory`, `reached`/`DESTINATIONS` from `env.orders`, `Milestone`/`run_campaign` from `env.campaign`, `WorldReader` from `env.world_reader`, `PokemonEmeraldEnv` from `env.pokemon_env`, `GbaEmulator` from `emulator.gba` — all verified against the codebase. `info["milestones"]` is a sorted list of fired names; `starter_obtained` fires on `party_count >= 1`. `snapshot().map_id` is the `(map_group, map_num)` tuple; `DESTINATIONS["route_101"][0] == (0, 16)`. Same-map advance outcome set `{arrived→campaign_complete, unreachable, left_map, timeout}` matches `navigate_to`.
