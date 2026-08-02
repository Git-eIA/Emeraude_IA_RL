# Capture route_101 free-roam savestate — design spec

**Date:** 2026-08-02
**Status:** implemented (2026-08-02)
**Follow-up to:** P4 étape 7 (run_campaign ROM smoke — descoped to gated plumbing, merge `a41cb75`)

## Problem

`tests/test_campaign_rom.py` is a gated ROM smoke that exercises `run_campaign`
against the real emulator. It needs a savestate where the player is in **free-roam
on route_101 `(0, 16)` with a level-5 party**. No such savestate exists, and P4
étape 7 proved the trained Explorer never naturally sits in that state:

- `starter_obtained` fires the instant `party_count >= 1` — during the bag-grab
  cutscene on `(0, 16)`, where the player cannot walk.
- The forced wild Poochyena battle starts a few steps later.
- After the battle the player is **warped to Birch's lab, map `(1, 4)`**, where
  `reader.in_battle()` returns a **false-positive `True`** the whole time.
- The Explorer's episode is **terminal at `starter_obtained`**; post-starter is
  off-distribution, and the lab has **script gates** (must talk to Birch for the
  Pokédex before leaving) that a wandering policy never clears.

So the smoke is currently a documented skip. This follow-up produces the missing
artifact `states/post_starter.state` so the smoke becomes load-bearing.

## Goal

Produce `states/post_starter.state`: party level 5, free-roam on route_101
`(0, 16)`, via a scripted walkthrough of the lab intro. The walkthrough is NOT
fully deterministic — Phase 0 reuses the stochastic Explorer policy (`deterministic=False`,
~9-10/10 reaches the starter) so the tool wraps the whole run in a bounded retry
loop, restarting from `initial.state` if a phase fails. The existing smoke
`tests/test_campaign_rom.py` is **unchanged** — it simply turns green once the
artifact exists.

## Approach (decided)

**A — Script the lab-intro walkthrough** (chosen over B "retarget to a reachable
state" and C "train a further-progressed Explorer"). Rationale: A is the only
option that yields the exact semantic state the smoke was written for, without
moving the goalposts (B keeps the `in_battle` false-positive and isn't a real
navigation destination) or a training campaign (C). Brittleness is confined to a
one-shot capture tool; we already have `capture_post_starter.py` as scaffolding.

## Deliverable

A disposable capture tool `tools/capture_route101_freeroam.py` that drives the
**raw emulator** + `WorldReader` (NOT `PokemonEmeraldEnv`, which terminates at the
starter milestone) through four phases, and writes `states/post_starter.state`
when the player is confirmed free-roam on route_101. The whole four-phase run sits
inside a **bounded retry loop** (`for _ in range(MAX_ATTEMPTS)`): if any phase
fails to reach its exit condition within its own bounded budget, the tool reloads
`initial.state` and retries. Every phase loop is bounded (code-safety #2).

It supersedes `tools/capture_post_starter.py`. Decision: keep the old tool until
the new one produces the artifact and the smoke passes, then delete it in the same
branch (it carries a now-obsolete KNOWN LIMITATION note).

Fighter wiring for Phase 1: the tool loads the Fighter PPO checkpoint to build
`predict: Callable[[np.ndarray], int]` and the type-chart `move_type_fn` (same
wiring `env/battle_player.py` consumers already use), then passes both to
`play_battle`. SB3/torch are imported inside the tool, not at module import of the
env package.

## The four phases

### Phase 0 — truck → starter
Run the trained Explorer (`checkpoints/ppo_emerald_final`) from `initial.state`
until `starter_obtained` is observed, within a bounded step budget. Reuses the
proven policy (9-10/10 reliable); scripting the intro by hand would be absurd. If
the budget is exhausted without the starter, the phase fails and the outer retry
loop restarts from `initial.state`.

Plumbing detail (resolved at implementation): the milestone signal comes from the
same `MilestoneTracker` the env uses, driven directly on the emulator, OR Phase 0
runs through the milestone env just until `starter_obtained` fires and then the
raw emulator is reused for phases 1-3. Either is acceptable; the constraint is that
phases 1-3 must NOT go through the terminating env.

### Phase 1 — forced Poochyena battle
`play_battle(emulator, move_type_fn, predict)` with the trained Fighter checkpoint
until the battle resolves (`won`). (Decided: A2 — delegate to the Fighter rather
than hardcoding battle inputs, since we already have a 10/10 Fighter.)

RISK (verify at smoke): the Fighter was trained on the 5 normal wild battles. The
forced tutorial battle is story-framed ("Birch is in trouble") but mechanically a
normal battle. If a forced dialogue frame breaks the Fighter's observation, fall
back to hardcoded first-move spam (the level-5 starter over-determines the win vs a
level-2 Poochyena). This fallback is only cut in if the smoke shows `play_battle`
misbehaving.

### Phase 2 — lab (the brittle part)
After the battle the player is warped into Birch's lab, where a **cutscene may
auto-trigger** (Birch hands over the Pokédex and Poké Balls without the player
walking to him). The manual probe (below) determines whether Phase 2 is:

- **auto-dialogue** (likely): press A in a bounded loop to advance the cutscene
  until the script gate clears — no `navigate_to` toward Birch needed; or
- **walk-to-Birch**: `navigate_to` toward Birch's hardcoded cell, then press A.

Either way, the exit condition is the **script gate clearing**, detected by a
concrete signal the probe must capture — NOT a fixed press count. Candidate
signals, resolved by the probe: a test d-pad press changes `pos` again (player
regained control), or the RAM `TOWN_STATE` var advances. Once controllable,
`navigate_to` toward the exit door cell to leave the lab.

`navigate_to` is called **without `memory`** so the only branch that reads
`in_battle()` (the `has_grass` learning) is short-circuited — the `(1, 4)`
false-positive does not affect pathfinding. Lab cells are hardcoded from the manual
probe (decided: A over reading the pokeemerald decomp or auto-`map_map`), each with
a `# NOTE:` recording its origin.

### Phase 3 — lab → Littleroot → route_101 (INTER-map)
The lab exit does NOT land on route_101. Leaving the lab warps the player to
**Littleroot Town `(0, 9)`**; route_101 `(0, 16)` is a **separate map** to the
north. This is a multi-map crossing, and `navigate_to` is intra-map only, so it
cannot be done in a single call. `memory` is empty (no learned portals), so
`travel_to` has no route either.

Phase 3 is therefore a **scripted edge-crossing sequence**, each leg bounded:
1. `navigate_to` toward Littleroot's north edge cell (hardcoded from the probe).
2. Press the north d-pad until the map transition fires (`snapshot().map_id`
   changes to route_101 `(0, 16)`), re-snapshotting through the SaveBlock
   relocation frames.
3. Confirm free-roam on route_101 with a real `pos` change, then
   `emulator.save_state()` → `states/post_starter.state`.

The probe must therefore also capture: the Littleroot north-edge cell, and confirm
the lab-exit → Littleroot → route_101 map-id chain `(1, ?) → (0, 9) → (0, 16)`
(the lab map-id `(1, 4)` is from an earlier trajectory trace and must be
re-confirmed by the probe).

## Technical risk — the `in_battle()` false-positive on `(1, 4)`

Two places to verify at the ROM smoke:

- **End of Phase 1:** `play_battle` loops on `outcome != 0 or not in_battle`.
  After the real win, the battle `outcome` flag goes terminal and `play_battle`
  returns `"won"` **before** the warp to `(1, 4)` activates the false-positive.
  The exit is driven by the outcome flag, not by `in_battle`, so this should be
  safe — to be confirmed empirically.
- **Phases 2-3:** `navigate_to` is called without `memory`; its core loop does not
  gate on `in_battle` (only the memory-learning branch does). Pathfinding on
  `(1, 4)` therefore works despite the false-positive. Documented in the tool.

## Prerequisite step (part of the plan, not shipped code)

A one-shot manual instrumented probe (walk the post-battle intro by hand, logging
`map_id`/`pos`/`TOWN_STATE`/`in_battle` on every change) that resolves ALL the
open unknowns before the driver is written:

1. **Lab map-id** — confirm/correct the lab map `(1, 4)` from the earlier trace.
2. **Phase 2 mode** — is the Birch cutscene auto-triggered (press A only) or does
   it require walking to Birch? If walk-to-Birch, his cell.
3. **Gate-clear signal** — how to detect the player regained control (test-press
   changes `pos`, or which `TOWN_STATE` value).
4. **Lab exit door cell** — the cell that warps out of the lab.
5. **Map chain** — confirm lab-exit lands on Littleroot `(0, 9)`, and the
   Littleroot **north-edge cell** whose north press transitions to route_101
   `(0, 16)`.

All of these are hardcoded into the driver with `# NOTE:` origin comments.

## Non-goals

- No multi-ROM generalisation — coordinates are BPEF (French Emerald) hardcoded.
- No retraining, no new milestones.
- `tests/test_campaign_rom.py` is not modified; it turns green with the artifact.
- `states/post_starter.state` is gitignored (a local artifact); the smoke stays
  gated (skips without it).

## Deliverables checklist

- Manual instrumented probe run resolving the 5 unknowns above (prerequisite).
- `tools/capture_route101_freeroam.py` — the four-phase capture driver, wrapped in
  a bounded retry loop, all phase loops bounded.
- Hardcoded lab/Littleroot cells + gate-clear signal with `# NOTE:` origin.
- `tools/capture_post_starter.py` deleted once the new tool works.
- `states/post_starter.state` generated locally (gitignored).
- `tests/test_campaign_rom.py` passing with the artifact present.

## Finding after implementation (2026-08-02)

The scripted walkthrough works and produces the artifact. `tests/test_campaign_rom.py`
now **passes** (was a documented skip) with `states/post_starter.state` present, and
the full suite stays green (251 passed, 12 skipped without the ROM).

Discovered constants (from `probe_lab_intro.py`, over `states/lab_entry.state`):

- **Lab map-id:** `(1, 4)` — confirmed (matched the earlier trajectory trace).
- **Phase 2 mode:** auto-dialogue, cleared by **A-spam only** (no walk-to-Birch). The
  intro dialogue is LONG — ~600 A-presses (Pokédex + Poké Balls + rival naming). The
  driver budgets 2000 A-presses and probes a test move every 200 to detect regained
  control. The earlier spec estimate of "a bounded loop" was right in shape but the
  count is an order of magnitude larger than a naïve guess (the first draft used 40).
- **Gate-clear signal:** a **test d-pad press that changes `pos`** (the RAM `in_battle`
  false-positive persists the whole time and is never gated on — confirmed).
- **Lab exit:** walk **DOWN** from ~`(6, 12)` → warps to **Littleroot `(0, 9)`**,
  landing at pos `(7, 16)`.
- **Phase 3 (inter-map) correction:** the **x=7 column in Littleroot is the lab-door
  re-entry warp** — pressing UP from x=7 re-enters the lab. `navigate_to` (planned) was
  therefore replaced by a proven scripted sequence: press **RIGHT 3 times** to reach the
  x=10 column, THEN walk **UP** → route_101 `(0, 16)` at `(10, 19)`. This avoids the A*
  optimistic-grid routing up the warp column.
- **Map chain confirmed:** lab `(1, 4)` → Littleroot `(0, 9)` → route_101 `(0, 16)`.

Deliverable shape differs slightly from the plan: instead of one four-phase driver with
an outer retry loop, the slow stochastic intro (Phases 0-1) was factored into a separate
cache tool `tools/capture_lab_entry.py` (Explorer → starter → Fighter wins the forced
battle → `states/lab_entry.state`), so `capture_route101_freeroam.py` (Phases 2-3) is
fast and deterministic from the cache. `capture_post_starter.py` and the disposable
`probe_lab_intro.py` were deleted; the two remaining tools reproduce the artifact.
