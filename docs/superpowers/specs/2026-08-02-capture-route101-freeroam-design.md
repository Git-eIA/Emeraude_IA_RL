# Capture route_101 free-roam savestate — design spec

**Date:** 2026-08-02
**Status:** design approved, spec under review
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
`(0, 16)`, via a deterministic scripted walkthrough of the lab intro. The existing
smoke `tests/test_campaign_rom.py` is **unchanged** — it simply turns green once
the artifact exists.

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
when the player is confirmed free-roam on route_101.

## The four phases

### Phase 0 — truck → starter
Run the trained Explorer (`checkpoints/ppo_emerald_final`) from `initial.state`
until `starter_obtained` is observed. Reuses the proven policy (9-10/10 reliable);
scripting the intro by hand would be absurd.

Plumbing detail (resolved at implementation): the milestone signal comes from the
same `MilestoneTracker` the env uses, driven directly on the emulator, OR Phase 0
runs through the milestone env just until `starter_obtained` fires and then the
raw emulator is reused for phases 1-3. Either is acceptable; the constraint is that
phases 1-3 must NOT go through the terminating env.

### Phase 1 — forced Poochyena battle
`play_battle(emulator, move_type_fn, predict)` with the trained Fighter checkpoint
until the battle resolves (`won`). (Decided: A2 — delegate to the Fighter rather
than hardcoding battle inputs, since we already have a 10/10 Fighter.)

### Phase 2 — lab `(1, 4)` (the brittle part)
After the battle the player is warped to the lab. Drive with `navigate_to`
(NOT via the env), targeting hardcoded cells:
1. `navigate_to` toward Birch's cell.
2. Press A to advance the Pokédex dialogue until the script gate clears.
3. `navigate_to` toward the exit door cell.

Cells are hardcoded from a **manual instrumented probe** (decided: A over reading
the pokeemerald decomp or auto-`map_map`), each with a `# NOTE:` recording its
origin. `navigate_to` is called **without `memory`** so the only branch that reads
`in_battle()` (the `has_grass` learning) is short-circuited — the `(1, 4)`
false-positive does not affect pathfinding.

### Phase 3 — lab → route_101
`navigate_to` toward route_101 `(0, 16)` free-roam, confirm a real position change
there, then `emulator.save_state()` → `states/post_starter.state`.

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

A one-shot manual instrumented probe to read three cells in the lab: Birch's
position, the exit door cell, and the route_101 entry cell. These are hardcoded
into the driver with origin notes.

## Non-goals

- No multi-ROM generalisation — coordinates are BPEF (French Emerald) hardcoded.
- No retraining, no new milestones.
- `tests/test_campaign_rom.py` is not modified; it turns green with the artifact.
- `states/post_starter.state` is gitignored (a local artifact); the smoke stays
  gated (skips without it).

## Deliverables checklist

- `tools/capture_route101_freeroam.py` — the four-phase capture driver.
- Hardcoded lab coordinates with `# NOTE:` origin, obtained via manual probe.
- `states/post_starter.state` generated locally (gitignored).
- `tests/test_campaign_rom.py` passing with the artifact present.
