# Battle-proof survey (Brique 2) — design spec

**Date:** 2026-08-03
**Status:** approved (pending spec review)
**Worktree/branch:** `feat/p4-battle-proof-survey`

## Context

Overarching goal (locked earlier): beat the rival on route_103 with full
autonomy from route_101, split into three independent sub-projects (briques),
each its own spec → plan → impl cycle.

- **Brique 1** (DONE, merged `b365f8c`): interruptible navigation. `navigate_to`
  detects a mid-move wild battle and hands it to the Fighter via
  `_handle_battle_interruption`, then resumes. `travel_to` and `execute_order`
  advance inherit it. `map_map` (survey) was explicitly left OUT of scope.
- **Brique 2** (THIS spec): make the *survey* battle-proof so it can cross
  grassy routes for real, and prove it against the ROM.
- **Brique 3** (later): rival trainer battle + campaign milestone.

## Problem

`env/map_explorer.py::map_map` still has the pre-Brique-1 bug. When a wild
battle fires mid-survey, the player freezes but the loop keeps pressing the
d-pad (`_follow_route` / `probe_step`). `resolve_move` reads that frozen frame
as `"blocked"`, `wallmap.block` records a **false wall**, and the survey spins
to `budget_exhausted`. Crossing grass while surveying is impossible today.

`env/world_surveyor.py::survey_world` calls both `travel_to` (now battle-proof
after Brique 1) and `map_map` (not). Neither threads Fighter deps, so even the
travel leg runs Fighter-less inside a survey sweep.

## Goal

Wire the existing `_handle_battle_interruption` helper into `map_map`, thread
the Fighter deps (`move_type_fn`, `predict`) through `map_map` and
`survey_world`, propagate battle outcomes, and add a **deterministic** live ROM
smoke that proves the survey survives a real route_101 wild battle.

## Non-goals

- No new Fighter capability (wild-win only, exactly like Brique 1).
- No capture / min_loss directive.
- No changes to `travel_to` / `navigate_to` (already battle-proof).
- No re-entry of building warps, no Strategist, no reward.
- No nearest-spot; no `tile_behavior` probe.

## Design

### 1. `map_map` adopts `_handle_battle_interruption`

`env/map_explorer.py::map_map` gains two optional deps:

```python
def map_map(
    emulator, reader, memory, wallmap, target_map,
    max_steps=2000, move_type_fn=None, predict=None,
) -> str:
```

**Promote the helper to public first.** `_handle_battle_interruption` is
currently underscore-private inside `live_navigator`; Brique 2 is its first
cross-module use. Rename it to `handle_battle_interruption` (public) and update
its internal call site in `navigate_to` — same "shared by 2+ modules → make it
public" fix that promoted `_reached` → `reached` in étape 6. Then import it from
`env.live_navigator`. Call it at the **top of each loop
iteration, right after the `enc_watcher.observe` recording block and before
`_nearest_frontier`** — the same ordering Brique 1 established in `navigate_to`
(learn `has_grass` first, then fight):

```python
reached.add(here.pos)
if enc_watcher.observe(reader.in_battle()):
    memory.observe(here, WorldEvent(encounter_started=True))

battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
if battle is not None:
    return battle  # "battle_lost" | "battle_timeout" | "battle_interrupted"

plan = _nearest_frontier(...)
```

Helper contract (unchanged from Brique 1): not in battle → `None`; in battle,
no Fighter → `"battle_interrupted"`; Fighter present → `play_battle` →
`"won"` → `None` (survey resumes, no wall recorded), else `"battle_lost"` /
`"battle_timeout"`.

Because position is frozen during combat and the helper runs before any
d-pad press, a won battle returns `None` and the loop re-snapshots cleanly next
iteration — no false wall. New `map_map` returns: existing
(`complete` / `budget_exhausted` / `left_map`) **plus** `battle_lost` /
`battle_timeout` / `battle_interrupted`.

### 2. `survey_world` threads deps + propagates battle outcomes

`env/world_surveyor.py::survey_world` gains `move_type_fn=None, predict=None`,
forwards them to **both** `travel_to` and `map_map`.

**Abort-at-first-battle-outcome** (chosen over log-and-continue): a wild battle
that the Fighter loses/times out, or that has no Fighter, leaves the game in a
state the sweep cannot reason about (a white-out warps the player away; a
Fighter-less interruption means the survey can't proceed). This mirrors
`travel_to`'s abort-at-first-battle-failure philosophy. Reuse the existing
`BATTLE_OUTCOMES = ("battle_lost", "battle_timeout", "battle_interrupted")`
constant from `env/map_traveler.py`.

**Preserve the existing travel guard.** `travel_to` today runs only inside
`if here != target and target != start:` (skip travel when already standing on
the target/start map — this also guards `_entry_cell` against an empty index).
Keep that guard; only add the battle-outcome check inside it:

```python
if here != target and target != start:
    outcome = travel_to(
        emulator, reader, memory, wallmap, target, _entry_cell(memory, target),
        move_type_fn=move_type_fn, predict=predict,
    )
    if outcome in BATTLE_OUTCOMES:
        failed.append((target, f"travel:{outcome}"))
        return SurveyReport(tuple(surveyed), tuple(failed))
    if outcome != "arrived":
        failed.append((target, f"travel:{outcome}"))
        continue

result = map_map(
    emulator, reader, memory, wallmap, target,
    move_type_fn=move_type_fn, predict=predict,
)
if result in BATTLE_OUTCOMES:
    failed.append((target, f"map:{result}"))
    return SurveyReport(tuple(surveyed), tuple(failed))
if result in ("left_map", "budget_exhausted"):
    failed.append((target, f"map:{result}"))
surveyed.append(target)
```

The non-battle failure modes keep their existing log-and-continue behaviour.
`SurveyReport` shape is unchanged.

### 3. Deterministic live grass smoke (option B)

Emerald wild encounters are stochastic (~10-20%/grass-step), so a "survey and
hope a battle fires" smoke proves nothing on a no-encounter run. Instead:

**Disposable capture tool** `tools/capture_route101_in_battle.py` (à la
`capture_lab_entry.py`): loads `states/post_starter.state`, walks the party into
route_101 grass pressing the d-pad until `reader.in_battle()` flips true, dumps
`states/route101_in_battle.state`. Bounded step budget; documents that this is a
one-shot artifact producer, gitignored like the other states.

**Gated ROM smoke** `tests/test_battle_proof_survey_rom.py`: double-skip on
`POKEMON_EMERALD_ROM` **or** missing `states/route101_in_battle.state`. Loads
that state (first `map_map` iteration always sees an in-progress battle), wraps
the real Fighter checkpoint `ppo_fighter_final.zip` into `predict`, runs
`map_map` on route_101, and asserts:
- the outcome is **not** a battle outcome (`result not in BATTLE_OUTCOMES`) — the
  Fighter won and the survey resumed rather than aborting;
- the battle was actually resolved (`not reader.in_battle()` after the call), so
  the assertion above isn't vacuous on a state that never re-entered combat;
- grass was learned at the starting cell
  (`memory.cells_labeled("has_grass")` is non-empty), proving the loop reached
  the recording block on the in-battle frame.

These are stable observables; asserting "no wall at the pre-battle cell" is
avoided because the probed frontier cell is emulator-dependent.

SB3/torch imported inside the test body (not at collect), same pattern as
`test_battle_player_rom.py`.

## Unit tests (no ROM)

- `test_map_explorer.py`: extend the fake world to fire an in-battle frame; a
  Fighter-less `map_map` returns `"battle_interrupted"` (no false wall
  recorded); a `map_map` with a winning fake Fighter resumes and reaches
  `"complete"`; a losing fake Fighter returns `"battle_lost"`.
- `test_world_surveyor.py`: a `map_map`/`travel_to` battle outcome aborts the
  sweep immediately with the leg recorded in `failed`; the Fighter deps reach
  both calls; the no-battle path is unchanged.

## Files

- Modify: `env/live_navigator.py` (promote helper to public `handle_battle_interruption`)
- Modify: `env/map_explorer.py` (deps + helper wiring + returns)
- Modify: `env/world_surveyor.py` (deps + battle-outcome propagation)
- Create: `tools/capture_route101_in_battle.py` (disposable artifact producer)
- Create: `tests/test_battle_proof_survey_rom.py` (gated deterministic smoke)
- Modify: `tests/test_map_explorer.py`, `tests/test_world_surveyor.py`

## Outcome contract

`map_map` → `complete | budget_exhausted | left_map | battle_lost |
battle_timeout | battle_interrupted`.

`survey_world` → `SurveyReport` where a battle outcome appears in `failed` as
`travel:<outcome>` / `map:<outcome>` and terminates the sweep early.
