# run_campaign ROM smoke (plumbing) — Design

**Date:** 2026-08-02
**Palier:** P4 étape 7 — first load-bearing ROM wiring for `run_campaign`.

## Goal

Prove `run_campaign` (the scripted-Strategist driver from étape 6) actually wires to
the emulator. Until now every campaign test is pure (fake `order_fn`, no ROM). This
palier captures a real post-starter overworld savestate and runs the driver from it:
it must skip `level_up` (the party is already at the required level) and drive a *real*
`advance` against the ROM, ending on a legal outcome after real navigation.

## Non-goals (deferred, same reasons as the grind/level_up smokes)

- The full grind loop against random wild battles (no deterministic encounter path).
- Verifying the `route_101` `(5, 12)` cell exactly (flagged unverified in `DESTINATIONS`).
- Fighting the champion/trainer at the destination (advance is navigation-only in v1).
- Any change to `env/campaign.py` or `env/orders.py` — this palier adds only a capture
  tool and a gated test.

## Architecture

Two new files, **zero production change**.

### 1. `tools/capture_post_starter.py` (one-shot artifact generator)

Modeled on `tools/capture_open_map.py`. Drives the trained Explorer to produce the
savestate the smoke needs.

- Loads `states/initial.state`, builds `PokemonEmeraldEnv(GbaEmulator(rom), [initial],
  max_steps=10_000_000)` and a `WorldReader(env.emulator.read_bytes)`, loads the PPO
  policy from `checkpoints/ppo_emerald_final`.
- Steps the env with `model.predict(obs, deterministic=False)`, watching
  `info["milestones"]` (the env already exposes `sorted(self._milestones.fired)`).
- **Clears the forced starter battle first.** `starter_obtained` fires on
  `party_count >= 1`, which happens the instant you grab the starter from Birch's bag —
  i.e. *during* the forced wild-Poochyena battle. So a fixed post-roll would capture
  mid-battle. Instead, once `"starter_obtained"` has fired, the tool keeps stepping until
  `reader.in_battle()` is `False` for a few consecutive frames (small settle), bounded by
  the remaining `--max-steps`, then writes `states/post_starter.state` via
  `env.emulator.save_state()`. The level-5 starter beats the level-2 Poochyena by
  mashing, so the Explorer's overworld policy clears it.
- Prints the final `map_id`, `pos`, `reader.in_battle()`, and `reader.party_levels()` so
  the operator confirms an *overworld* (not mid-battle), level-5-party capture on the
  route_101 map before trusting the artifact.
- Bounded by `--max-steps` (default 8000); prints a "not reached" message and exits
  without writing if `starter_obtained` never fires (or the battle never clears in
  budget).

Run once locally in the main repo where the checkpoint/ROM live. Output is a gitignored
artifact (like `open_map.state`).

Usage:
```
POKEMON_EMERALD_ROM=... .venv/bin/python tools/capture_post_starter.py --max-steps 8000
```

### 2. `tests/test_campaign_rom.py` (gated ROM smoke)

Double-skip, matching the existing gated-smoke pattern:
- skip if `POKEMON_EMERALD_ROM` is unset;
- skip if `states/post_starter.state` is missing (resolved absolutely via
  `parents[2] / "Emu" / "states" / "post_starter.state"`, since `states/` lives in the
  main repo, not the worktree).

The single test:
1. Loads the savestate into `GbaEmulator`, `emu.step(0, 4)` to settle after load.
2. Builds `reader = WorldReader(emu.read_bytes)` (`MapMemory` from `env.map_memory` and
   `WallMap` from `env.local_navigator` are constructed inline at the `run_campaign` call
   in step 6).
3. Records `start = reader.snapshot()` (assert not `None`), keeps `start.pos`.
4. **Same-map precondition.** Asserts `start.map_id == DESTINATIONS["route_101"][0]`
   (imported from `env.orders`, `== (0, 16)`). This guarantees `advance` takes
   `travel_to`'s same-map branch — which delegates straight to `navigate_to` and needs no
   portals, so an empty `MapMemory()` is correct. A capture on the wrong map fails here
   loudly instead of silently returning `unknown_route`.
5. Asserts `reached(reader.party_levels(), 5)` is `True` — confirms the capture really
   holds a party whose mean level is ≥ 5, so `run_campaign` *will* skip `level_up` by
   construction (target 5 is a floor; a level-5-or-6 starter both pass).
6. Runs `outcome = run_campaign(emu, reader, MapMemory(), WallMap(),
   curriculum=(Milestone("route_101", 5),))` with **no Fighter**
   (`move_type_fn`/`predict` default `None` — fine, since `level_up` is skipped and
   `grind` never runs).
7. Asserts `outcome in {"campaign_complete", "unreachable", "left_map", "timeout"}` —
   the exact same-map advance outcome set (`arrived` → `campaign_complete`, else
   `navigate_to`'s `unreachable`/`left_map`/`timeout` surfaced verbatim). This
   *simultaneously* proves `level_up` was skipped: had it fired without a Fighter, the
   first `_execute_grind` would surface a grind outcome (`"encounter_started"` /
   `"no_grass_spot_known"`), which is not in the set.
8. Asserts `outcome == "campaign_complete" or reader.snapshot().pos != start.pos` — real
   navigation happened on the ROM (or the driver arrived immediately).

## Why this is load-bearing

- The map precondition (step 4) fails loudly if the capture landed on the wrong map,
  turning a would-be confusing `unknown_route` into a clear diagnostic.
- The party-level assertion (step 5) fails loudly if the capture produced a state without
  a ≥5 party — the smoke can't silently pass on a bad or mid-battle savestate.
- The outcome-set assertion (step 7) fails if `run_campaign` crashes, if the `level_up`
  branch fires unexpectedly, or if `advance` returns a string outside the same-map set.
- The pos assertion (step 8) fails if the emulator was never actually stepped — proving
  the driver reached and exercised `travel_to` / `navigate_to` against real RAM.

## Files

- Create: `tools/capture_post_starter.py`
- Create: `tests/test_campaign_rom.py`
- Manual: run the tool once in the main repo to produce `states/post_starter.state`.

## Testing

- The smoke is gated; the pure suite is unchanged (still 261 passed + 1 skipped).
- Acceptance: run the capture tool, confirm it prints a level-5 party on the route_101
  map, then run the smoke and confirm it PASSES (not skips) — making it load-bearing.
