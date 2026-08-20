# Running Shoes Driver — Design

**Date:** 2026-08-21
**Status:** validated (brainstorming, user-approved section by section)
**Depends on:** A2 pokedex-return driver (merged `3bc717d`), control-return crux resolution
(2026-08-17 investigation, MEMORY.md)

## Context

`run_pokedex_return` (env/campaign.py) currently ends its last leg at the FIRST frame
`has_pokedex()` is True — mid-cutscene, ~55 A-presses before the 5 Poke Balls land and
before the script's `releaseall`. The dumped `states/post_pokedex.state` is therefore
mid-cutscene: resuming from it looks like a control lock (the "false lock" crux). The
investigation proved control IS returned; the recipe out is: finish the cutscene
(dex + 5 balls + `VAR_BIRCH_LAB_STATE`(0x4084)==5), then B x10 to drain the Birch
dialogue boxes that continued A-spam re-opened (player ends at (6,5) facing Birch).

Downstream, the mom/running-shoes event (`FLAG_RECEIVED_RUNNING_SHOES` 0x112) was
located by a throwaway probe (2026-08-21, 3 runs): it fires while walking NORTH through
Littleroot toward the route_101 connection — no house visit needed. Ground truth:

- Any northbound nav is intercepted: a `hop_via_explore(LITTLEROOT, ROUTE_101, "up")`
  sweep got frozen by the scripted event (player left at (10,9), hop returned
  `no_portal`). The interception IS the expected success path.
- ~40 bounded A-presses through mom's dialogue flip FLAG 0x112.
- `VAR_LITTLEROOT_TOWN_STATE`(0x4050, `town_state`) transitions 3 -> 4 exactly when
  the event completes; control returns immediately after (DOWN moved the player).
- Exit predicate: `has_running_shoes() AND town_state == 4`. Location-agnostic —
  no trigger coordinates needed.

## Decisions

1. **Frontier** (user-decided): extend `run_pokedex_return` to finish the cutscene
   properly and return WITH control; add a new short driver `run_shoes_leg` starting
   from that point. No monolithic driver.
2. **Leg internals** (user-decided): predicate-driven drain built 100% from existing
   primitives. No new generic `walk_into_scripted_event` primitive (YAGNI until a
   second interception event exists), no hardcoded trigger coordinates.
3. No new emulator capability, no RAM poke, no splice — pure orchestration.

## Architecture

```
post_rival.state
   |
   v
run_pokedex_return (EXTENDED)                          env/campaign.py
   hop_via_explore route_103 -> Oldale "down"          (unchanged)
   cross_scripted_npc Flora gate                       (unchanged)
   reach_map -> LAB                                    (unchanged)
   _finish_lab_cutscene:                               (wraps _advance_story_dialogue)
       A-spam (bounded) until has_pokedex()
                          AND has_poke_balls(5)
                          AND birch_lab_state()==5
       B x10 release
   -> "pokedex_delivered", player in LAB (6,5), CONTROL RETURNED
   |
   |  (smoke A2 re-dumps a HEALTHY post_pokedex.state here)
   v
run_shoes_leg (NEW)                                    env/campaign.py
   1. exit lab: DOWN <=60 until map==LITTLEROOT, settle 60 frames
   2. walk north: hop_via_explore(LITTLEROOT, ROUTE_101, "up")
      -> result IGNORED on purpose (mom event freezes the sweep; 'no_portal'
         is the expected outcome — explicit comment on this contract)
   3. drain event: bounded A/B cycles until
         has_running_shoes() AND player_state().town_state == 4
   4. control check: bounded direction presses until position moves
   -> "shoes_delivered", player in LITTLEROOT, CONTROL RETURNED
   |
   |  (smoke dumps states/post_shoes.state here)
   v
downstream Phase 2 (route_101 northbound with running shoes)
```

## Components

### 1. `EmeraldReader.birch_lab_state()` — env/game_state.py

Public read of `VAR_BIRCH_LAB_STATE` (0x4084), today only reachable through the
private `_var`. Returns `int | None` (None while save blocks relocate), mirroring
the existing var-read style used for `town_state`.

### 2. `run_pokedex_return` extension — env/campaign.py

The final leg `_advance_story_dialogue(emulator, reader, lambda r: r.has_pokedex())`
is replaced by a private module helper:

```
_finish_lab_cutscene(emulator, reader) -> bool
    # DRY: reuses orders._advance_story_dialogue (STORY_MAX_PRESSES=2000 covers the
    # ~150 measured presses) with the COMBINED predicate:
    #   has_pokedex() AND has_poke_balls(5) AND birch_lab_state()==5
    # On "story_done": B x _RELEASE_B_PRESSES=10 (8/8 frames) drains the re-opened
    # Birch boxes, return True. On "story_timeout": return False.
```

On False, `run_pokedex_return` keeps returning `"pokedex_not_delivered"` (status
surface unchanged for callers). On True it returns `"pokedex_delivered"` — now
meaning "delivered AND control returned".

### 3. `run_shoes_leg` — env/campaign.py

```
run_shoes_leg(emulator, reader, memory, *, move_type_fn=None, predict=None) -> str
```

`reader` follows the same composite duck-typed contract as `run_pokedex_return`
(the A2 smoke's wrapper exposing BOTH WorldReader — snapshot/grid_reader — and
EmeraldReader — flags/vars — attributes). Legs:

1. **Lab exit**: press DOWN (12/4 frames) up to `_LAB_EXIT_MAX_PRESSES=60`; success
   when `player_state()` map == LITTLEROOT (0,9). Else `"lab_exit_timeout"`.
   Then `emulator.step(0, _SETTLE_FRAMES)` (reuse the existing 60-frame constant).
2. **North walk**: `hop_via_explore(emulator, reader, memory, LITTLEROOT, ROUTE_101,
   "up", move_type_fn=..., predict=...)`. Return value deliberately not branched on.
3. **Event drain**: up to `_SHOES_MAX_CYCLES=80` cycles of (A x4, B x1); after each
   cycle test `has_running_shoes() AND town_state == 4`. Else `"shoes_timeout"`.
4. **Control check**: up to `_CONTROL_MAX_CYCLES=30` cycles of (DOWN press; if the
   position/map did not change, B x2 to drain any box the drain's last A re-opened
   — the probe's P6 pattern). Success when the position (or map) changes. Else
   `"control_timeout"`.

Returns `"shoes_delivered"` | `"lab_exit_timeout"` | `"shoes_timeout"` |
`"control_timeout"`. New module constant `ROUTE_101` already exists; LITTLEROOT too.

## Bounds (probe-measured, margin >= x2)

| Step | Bound | Measured |
|---|---|---|
| Cutscene A-spam | STORY_MAX_PRESSES=2000 (reused) | ~150 from lab_arrival (~55 from old mid-cutscene dump) |
| Release B | 10 | 5+ works, 3 insufficient |
| Lab exit DOWN | 60 | 11 |
| Mom-event drain cycles | 80 | ~10 to shoes, +4 to town_state==4 |
| Control-check cycles | 30 | 1 |

## Error handling

- Each leg surfaces its first failure verbatim (same philosophy as
  `run_pokedex_return`); no retries, no destructive recovery.
- The ignored `hop_via_explore` result is the ONE deliberate exception, documented
  in-code: the mom interception is the success path; the shoes/town_state predicate
  is the arbiter, not the hop status. If the event never fires (unexpected ROM
  state), the drain expires into an honest `"shoes_timeout"`.
- Starting `run_shoes_leg` from a mid-cutscene state (e.g. today's stale
  `post_pokedex.state`) fails cleanly as `"lab_exit_timeout"` — no false positive.

## State ordering (smoke dependency)

The current `states/post_pokedex.state` is mid-cutscene. The modified A2 smoke
re-dumps it healthy (post-release, control verified). The shoes smoke consumes that
healthy state and double-skips (ROM / state) when missing — no Fighter needed:
Littleroot and the lab have no wild grass, so no battle can interrupt the leg.
Run order: A2 smoke first, then shoes smoke.

## Testing

**Unit (fakes, no ROM):**
- `birch_lab_state()`: plant the var in the fake SaveBlock1 (existing fixture style).
- `_finish_lab_cutscene`: fake emulator/reader scripting the predicate sequence;
  pin that A-spam does NOT stop at `has_pokedex()` alone (anti-false-lock
  regression) and that the B x10 release follows.
- `run_shoes_leg` orchestration: stub each leg; assert the 4 failure statuses
  propagate verbatim and the success path returns `"shoes_delivered"`; assert the
  hop result is ignored (stub returns `"no_portal"`, leg still succeeds); assert
  bounds are respected (counting fakes).

**ROM smokes (gated):**
- `test_pokedex_return_rom.py` (modified): existing assertions kept
  (`pokedex_delivered`, `has_pokedex()`, map LAB); then dump the healthy
  `post_pokedex.state` FIRST, then reload it and verify control (a DOWN press
  moves the player) — dump-then-reload so the verification never mutates the
  dumped state, and the pin proves the DUMP is healthy, not the live session.
- `test_shoes_leg_rom.py` (new): from healthy `post_pokedex.state`,
  `run_shoes_leg` == `"shoes_delivered"`, `has_running_shoes()` True,
  `town_state == 4`, dump `states/post_shoes.state` FIRST, then RELOAD the dump
  and verify control (direction press moves) — same anti-false-lock pin.

## Out of scope

- Any nav beyond the mom event (route_101 northbound with shoes = next phase).
- Generic scripted-event primitive (`walk_into_scripted_event`) — revisit when a
  second interception event is needed.
- Campaign/milestone wiring (`PHASE2_CAMPAIGN`) — the leg is a standalone driver
  like `run_pokedex_return`; sequencing them is a later composition decision.
- Emulator capabilities, RAM pokes, savestate splices.
