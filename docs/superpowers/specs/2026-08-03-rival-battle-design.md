# Rival battle at a destination (`battle_trainer` mode) — design

**Brique 3 part 2** of the "beat the rival on route_103 autonomously from route_101"
goal. Part 1 built the multi-mon combat primitive `play_trainer_battle`; this part
wires it: a new Order mode that travels to a trainer, engages, and plays the battle,
plus the campaign milestone that emits it.

Scope chosen (user): **structure-first**. Wire the mode + campaign milestone + a
disposable capture tool + a gated ROM smoke, structured correctly. Defer the *live*
discovery of the route_103 rival trigger (exact map/cell/approach heading) to when
the capture tool is run — mirroring how Brique 3 part 1 deferred `trainer_battle.state`
and how étape 8 discovered the lab layout via a probe.

## Goal

`execute_order(Order("route_103", "battle_trainer", "win"), ...)` navigates to the
rival, engages, and — with a Fighter wired — plays the trainer battle to `"won"`.
`run_campaign` reaches this through a milestone marked `trainer=True`.

## Architecture

A new Order **mode**, chosen over a campaign-layer flag or an `advance` directive
because:
- Modes are the established vocabulary of Explorer/Fighter skills
  (advance/heal/grind/level_up); each is self-contained and fake-testable.
- It keeps `campaign.py` pure sequencing (its docstring forbids navigation/combat/RAM
  logic there). Campaign only decides *which* milestones carry a trainer.
- `advance` stays a clean navigation-only primitive.

## Components

### 1. `env/orders.py` — the `battle_trainer` mode

`execute_order` dispatches `mode == "battle_trainer"` (alongside heal/grind/level_up,
resolved before the DESTINATIONS lookup path is reached — but this mode *does* use the
destination, so it resolves it like advance).

New private `_execute_battle_trainer(emulator, reader, memory, wallmap, destination,
max_hops, move_type_fn, predict) -> str`:
1. Resolve `destination` via `DESTINATIONS`; unknown name → `"unknown_destination"`.
2. `travel_to(...)` to the trainer's approach cell. Non-`"arrived"` → pass through
   verbatim (`unknown_route`/`unreachable`/`lost`/`timeout`/battle outcomes).
3. **Engage**: a bounded walk toward the rival (twin of grind's
   `_walk_until_encounter`, but a fixed approach heading rather than cycling the four
   directions) until `EncounterWatcher` sees the in_battle front. No battle after the
   budget → `"no_trainer"`.
4. If `move_type_fn`/`predict` are both provided →
   `play_trainer_battle(emulator, move_type_fn, predict)` and return its outcome.
   Otherwise → `"encounter_started"` (no Fighter wired, same convention as grind).

New engage helper `_walk_until_trainer(emulator, reader, heading) -> str`: bounded
`range(ENGAGE_MAX_STEPS)`, press `heading` (ENGAGE_STEP_FRAMES) + release
(ENGAGE_RELEASE_FRAMES, GBA debounce), return `"engaged"` on the in_battle front, else
re-check `in_battle()` after the loop → `"engaged"` | `"no_trainer"`. Constants mirror
grind's (`GRIND_STEP_FRAMES=24`, `GRIND_RELEASE_FRAMES=8`, bound 60); reuse the grind
constants rather than duplicate.

**Approach heading** is a per-destination constant. For v1 the seed uses a single
heading (up, walking north into route_103 toward the rival); it lives next to the
destination cell and is corrected during live discovery. To avoid overloading
`DESTINATIONS` (which is `name -> (map, cell)` shared with advance), the engage heading
is a small separate `TRAINER_APPROACH: dict[str, int]` keyed by destination name.

Outcomes: `unknown_destination` | travel pass-through | `no_trainer` |
`encounter_started` (no Fighter) | `won` | `lost` | `battle_timeout`.

### 2. `env/campaign.py` — the trainer milestone

`Milestone` gains `trainer: bool = False`. `run_campaign`, for a milestone with
`trainer=True`, after the `advance` Order returns `"arrived"`, emits
`Order(destination, "battle_trainer", "win")` via `order_fn` with `move_type_fn`/
`predict`/`max_hops`. Non-`"won"` → returned verbatim (abort on first failure, as
elsewhere).

Per-milestone sequence: level_up (if under level) → advance → *if trainer*
battle_trainer. `campaign.py` stays pure sequencing.

`CAMPAIGN` seed gains `Milestone("route_103", 5, trainer=True)` after the existing
`route_101` milestone.

### 3. `DESTINATIONS` + `TRAINER_APPROACH`

`DESTINATIONS["route_103"] = ((MAP), (CELL))` — coordinates **unverified**, flagged
`# unverified` until the capture tool resolves them.
`TRAINER_APPROACH["route_103"] = KEY_UP` — the engage heading, likewise unverified.

### 4. `tools/capture_rival_battle.py` (disposable)

Mirror of `tools/capture_trainer_battle.py`: load `states/post_starter.state`, walk
north toward route_103 (cycle directions or a `--heading`) until `reader.in_battle()`,
dump `states/trainer_battle.state`, `SystemExit(1)` if none. Docstring flags TWO things
the operator must verify on the saved state: the `in_battle()` false-positive, and that
the captured battle is the **rival on route_103** (check `map_id`), not a wild encounter
in route_101 grass. This run also **discovers** the real route_103 map/cell/approach
heading to backfill into `DESTINATIONS` + `TRAINER_APPROACH`. Output gitignored.

The captured `states/trainer_battle.state` is exactly the artifact Brique 3 part 1's
already-written gated smoke (`test_fighter_wins_a_real_trainer_battle`) needs — so
capturing the rival battle makes **that** smoke load-bearing. One artifact, one job.

### 5. ROM smoke for the full mode — deferred (documented)

There is **no** new ROM smoke for the full `battle_trainer` mode in this brique, and
the reason is documented in the mode's docstring: a mode-level smoke would need a
*pre-trigger overworld* savestate near the rival **plus** route_103 already surveyed
into map memory so `travel_to` can path there. A "walk until in_battle" capture tool
produces an *already-in-battle* state instead — on which `travel_to` runs first, cannot
"arrive" while the character is frozen in battle, and spins to `timeout` without ever
reaching `play_trainer_battle`. So the full-mode ROM smoke is deferred (same posture as
grind's deferred ROM smoke).

Coverage instead:
- The mode's travel→engage→battle sequencing is fully covered by **pure tests** (fake).
- The `play_trainer_battle` leg on a real rival battle is covered by part 1's gated
  smoke, which the capture tool above makes load-bearing.

## Data flow

campaign milestone (trainer=True) → advance arrives → `Order(dest, battle_trainer)` →
`_execute_battle_trainer`: travel_to → `_walk_until_trainer` (in_battle front) →
`play_trainer_battle` → outcome bubbles up verbatim.

## Key assumption (documented)

The destination cell is chosen **out of the rival's line of sight**, so `travel_to`
arrives cleanly without the interruptible-nav layer (Brique 1) catching the battle and
routing it to `play_battle` (the wild, single-mon primitive — wrong for a trainer).
The engage step then triggers the battle deliberately and hands it to
`play_trainer_battle`. The exact safe cell + approach heading are the live-discovery
outputs. If live discovery finds the rival unavoidably triggers mid-travel, that is a
follow-up (teach interruptible nav to distinguish trainer battles) — out of scope here.

## Error handling

Every leg passes its failure outcome up verbatim so a future Strategist can react.
Loops are bounded (code-safety #2): travel_to (max_hops), `_walk_until_trainer`
(ENGAGE_MAX_STEPS), `play_trainer_battle` (max_turns). No new RAM reads beyond the
existing `in_battle()`.

## Testing

Pure (no ROM):
- `test_orders.py`: `battle_trainer` via a fake world — `unknown_destination`;
  travel pass-through (unknown_route); engage→won (fake fires in_battle after N
  approach steps, scripted trainer battle → won); no battle → `no_trainer`; no
  Fighter → `encounter_started`.
- `test_campaign.py`: `Milestone.trainer` defaults False; a trainer milestone emits
  advance then battle_trainer in order; battle_trainer non-`won` → abort without
  advancing further; seed contains `route_103` with `trainer=True`.

ROM: no new mode-level smoke (deferred, §5). The capture tool makes part 1's existing
`play_trainer_battle` smoke load-bearing.

## Scope / non-goals

- Engage = a bounded approach walk; no fine line-of-sight geometry.
- route_103 map/cell/heading unverified until the capture tool runs.
- No capture/min_loss directive (Fighter only wins).
- No Fighter re-training (the wild-trained-vs-trainer in-distribution risk was flagged
  in part 1; part 1's smoke probes it once load-bearing).
- Interruptible nav is not taught to distinguish trainer battles (assumption above).
- No new mode-level ROM smoke (deferred; §5).

## Files

- Modify: `env/orders.py` (mode dispatch + `_execute_battle_trainer` +
  `_walk_until_trainer` + `TRAINER_APPROACH` + `DESTINATIONS["route_103"]`)
- Modify: `env/campaign.py` (`Milestone.trainer` + run_campaign wiring + seed)
- Create: `tools/capture_rival_battle.py`
- Modify: `tests/test_orders.py`, `tests/test_campaign.py`

## Finding (implementation)

`tools/capture_trainer_battle.py` already existed and saves `states/trainer_battle.state`
on ANY in_battle. `tools/capture_rival_battle.py` is the route_103-specific variant: it
verifies the battle is on the target map before saving (a wild battle on the way aborts
with guidance), and defaults `--heading up`. Running it locally is what makes Brique 3
part 1's gated `test_fighter_wins_a_real_trainer_battle` load-bearing and reveals the
real route_103 map/cell/heading to backfill into `DESTINATIONS` + `TRAINER_APPROACH`
(currently the unverified placeholders `((0, 18), (9, 5))` and `KEY_UP`). Not run in CI.
