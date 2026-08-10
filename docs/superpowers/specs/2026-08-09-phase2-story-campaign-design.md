# Phase 2 — Story Campaign (return to lab, Pokédex, Poké Balls, running shoes)

**Date:** 2026-08-09
**Status:** design, approved for spec
**Branch/worktree:** `feat/phase2-story-campaign` / `../Emu-phase2-story-campaign`

## Goal

Extend the scripted Order language so the campaign can drive the post-rival story
sequence deterministically: from `states/post_rival.state` (route_103, overworld
control regained after beating the rival) walk back to Littleroot, enter Birch's
lab, receive the Pokédex and the 5 Poké Balls, then receive the running shoes on
Route 101. Each event is a detected milestone. This is reusable, tested campaign
machinery — NOT an RL training objective (the lab is script-gated; a wandering
policy cannot complete it, proven at étape 7).

## Canonical story order (user-confirmed)

starter → save Birch → brief lab warp → NORTH (route_101 → Oldale → route_103) →
**beat rival (DONE — `post_rival.state`)** → return to lab → Pokédex + Poké Balls
(Birch/rival lab cutscene) → running shoes (Mom, Route 101). Optional Flora talks
on the route exist only if the live probe confirms them.

## Approach (chosen: generalize the heal pattern)

A new `story` Order mode is the sibling of `heal`: `travel_to(cell)` then A-spam a
bounded dialogue loop until a **target predicate** holds. Phase 2 events are
mostly cutscenes auto-triggered on arriving at a map/cell (Birch in the lab, Mom
on Route 101), so "arrive then A-spam until a flag/item flips" is exactly the heal
shape with the completion condition swapped for an injected predicate.

Rejected alternatives: a throwaway per-event driver tool (not reusable, weak
tests); an NPC-object-scan interaction mode like the rival capture (overkill for
cell-triggered cutscenes; kept as a per-event fallback only if the probe shows an
event needs walking into a mobile NPC).

## Architecture & components

Three touched modules, all siblings of existing patterns; one throwaway probe.

### `env/orders.py` — new `story` mode

- `execute_order` gains a `story_target: Callable[[Any], bool] | None = None`
  kwarg, threaded exactly like `target_level` is for `level_up`. `Order` stays
  pure data (three strings); the predicate never lives on the dataclass.
- `_execute_story(emulator, reader, memory, destination, story_target, max_hops,
  move_type_fn, predict)`:
  1. `dest = DESTINATIONS.get(destination)` → `"unknown_destination"` if missing.
  2. `travel_to(cell, ...)` → pass through any non-`"arrived"` outcome verbatim.
  3. If `story_target(reader)` already True → `"story_done"` (idempotent /
     re-entrant, mirroring heal returning `"healed"` when already full).
  4. `_advance_story_dialogue`: bounded `range(STORY_MAX_PRESSES)` loop pressing
     A (press + release for GBA debounce), returning `"story_done"` as soon as
     `story_target(reader)` holds; else `"story_timeout"`.
- New constants: `STORY_PRESS_A_FRAMES`, `STORY_RELEASE_FRAMES`,
  `STORY_MAX_PRESSES` (bound the interaction — code-safety #2). Sized from the
  probe's measured A-press counts (lab cutscene is long, cf. ~600 A in the
  route_101 free-roam capture).
- New `DESTINATIONS` entries for the lab and the Route 101 shoes-trigger cell.
  Exact cells frozen by the probe. **The lab `story` cell is the warp-landing
  tile, not an interior cell**: the Birch cutscene auto-fires on entry and locks
  overworld control before any intra-map walk, so `travel_to` must arrive with
  ZERO intra-map steps. If it aimed at an interior cell, `navigate_grid` would
  press the d-pad on a frozen game → false walls → `unreachable`/`timeout`.
  Consequently `_execute_story` must tolerate control-lost-before-arrived: on the
  lab hop, `travel_to` crossing the door already lands on the trigger tile, and
  the story loop A-spams from wherever it lands (it does not re-assert an exact
  cell match once the cutscene has control).

`travel_to` still receives optional `move_type_fn`/`predict`: a wild on the
southbound return is handled by `handle_battle_interruption` (Brique 1). Without a
Fighter wired a wild yields `"battle_interrupted"`, so the Phase 2 run wires the
Fighter as the rival run does.

### `env/game_state.py` (EmeraldReader) — detectors

Backed by the existing `read_flag` / RAM reads.

- `has_pokedex()` → `read_flag(FLAG_SYS_POKEDEX_GET)`. Flag ID confirmed live by
  the probe (candidate `0x801`, NOT frozen).
- `has_running_shoes()` → `read_flag(FLAG_RECEIVED_RUNNING_SHOES)`. Candidate
  `0x86F`, NOT frozen.
- `has_item(item_id, min_qty=1)` → the hard part. Emerald encrypts item
  quantities: `real_qty = stored_qty XOR (securityKey & 0xFFFF)`, `securityKey`
  at `gSaveBlock2Ptr + 0xAC`. This requires a new `SAVE_BLOCK2_PTR` constant on
  `game_state.py` (only `SAVE_BLOCK1_PTR` exists today). Bag pockets live in
  SaveBlock1. `has_item` reads the Poké Balls pocket, finds `item_id` (Poké
  Ball = `0x4`), decrypts the quantity, compares to `min_qty`. All offsets
  (pocket base, slot size, securityKey offset) frozen live by the probe before
  wiring.

### `env/campaign.py` — hand-seeded return portals + Phase 2 curriculum

**The memory-seeding gap (was the biggest design hole).** `travel_to` routes via
`plan_route(memory, ...)` and `memory.portal(from_map, to_map)`; if a portal is
not already in `memory` it returns `"unknown_route"` immediately
(`map_traveler.py:47-53`). But `MapMemory` is a live Python object — it is NOT
serialized in the savestate. Loading `post_rival.state` yields an EMPTY memory,
so the very first story milestone's `travel_to` would return `"unknown_route"`
before taking a step. The rival (northbound) leg never relied on this: it was
driven by the RL Explorer / capture tools using single-map A*
(`grid_navigator.plan_path_grid`), NOT inter-map `travel_to`. So the spec's
earlier "same assumption as the rival leg" was wrong.

- Fix: a hand-seeded portal registry, same philosophy as `DESTINATIONS` — a
  name/edge means something to the chef before any exploration. A
  `seed_return_portals(memory)` helper records the known southbound connections
  (route_103 ↔ Oldale ↔ route_101 ↔ Littleroot ↔ lab) into a fresh `MapMemory`
  before the run, using `MapMemory`'s existing portal-recording API. Each portal
  is `(from_map, to_map, from_cell, direction)`; cells frozen by the probe
  (the same map-transition tiles the northbound leg crossed).
- `run_campaign` for Phase 2 seeds the memory once at the start (or the runbook
  passes a pre-seeded memory). Exact portal cells are NOT frozen here — the probe
  dumps them, same as every other Phase 2 constant.

### `env/campaign.py` — Phase 2 curriculum

- `Milestone` gains `story_target: Callable[[Any], bool] | None = None` (last
  field, positional-compat with the existing `trainer` flag).
- New `PHASE2_CAMPAIGN` tuple. **Milestone modes are chosen by event shape**:
  - Return-to-Littleroot and enter-the-lab are pure arrivals (no dialogue to
    advance until you're inside) → plain `advance` milestones (`story_target =
    None`), reusing the existing advance dispatch verbatim. They just walk there.
  - Pokédex, Poké Balls, and running shoes are the cutscene events → `story`
    milestones, each carrying a `story_target` predicate (e.g.
    `lambda r: r.has_pokedex()`).
- **Pokédex and Poké Balls are ONE lab cutscene, not two.** The single Birch/
  rival cutscene grants both. So the Balls milestone is an idempotent post-assert
  of the SAME A-spam: after the Pokédex `story` milestone returns `"story_done"`,
  the Balls milestone's `story_target` (`lambda r: r.has_item(0x4, 5)`) already
  holds → `_execute_story` returns `"story_done"` with no extra A-spam (the
  idempotent branch). It is a verification checkpoint, not a second interaction.
- `run_campaign` emits `Order(dest, "story", "win")` for story milestones and
  threads `milestone.story_target` into `execute_order`, aborting on the first
  non-`"story_done"` outcome (surfaced verbatim), exactly as it aborts on
  non-`"arrived"` / non-`"won"` today. It also seeds the return portals
  (`seed_return_portals`) into a fresh memory before walking the curriculum.

## Data flow

```
run_campaign(PHASE2_CAMPAIGN)
  └─ seed_return_portals(memory)      # once: else first travel_to = unknown_route
  └─ per Milestone:
       story_target is None (advance milestone: return / enter lab)
         └─ Order(dest, "advance", "win") → travel_to → abort if != "arrived"
       story_target set (story milestone: Pokédex / Balls / shoes)
         └─ Order(dest, "story", "win")
              └─ execute_order(..., story_target=milestone.story_target)
                   └─ _execute_story:
                        1. DESTINATIONS[dest]           → else "unknown_destination"
                        2. travel_to(cell)              → non-"arrived": pass-through
                        3. story_target(reader) already → "story_done" (idempotent;
                             this is the Balls-after-Pokédex path)
                        4. _advance_story_dialogue (A-spam, bounded):
                             story_target True  → "story_done"
                             bound hit          → "story_timeout"
              abort on any outcome != "story_done"
  └─ "campaign_complete"
```

`_execute_story` outcomes: `"unknown_destination"` | travel_to pass-through
(`unknown_route`/`unreachable`/`lost`/`timeout`/`battle_lost`/`battle_timeout`/
`battle_interrupted`) | `"story_done"` | `"story_timeout"`.

The southbound return route_103 → Oldale → Route 101 → Littleroot → lab is four
map transitions; `travel_to` routes via `memory`'s portals. Those portals are NOT
in the loaded savestate (memory is a live object, not serialized), so the run
seeds them explicitly via `seed_return_portals(memory)` before the first
milestone — otherwise the first `travel_to` returns `"unknown_route"` on step
zero. This is a hand-seeded registry (like `DESTINATIONS`), NOT a survey.

## Testing strategy

### Throwaway probe first — `tools/probe_phase2_facts.py`

From `post_rival.state`, Fighter wired, navigate route_103 → lab, A-spam the Birch
cutscene, then A-spam the exit / Route 101 (Mom). Dumps the facts that freeze the
design constants — nothing durable is wired before this runs:
- real Pokédex / running-shoes flag IDs (before/after scan of SaveBlock1 flags),
- securityKey offset + XOR value for the Poké Ball slot (real qty vs stored),
- trigger cells (lab warp-landing tile, Route 101 Mom),
- the southbound portal cells + directions for `seed_return_portals`
  (route_103→Oldale→route_101→Littleroot→lab map-transition tiles),
- A-press count per cutscene (sizes `STORY_MAX_PRESSES`),
- whether the Flora route talks exist as separate events.

### Pure tests (fakes, no ROM)

- `_execute_story`: predicate flips after N A-presses → `"story_done"`; predicate
  never true → `"story_timeout"`; predicate already true → `"story_done"` with no
  A-spam (idempotence); `travel_to` non-`"arrived"` → pass-through.
- Detectors: crafted bytes — flag byte set/unset; item slot `(item_id, qty XOR
  key)` with a known key → `has_item` returns the correctly decrypted quantity.
- `run_campaign(PHASE2_CAMPAIGN)`: scripted fake `order_fn` → correct sequence
  (advance milestones dispatch `advance`, story milestones dispatch `story`),
  abort on first non-`"story_done"` / non-`"arrived"`.
- `seed_return_portals(memory)`: a fresh `MapMemory` ends up with each seeded
  edge queryable via `memory.portal(from_map, to_map)`, so `plan_route` finds the
  southbound path (regression guard for the empty-memory bug).

### Anti-circularity guard

Pure tests validate the ARITHMETIC (flag bit set; XOR decrypt against a known key)
against the detector's own constants — true by construction. ONLY the live probe
and the ROM smoke are load-bearing: the probe confirms that after the real lab
cutscene `has_pokedex()` / `has_item(0x4)` read True with the frozen offsets; the
smoke replays the whole sequence.

### Gated ROM smoke — `tests/test_phase2_rom.py`

Triple-skip (ROM | Fighter ckpt | `states/post_rival.state`). Loads
`post_rival.state`, runs the Phase 2 curriculum, asserts `has_pokedex()` AND
`has_running_shoes()` become True, dumps `states/post_phase2.state`. Load-bearing
once the probe freezes the constants.

## Risks

- **Item decryption (securityKey)** = biggest unknown. Fallback: if the probe
  shows the inventory read is unreliable, detect the Poké Balls receipt via a lab
  event flag (if one exists) instead of the bag — decided at plan time on probe
  evidence, not now.
- **Lab cutscene locks control on entry**: the Birch cutscene auto-fires on the
  warp-landing tile, before any intra-map walk. The lab `story` cell is therefore
  the landing tile (0 intra-map steps) and `_execute_story` A-spams from wherever
  it lands rather than re-asserting an exact cell — else `navigate_grid` presses a
  frozen game and poisons the blocked set. Landing tile frozen by the probe.
- **Lab script-gate order**: you must talk to Birch (Pokédex) before you can
  leave; harmless here since the story loop A-spams immediately on arrival.
- **Flora route talks** may not exist as separate events → that milestone is
  dropped if the probe does not find them.
- **Shoes trigger**: Mom auto-runs on a specific southbound Route 101 tile → cell
  frozen by the probe.
- **Multi-map return nav** needs seeded portals: a fresh `MapMemory` is empty
  (not serialized in the savestate), so `seed_return_portals(memory)` must run
  before the first `travel_to` or it returns `"unknown_route"` on step zero. The
  northbound rival leg did NOT use `travel_to` (it used single-map A*), so there
  is no inherited memory to rely on — the seed is mandatory, not optional.

## Non-goals

- No RL / `env/milestones.py` change (scripted path only — user's choice); the
  route103-rival run and `route103_milestones()` are untouched.
- No capture/min_loss directive; `play_battle` intact; no Fighter retraining.
- Existing modes (advance/heal/grind/level_up/battle_trainer) stay byte-identical
  — every new parameter is defaulted.

## Files

- `env/orders.py` — `story` mode, `_execute_story`, `_advance_story_dialogue`,
  STORY_* constants, new DESTINATIONS entries.
- `env/game_state.py` — `has_pokedex`, `has_running_shoes`, `has_item`,
  `SAVE_BLOCK2_PTR` constant (securityKey).
- `env/campaign.py` — `Milestone.story_target`, `PHASE2_CAMPAIGN` (mixed
  advance/story milestones), `seed_return_portals`, `run_campaign` story dispatch.
- `tests/test_orders.py` (story sequencing), `tests/test_game_state.py`
  (detectors), `tests/test_campaign.py` (PHASE2 dispatch) — pure tests.
- `tests/test_phase2_rom.py` — gated ROM smoke.
- `tools/probe_phase2_facts.py` — throwaway live-facts probe (run first).
