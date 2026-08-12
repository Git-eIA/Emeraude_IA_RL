# Phase 2 return descent — B2 design (start from route_101)

> Revision of `2026-08-10-phase2-return-descent-design.md`. Same durable
> `reach_map` primitive; the **anchor state and the return chain shrink** because
> the Oldale hop is proven non-viable in continuous descent.

## Why this revision

The merged Approach 1 spec assumed the southbound return is a continuous
greedy-descent in reverse: `route_103 -> Oldale -> route_101 -> Littleroot -> lab`.
Two throwaway probes disproved the **Oldale hop** (see the crux memo):

- **South-edge DOWN sweep (`probe_oldale_south_edge.py`).** Entering Oldale from
  the NORTH (route_103), the south border splits into a **west pocket (x<=8)** and
  an **east region (x>=9)** that are NOT connected on the south rows in continuous
  traversal. The route_101 connection lives at x=11 (east region), walled off from
  the north pocket. Columns x=4..8 are reachable but DOWN crosses nowhere; x=9..15
  are unreachable (walk stalls at (8,19)).
- **Battle-proof instrumented walk (`probe_oldale_desync.py`).** The (10,19) stall
  persists with `in_battle=False battle_starting=False` — not a battle freeze, a
  real grid-decode / connection-border desync. `plan_path_grid` classifies (11,19)
  FREE and reachable; the game refuses the (10,19)->(11,19) step (the cell above is
  `LEDGE_RIGHT`, 1-way).

The prior "REACHED lab=True" was obtained only because the discovery probe
**reloaded `post_starter.state`** to jump past Oldale — a continuous `reach_map`
cannot reload mid-descent, so that result does not carry.

## The B2 decision

**Start the Phase 2 return from `states/post_starter.state`, not `post_rival.state`.**
`post_starter` already has the player on **route_101** (map (0,16), cell (10,17)),
so the return is the continuous, already-proven chain:

```
route_101 (south border) down -> Littleroot (11,1)   [reversible]
Littleroot (8,17)         up   -> lab (6,12)          [door warp, MB_WARP 0x60]
```

The Oldale legs are **dropped entirely**. This matches the real game chronology:
the lab visit (Pokedex + Balls) happens BEFORE the route_103 rival, so descending
from route_103/Oldale was always the wrong direction to model.

### Ground truth (probed on both savestates)

| state        | map                       | party | has_pokedex | shoes | 5 balls |
|--------------|---------------------------|-------|-------------|-------|---------|
| post_rival   | (0,18) route_103 (10,7)   | L8    | **False**   | True  | False   |
| post_starter | (0,16) route_101 (10,17)  | L5    | **False**   | True  | False   |

- `has_pokedex` False on both -> the Pokedex objective is real from post_starter.
- No Phase 2 event gates on rival-beaten -> starting pre-rival is coherent.
- `has_running_shoes` already True on both -> the shoes milestone is an idempotent
  no-op (predicate checked first), exactly as in the merged story campaign.

## Architecture (4 units)

### Unit 1 — De-risk probe (throwaway)

A lean throwaway that, from `post_starter.state` with the Fighter and a healed
party, proves TWO facts before any durable code:

1. **Continuous descent route_101 -> Littleroot -> lab reaches the lab** using the
   two crossing primitives below (explore_grid sweep DOWN, then door-warp UP at
   (8,17)) — with NO reload. Prints `REACHED lab = True/False` and the real
   `Portal` records discovered per hop.
2. **Lab A-spam raises `has_pokedex()`** — from the lab landing cell, bounded
   A-spam flips the flag False->True (and post-asserts `has_item(POKE_BALL_ITEM_ID,
   5)`), confirming the story leg still fires from this anchor.

Output: stdout only, not committed. Purpose: if either fact fails, durable code
pauses on that fact. This may reuse `probe_return_portals.py`'s proven helpers
(`_hop_via_explore_then_scan`, `_precision_walk_to`, `_find_warp_cells`).

### Unit 2 — `reach_map` (durable)

A reusable greedy-descent function beside `travel_to` in `env/map_traveler.py`:

```
reach_map(emulator, reader, memory, goal_map, direction_by_map, *,
          move_type_fn=None, predict=None, max_hops=...) -> str
```

`direction_by_map: dict[tuple[int, int], str]` maps each map the caller expects to
traverse to the crossing direction. `goal_map` is the **stop condition only** —
`reach_map` never computes "toward the goal" geometrically.

Per hop: snapshot the current map; if it is `goal_map`, return `arrived`. Else look
up `direction_by_map[current_map]`; if absent, return `stall`. Cross the border in
that direction with the crossing kind selected by direction, record the real
`Portal` in `MapMemory`, re-evaluate on the new map.

**Two crossing kinds, selected by direction:**

- **Border-connection legs (DOWN)** use the proven `explore_grid` greedy border
  sweep (reaches border cells behind ledge barriers that plain A* cannot). This is
  the route_101 -> Littleroot leg.
- **Door-warp leg (UP)** navigates to the warp cell and steps across: precision-walk
  to the `MB_WARP` (0x60) door cell (Littleroot (8,17)), press the direction, then a
  settle step to complete the warp transition into the lab. This is the
  Littleroot -> lab leg.

Battle-proof: each crossing calls `handle_battle_interruption` itself (it does NOT
get this for free unless routed through `navigate_grid`), so a wild encounter during
a DOWN sweep is cleared by the Fighter; a loss short-circuits.

Returns `arrived | battle_lost | battle_timeout | battle_interrupted | stall |
timeout`. `stall` = current map not in `direction_by_map`, or no crossing fired.
`timeout` = hop budget exhausted.

`reach_map` records portals as it discovers them, so a later `travel_to`/`plan_route`
over the same `MapMemory` sees a real graph. It depends on no pre-seeded portal.
`reach_map` stays generic (direction-agnostic): the return chain constant lives in
`campaign.py` and is passed in as `direction_by_map`.

### Unit 3 — Campaign wiring

`run_campaign` dispatches the descent as a single **reach-home milestone** handled
directly by `reach_map(goal=LAB)` — NOT through `execute_order`'s advance/story
modes. `env/orders.py` is untouched: `run_campaign` already dispatches `story` at
the head of its loop, so a `reach` dispatch is a sibling branch. The milestone
carries a `reach: tuple[int, int] | None` goal-map field; when set, `run_campaign`
calls `reach_map(goal=milestone.reach, direction_by_map=_RETURN_DIRECTIONS)` and
aborts on a non-`arrived` outcome.

**`PHASE2_CAMPAIGN` edit.** The current curriculum opens with advance milestones
(`littleroot`, `lab`); these are **replaced by a single reach-home milestone**
carrying `reach=LAB`. The story milestones (Pokedex / Balls / shoes) are untouched.

```
_RETURN_DIRECTIONS = {ROUTE_101: "down", LITTLEROOT: "up"}
```

(Two hops only — Oldale/route_103 are gone.)

The placeholder `_RETURN_PORTALS` and `seed_return_portals(memory)` are **removed**
(portals are discovered live by `reach_map`); their MapMemory regression test is
removed with them. `run_campaign` stays a pure sequencer.

### Unit 4 — ROM smoke (load-bearing)

`tests/test_phase2_rom.py` runs the **full** campaign from **`post_starter.state`**
(not `post_rival.state`): `reach_map`-driven descent + story A-spam, with the real
Fighter. Asserts `campaign_complete`, `has_pokedex()`, and
`has_item(POKE_BALL_ITEM_ID, 5)` (shoes already True -> idempotent no-op). Dumps
`states/post_phase2.state`. Triple-skips without ROM / Fighter checkpoint /
`post_starter.state`.

## Data flow

```
post_starter.state
  -> run_campaign(PHASE2_CAMPAIGN)
       -> reach milestone: reach_map(goal=LAB)   # greedy descent, records portals
            route_101 --down--> Littleroot --up(warp)--> lab
       -> story milestone: execute_order(mode=story, target=has_pokedex)  # A-spam
       -> story milestone: has_item(Ball, 5)   # idempotent post-assert
       -> story milestone: has_running_shoes   # idempotent no-op
  -> assert + dump post_phase2.state
```

## Error handling

- `reach_map` returns a battle outcome -> `run_campaign` aborts that milestone with
  the battle code (same contract as advance/story dispatch).
- `reach_map` returns `stall` -> surfaced as a milestone failure so the smoke fails
  loudly (no silent papering).
- The Unit 1 probe must reach the lab AND raise `has_pokedex` before Unit 2 is
  coded; a probe failure means a crossing primitive is still wrong and the design
  pauses on that fact.

## Testing

- Unit 2: pure unit tests with a fake reader/emulator that transitions maps on a
  scripted crossing, covering `arrived`, `stall`, `timeout`, a battle outcome, AND
  the door-warp branch (UP). Follows the fake-reader pattern of
  `test_map_traveler`/`test_grid_navigator`.
- Unit 3: `test_campaign` gains coverage that a reach milestone dispatches
  `reach_map` and that a `stall`/battle outcome aborts; removes the
  `seed_return_portals` MapMemory regression test.
- Unit 4: the gated ROM smoke is the single load-bearing end-to-end assertion.

## Gap-check

- **G1 — anchor state exists?** `post_starter.state` is already produced and used by
  `test_campaign_rom.py` (etape 8) -> Unit 4 skip-guard just checks its presence.
- **G2 — reach_map two kinds must not collapse.** The DOWN sweep and the UP warp are
  distinct code paths; Unit 2's fake MUST exercise both (a fake that only flips map
  on DOWN would leave the warp branch untested). Explicit test required.
- **G3 — door-warp settle.** The Littleroot->lab warp needs an extra settle step
  after the UP press (proven in probe_return_portals: UP lands on an intermediate
  wall row, a 64-frame settle completes the transition to (6,12)). reach_map's
  warp crossing MUST include this settle, else it reports `stall` on a mid-warp
  frame.
- **G4 — Fighter required for the DOWN sweep.** route_101 has tall grass; without a
  wired Fighter the sweep returns `battle_interrupted`. Unit 4 MUST wire the real
  Fighter (Unit 1 probe already does).
- **G5 — post_starter starts mid-route_101, not on the border.** reach_map's first
  hop must sweep from (10,17) DOWN to the south border; the explore_grid greedy
  sweep already handles arbitrary start cells (proven northbound). No extra nav.
- **G6 — MapMemory empty at load.** A fresh `post_starter.state` carries an empty
  MapMemory; reach_map discovers portals live, so no seed is needed (the whole
  point of deleting `seed_return_portals`).

## Scope / non-goals

- No change to the RL `env/milestones.py` path.
- No change to the story Order mode or the Phase 2 detectors (already merged).
- No `reach_map`-with-reload (dead with B1); no Oldale/route_103 handling.
- The Unit 1 probe is throwaway; only Units 2-4 are durable.
