# Phase 2 return descent — design

## Problem

The Phase 2 story campaign (return to Birch's lab -> Pokedex + 5 Poke Balls ->
running shoes) is built, tested, and merged. The only gap is **physically getting
the player home**. `run_campaign` relies on `travel_to` + hand-seeded
`_RETURN_PORTALS`, but those portal records are **placeholders**
(`from_cell == map_id`, `to_cell == (0, 0)`) that were never verified. The gated
ROM smoke `tests/test_phase2_rom.py` fails on the first hop with `unreachable`:
`navigate_grid` cannot path to a bogus `from_cell`.

## The crux (explicit hypothesis to resolve first)

The southbound probe `tools/probe_return_portals.py` **stalled** at
route_103 -> Oldale -> route_101: it discovered a clean first hop
(`_PortalSeed((0, 18), (8, 21), 'down', (0, 10), True, (8, 1))`) but then every
south border cell of Oldale returned `blocked`/`unreachable`.

Root cause (user domain knowledge, confirmed): a **ledge (muret)** is crossed by
**walking INTO it and being auto-carried across the map border** — not by
A*-routing to a precise border cell and issuing a single `probe_step`. The
ledge-aware A* (`plan_path_grid`) treats the muret edge as a wall, so the
"navigate to border cell + one step across" model used by both `travel_to` and
the stalled probe is the wrong primitive for a ledge descent. The user reports the
descent is a **straight shot** (hold DOWN on the correct column, no tall grass
needed), which matches the proven **northbound** reach in
`tools/probe_north_exit_truth.py` that used `explore_grid`'s greedy border sweep
(NOT `travel_to`/portals) to reach route_103.

So before wiring durable code, we must **prove which crossing primitive actually
descends south** and **record the real portal chain**.

## Approach

**Approach 1 — durable greedy-descent (`reach_map`), campaign calls it for the
return legs.** Port the proven greedy border-sweep into durable code; discover the
real portals live so no placeholder guessing; battle-proof via the Fighter. This
reuses the ONLY mechanism proven to work (the northbound reach) and is the direct
"do the route in reverse" answer.

(Rejected: Approach 2 patch-the-portal-cells keeps the wrong crossing primitive and
needs both data AND crossing-logic surgery; Approach 3 hybrid fallback is the most
complex, two code paths.)

## Architecture (4 units)

### Unit 1 — Discovery probe (throwaway)

A throwaway probe that, from `states/post_rival.state` with the Fighter and a
healed party, drives a greedy descent to the lab and **records the real portal
chain**: for each hop `(from_map, from_cell, direction, to_map, to_cell,
reversible)`. It also settles the crux: does `explore_grid`'s greedy sweep cross
each southbound border, or is a distinct **hold-DOWN-through-murets** primitive
needed? The probe prints the recorded chain in `_PortalSeed` order and reports the
crossing mechanism that worked.

- Input: `POKEMON_EMERALD_ROM`, `states/post_rival.state`, Fighter checkpoint.
- Output (stdout, not committed): the real portal geometry for each return hop, a
  `REACHED lab = True/False`, and which crossing primitive fired per hop.
- Purpose: de-risk Unit 2 before any durable code. If hold-DOWN is required, the
  probe pins exactly how (which column, how many DOWN presses per map).

This may extend/replace the existing `tools/probe_return_portals.py`. Throwaway:
not covered by durable tests, deleted or left untracked after the chain is frozen.

### Unit 2 — `reach_map` (durable)

A reusable greedy-descent function living beside `travel_to` in
`env/map_traveler.py`:

```
reach_map(emulator, reader, memory, goal_map, *, move_type_fn=None,
          predict=None, max_hops=...) -> str
```

Per hop: snapshot the current map; if it is `goal_map`, return `arrived`. Else
cross the border toward the goal in the hop's direction, record the real `Portal`
in `MapMemory`, and re-evaluate on the new map. Battle-proof: a wild encounter
during a crossing is cleared by the Fighter (same battle handling
`navigate_grid`/`explore_grid` already use); a battle loss short-circuits.

Returns `arrived | battle_lost | battle_timeout | battle_interrupted | stall |
timeout`. `stall` = no crossing found from the current map (distinct from `timeout`
= hop budget exhausted).

**Direction is per-hop, not uniform (gap-check).** The return is NOT a single
"greedy DOWN": it is a ledge descent (DOWN) route_103 -> Oldale -> route_101 ->
Littleroot, then a **door warp UP** into the lab from Littleroot. Two distinct
crossing kinds:

- **Ledge legs (DOWN)** use the primitive the probe proved (hold-DOWN through
  murets / greedy border sweep). These are where the placeholder model broke.
- **The lab-door leg (UP)** is a normal warp tile, NOT a ledge — the existing
  "navigate to the door cell + step across" model (`travel_to`'s crossing, which
  `navigate_grid` already handles for warps) works. Its real door cell is pinned by
  the probe.

`reach_map` therefore takes a **direction hint per map** (the return chain is a
fixed, known sequence). Concretely: it descends (DOWN legs) until it lands on
Littleroot, then crosses the lab door (UP). The direction map for the return chain
is a small constant derived from the probe:
`{route_103: down, Oldale: down, route_101: down, Littleroot: up}`.

`reach_map` records portals as it discovers them, so a subsequent `travel_to`/
`plan_route` over the same `MapMemory` sees a real graph. It does NOT depend on any
pre-seeded portal.

### Unit 3 — Campaign wiring

`run_campaign` dispatches the descent as a **single "reach home" milestone**
handled directly by `reach_map(goal=lab)` — NOT through `execute_order`'s
advance/story modes. This keeps `env/orders.py` untouched: `run_campaign` already
dispatches the `story` mode at the head of its loop, so a `reach` dispatch is a
sibling branch. The milestone carries a `reach: tuple[int, int] | None` goal-map
field (like the existing `story_target`); when set, `run_campaign` calls
`reach_map` and aborts on a non-`arrived` outcome.

The placeholder `_RETURN_PORTALS` and `seed_return_portals(memory)` are **removed**
(portals are discovered live by `reach_map`), unless the probe shows seeding is
still needed as a direction/landing hint — in which case they are replaced with the
real cells the probe pinned. `run_campaign` stays a pure sequencer.

The story milestones (Pokedex / Balls / shoes) are unchanged: once `reach_map`
lands the player at the lab, the existing `story` Order mode (travel_to(cell) then
bounded A-spam until the injected predicate holds) delivers the events.

### Unit 4 — ROM smoke (load-bearing)

`tests/test_phase2_rom.py` runs the **full** campaign from `post_rival.state`:
`reach_map`-driven descent + story A-spam, with the real Fighter. Asserts
`campaign_complete`, `has_pokedex()`, and `has_item(POKE_BALL_ITEM_ID, 5)`
(the shoes flag is already True at post_rival -> idempotent no-op). Dumps
`states/post_phase2.state`. Triple-skips without ROM / Fighter checkpoint /
`post_rival.state`.

## Data flow

```
post_rival.state
  -> run_campaign(PHASE2_CAMPAIGN)
       -> return milestone: reach_map(goal=lab)   # greedy descent, records portals
            route_103 -> Oldale -> route_101 -> Littleroot -> lab
       -> story milestone: execute_order(mode=story, target=has_pokedex)  # A-spam
       -> story milestone: has_item(Ball, 5)  # idempotent post-assert
       -> story milestone: has_running_shoes  # idempotent no-op
  -> assert + dump post_phase2.state
```

## Error handling

- `reach_map` returns a battle outcome -> `run_campaign` aborts that milestone with
  the battle code (same contract as the existing advance/story dispatch).
- `reach_map` returns `stall` -> the campaign cannot progress; surfaced as a
  milestone failure so the smoke fails loudly (no silent papering).
- The probe must actually reach the lab before Unit 2 is coded; a probe failure
  means the crossing primitive is still wrong and the design pauses on the crux.

## Testing

- Unit 2: pure unit tests with a fake reader/emulator that transitions maps on a
  scripted crossing, covering `arrived`, `stall`, `timeout`, and a battle outcome.
  Follows the fake-reader pattern already used by `test_map_traveler`/
  `test_grid_navigator`.
- Unit 3: `test_campaign` gains coverage that a return milestone dispatches
  `reach_map` and that a `stall`/battle outcome aborts.
- Unit 4: the gated ROM smoke is the single load-bearing end-to-end assertion.

## Scope / non-goals

- No change to the RL `env/milestones.py` path.
- No change to the story Order mode or the Phase 2 detectors (already merged).
- No northbound change; only the southbound return uses `reach_map`. (`reach_map`
  is direction-agnostic, but this feature wires only the return legs.)
- The probe is throwaway; only Units 2-4 are durable.
```