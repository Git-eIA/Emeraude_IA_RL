# M6 — Intro Milestones Design (truck start)

**Date:** 2026-07-22
**Status:** Approved by user

## Context

The starting savestate `states/initial.state` was re-recorded **inside the moving
truck at the very start of a new game** (map `(25, 40)` InsideOfTruck, position
`(2, 2)`, 0 Pokémon, 0 badges). The previous M5 design assumed a post-intro
start; evaluation showed the agent stuck indoors because the whole scripted
intro (mom, clock, going back outside) still had to be played.

User decision: **do not skip the intro** — the long-term goal is an agent that
finishes the game entirely on its own, intro included. We therefore extend the
milestone chain backwards to guide the agent through the intro, and retrain
from scratch.

## Milestone chain (one-shot per episode, in `starter_milestones()`)

| # | Name | Condition (PlayerState) | Points | Terminal |
|---|------|--------------------------|--------|----------|
| 1 | `exit_truck` | `(map_group, map_num) == (0, 9)` (Littleroot) | +5 | no |
| 2 | `enter_house` | map in `{(1, 0), (1, 2)}` (Brendan's / May's house 1F) | +5 | no |
| 3 | `clock_set` | `clock_set is True` (FLAG_SET_WALL_CLOCK = 0x51) | +15 | no |
| 4 | `back_outside` | map == `(0, 9)` AND `clock_set` | +10 | no |
| 5 | `reach_route_101` | map == `(0, 16)` | +20 | no |
| 6 | `starter_obtained` | `party_count >= 1` | +100 | **yes** |

Total chain: **155 points**, dwarfing per-tile exploration (+1, ~60-80/episode
observed). Milestones 1-2 are the user's "+5 quitter le camion / +5 rentrer
dans la maison".

## Design decisions

1. **`PlayerState` gains `clock_set: bool = False`.** `EmeraldReader.player_state()`
   reads it via the existing `_flag(sb1, FLAG_SET_WALL_CLOCK)` helper. This keeps
   milestone conditions pure `Callable[[PlayerState], bool]` — no reader closure.
2. **`FLAG_SET_WALL_CLOCK = 0x51`** pinned from pret/pokeemerald
   `include/constants/flags.h`. Flags array offset unchanged (`0x1270`), so
   byte index `0x51 // 8 = 10`, bit `1`.
3. **Map IDs** from pret `data/maps/map_groups.json`, InsideOfTruck `(25, 40)`
   verified empirically by RAM read on the new savestate.
4. **FakeEmulator** gets a `clock_set` attribute plus a flags-byte branch; its
   default map moves from `(1, 2)` (now May's house → would fire `enter_house`)
   to neutral `(0, 10)` (Oldale). `make_state` test defaults likewise move from
   `(0, 9)` (would fire `exit_truck`) to `(0, 10)`.
5. **Retrain from scratch.** The 500k checkpoint learned a tile-farming local
   optimum on a different start state; it is kept on disk but not resumed.
6. **Exploration-vs-speed:** keep tile reward at +1 (needed to find the
   objectives at all); gamma = 0.99 already pressures for speed. Reserve levers
   if the agent farms tiles: per-step penalty (-0.01) and/or reduced tile reward.
   Not applied now.
7. If the agent stalls at the clock, a future `upstairs` milestone
   (maps `(1, 1)` / `(1, 3)`) can be inserted — out of scope for M6.

## Addendum M6.1 (2026-07-22, approved) — anti-farming levers activated

After 3M steps the intro chain is 100% learned (back_outside 5/5) but
`reach_route_101` never fires: the agent tile-farms Littleroot (345-410
tiles/ep) and hovers at y=2-4, one step short of the northern exit (reached
y=0 once, never crossed). Both reserve levers are now applied:

1. **Revisit penalty:** `ExplorationTracker.update()` returns
   `REVISIT_PENALTY = -0.01` for already-visited tiles (first visit stays +1).
   Standing still also pays the penalty — intentional.
2. **New milestone `north_littleroot` (+10):** `clock_set` AND map `(0, 9)`
   AND `y <= 1` (`NORTH_LITTLEROOT_MAX_Y`), inserted between `back_outside`
   and `reach_route_101`. Chain total becomes **165 points**.

Training resumes from the 3M checkpoint (intro knowledge kept); reward-scheme
change does not require a from-scratch retrain.

## Addendum M6.2 (2026-07-23, approved) — meet_rival unlock chain

After the M6.1 run (4.5M steps) the agent reaches the north edge (4/5) but
`reach_route_101` still never fires: **the game itself blocks the exit**. A twin
NPC pushes the player back while `VAR_LITTLEROOT_TOWN_STATE == 0`. The unlock
(verified empirically on BPEF with a scripted probe, 2026-07-23) is: enter the
rival's house (door (14,8)), go upstairs, press A facing the Pokéball at (5,4)
→ cutscene sets `VAR_LITTLEROOT_TOWN_STATE = 1` → the twin steps aside
(she moves to (10,1); passage is via x=11, where (11,1) triggers a short
dialog setting the var to 2).

### Event vars on BPEF (validated by probe)

Vars live at `SaveBlock1 + 0x139C`, indexed `(var_id - 0x4000) * 2`, u16 LE.
`VAR_LITTLEROOT_INTRO_STATE (0x4092)` stepped 0→7 during the intro at this
offset, confirming it for the French ROM. `VAR_LITTLEROOT_TOWN_STATE = 0x4050`.

### Design (user-approved option B — guided sub-chain)

1. **`PlayerState` gains `town_state: int = 0`**, read via a new
   `EmeraldReader._var()` helper (same pattern as `clock_set`/`_flag`).
2. **Three milestones** inserted between `back_outside` and `north_littleroot`:

| Name | Condition | Points |
|------|-----------|--------|
| `enter_rival_house` | `clock_set` AND map == `(1, 2)` (May's house 1F) | +5 |
| `rival_upstairs` | `clock_set` AND map == `(1, 3)` (May's house 2F) | +5 |
| `meet_rival` | `town_state >= 1` | +15 |

Chain total: 165 → **190 points**, 10 milestones. `clock_set` gates the two
house milestones so they only pay after the intro (the houses are enterable
before). FakeEmulator gains a `town_state` attribute; `build_memory` grows its
SaveBlock1 buffer to `0x1600` to cover the vars array.
3. **Resume training from the 4.5M checkpoint** (reward-scheme change only).

## Testing

- Unit: flag reading (`build_memory` gains `clock_set` param), each new
  milestone condition, full-chain sum = 155, tracker one-shot semantics intact.
- Env: chain test on FakeEmulator (truck → Littleroot → house → clock → outside).
- Real ROM (gated by `POKEMON_EMERALD_ROM`): savestate reads map `(25, 40)`,
  `clock_set` False, empty party.
- Sanity training run (~20k steps) before the long run.
