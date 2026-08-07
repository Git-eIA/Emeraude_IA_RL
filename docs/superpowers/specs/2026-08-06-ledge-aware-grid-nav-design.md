# Ledge-Aware Grid Navigation (Brique 2) — Design

**Date:** 2026-08-06
**Status:** Approved design, pending implementation plan
**Depends on:** P4 Brique 1 (MapGridReader, merged) — provides RAM-decoded map ground truth.

## Problem

Autonomous route_101 exploration is blocked by two defects in the bump-learned
navigation stack:

- **(A) `map_map` thrash.** `_nearest_frontier` runs BFS over edges that were
  never actually walked. A one-way ledge (crossing B→A is forbidden) makes
  `_follow_route` fail → `continue` → the same frontier cell is re-proposed
  forever → `budget_exhausted` instead of `complete`/`left_map`.
- **(B) `WallMap` cannot model a one-way ledge.** Bump-learned edges are
  bidirectional; a ledge you may hop DOWN but never climb UP is a *directed*
  constraint the current model cannot express.

The user's governing terrain fact: one-way ledges are everywhere in Emerald.
The intended route around the mid-route_101 ledge is to pass to the RIGHT
(around the ledge's end) then climb UP — nav must route AROUND ledges, not treat
them as walls.

Brique 1 already reads the true map (walls, grass, directed ledges) from RAM.
This brick makes navigation *consume that ground truth* instead of blind bumping.

## Approach

Read the whole classified grid from RAM, plan over it with a ledge-faithful
graph model, and drive the emulator. No bump-learning, no false walls, ledges
respected by construction.

Five units, each with one responsibility and a well-defined interface. New
modules live alongside the existing bump-nav; consumers migrate; the old code
is removed once unreferenced (keeps the diff traceable and the suite green
during migration).

**Coordinate invariant (load-bearing).** `WorldReader.snapshot().pos` and
`MapGridReader.classify_at(x, y)` share the same coordinate origin (Brique 1:
player at (10,17) indexes the stripped logical grid directly). Every
`plan_path_grid` call relies on this: `start`/`goal` are snapshot-frame `(x, y)`
that index `GridSnapshot.tiles[y][x]`.

**Migration surface (real consumers).** Nothing calls `navigate_to`/`map_map`
directly. `navigate_to` is wrapped by `map_traveler.travel_to`; `map_map` is
wrapped by `world_surveyor.survey_world`. The `WallMap` parameter threads
`travel_to` → `orders.execute_order` → `campaign.run_campaign`. `navigate_grid`
drops `WallMap` (the RAM grid replaces bump-learning), so migration removes the
`WallMap` parameter from `travel_to`, `execute_order`, and `run_campaign`, and
rewires `survey_world` onto `explore_grid`. This cascade is planned as explicit
tasks; the old `local_navigator`/`live_navigator`/`map_explorer` bump-nav is
deleted only once nothing imports it.

**Obtaining the grid reader.** `WorldReader` gains a public `grid_reader`
property exposing its `MapGridReader` (private `self._grid` in Brique 1).
`navigate_grid`/`explore_grid` read `reader.grid_reader` — no re-construction, no
duplicate `read`-fn wiring.

### Unit 1 — `plan_path_grid` (pure pathfinder)

`env/grid_navigator.py`

```python
def plan_path_grid(
    grid: GridSnapshot,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[tuple[int, int], str]] | None = None,
) -> list[str] | None:
    """A* over a GridSnapshot; list of directions start→goal, or None.

    Reuses DELTAS/DIRECTIONS from local_navigator (DRY). Bounded by the finite
    grid (no MAX_EXPANSIONS needed: the node set is width*height). `blocked` is
    an optional set of directed edges (cell, direction) to skip — the live
    navigator's transient NPC-avoidance set (unit 4, G2(a)); default None = none.
    """
```

**Graph semantics (the ledge model, APPROVED):** faithful 2-tile jump.

- Nodes are only *standable* tiles: `FREE` and `GRASS`. `LEDGE_*` and `WALL`
  are never nodes (you cannot stop on them).
- From standable node `C`, direction `d` (delta `Δ`):
  - adjacent `C+Δ` is `FREE`/`GRASS` → normal edge `C→C+Δ`, cost 1 (one press).
  - adjacent `C+Δ` is `LEDGE_d` (arrow matches `d`) **and** landing `C+2Δ` is
    `FREE`/`GRASS` → directed jump edge `C→C+2Δ`, cost 1 (one press).
  - otherwise (`WALL` / out-of-bounds / ledge with wrong arrow / bad landing)
    → blocked.

This makes ledges strictly one-way, exactly matching the terrain: `LEDGE_DOWN`
is traversable only while moving `down`, never `up`.

### Unit 2 — `GridSnapshot` (immutable classified-grid value object)

`env/grid_snapshot.py` (its own module: a value object that both the navigator
and `MapMemory` depend on — keeping it standalone avoids a `grid_navigator` ↔
`map_memory` import cycle)

```python
@dataclass(frozen=True)
class GridSnapshot:
    map_id: tuple[int, int]
    width: int
    height: int
    tiles: tuple[tuple[TileKind, ...], ...]  # [y][x], WALL-pinned, never None

    def classify_at(self, x: int, y: int) -> TileKind:
        """Bounds-checked; out-of-range returns WALL."""

@classmethod
def from_reader(cls, grid_reader, map_id) -> GridSnapshot | None:
    """Capture MapGridReader.grid(); None if the map is not ready.

    map_id is a label the caller supplies (the snapshot's map_id); the grid
    reader always decodes the currently-loaded map and does not verify the two
    agree — the caller reads pos and map_id from the same snapshot.
    """
```

Decouples the pure planner from the live reader: `plan_path_grid` consumes a
`GridSnapshot`, never touches ROM. `TileKind` is reused from Brique 1
(`env/map_grid_reader.py`).

### Unit 3 — `MapMemory` grid store (per-map remembered grid)

`env/map_memory.py` (extend, additive)

```python
def remember_grid(self, snap: GridSnapshot) -> None:
    """Store the snapshot keyed by map_id; last-write-wins."""

def grid_for(self, map_id) -> GridSnapshot | None:
    """Return the remembered grid, or None if never seen."""
```

This is the map memory the Explorer needs: learn a map once, recall it without
re-probing. Additive — portals/labels untouched. `GridSnapshot` is imported
under `TYPE_CHECKING` (annotations only; `remember_grid` reads `snap.map_id` at
runtime, no runtime import needed) so `map_memory` stays free of any cycle.

### Unit 4 — live ledge-aware navigator

`env/grid_navigator.py`

`navigate_grid(...)` replaces `navigate_to`. Loop:
1. live snapshot (`snapshot_settled`) → `GridSnapshot.from_reader` →
   `remember_grid`.
2. `plan_path_grid(snap, pos, target, blocked=blocked)` (see below).
3. press first move (`probe_step`), classify (moved/blocked/transition).

Because the plan already knows walls+ledges from the RAM grid, it learns **zero
false walls** and respects one-way ledges by construction. Reuses
`handle_battle_interruption`, portal recording, heal/grass observation. Same
return set as `navigate_to`: `arrived | unreachable | left_map | timeout |
battle_lost | battle_timeout | battle_interrupted`.

**Transient-block set (G2(a) — dynamic obstacles).** The RAM grid is static: an
NPC standing on a FREE tile is not modelled, so a planned move onto it presses
and does not move (`blocked`). Without mitigation the loop re-plans the same
path forever → `timeout` (the very thrash we are killing, re-triggered by NPCs).
Mitigation: `navigate_grid` keeps a per-run `blocked: set[tuple[cell, str]]` of
directed edges whose planned press unexpectedly failed to move; it is passed to
`plan_path_grid` which skips those edges, so the next plan routes around the
obstruction. The set is **per-call only** (not persisted, not a `WallMap`): a
transient NPC must not permanently scar the remembered grid. `plan_path_grid`
therefore takes an optional `blocked` parameter (default empty) that the pure
tests exercise directly. Full dynamic-obstacle modelling stays deferred; this is
just the minimum so a live NPC degrades to a detour, not a hang.

### Unit 5 — grid-based discovery loop

`env/grid_explorer.py`

`explore_grid(...)` replaces `map_map`. **Reframing:** with the RAM grid the
map *geometry* (walls, grass, ledges) is known the instant the map loads — there
is nothing left to discover about the terrain. The grid does **not** contain
warp destinations, so the only thing left to probe is **portals**: which border
tiles warp, and where to.

So `explore_grid` (1) reads the current grid once and `remember_grid`s it, then
(2) routes with `plan_path_grid` to each reachable border `FREE`/`GRASS` cell and
**steps outward** off the map edge; a `transition` records a portal (reusing the
step-back reversibility check from `map_map`), a `blocked`/no-op means that edge
is not a portal. `complete` = every reachable border-edge candidate has been
tested. The RAM grid kills the thrash: geometry is never re-probed and blocked
edges are never re-proposed. Returns `complete | budget_exhausted | left_map` +
battle outcomes.

## Error handling

- Map not ready (`from_reader` None) → idle a beat and retry, bounded (mirrors
  `snapshot_settled`).
- Goal unreachable (`plan_path_grid` None) → `unreachable`.
- Unexpected map transition mid-move → `left_map`.

## Testing strategy

- **Pure (no ROM):**
  - `plan_path_grid`: straight path, wall routed around, **one-way ledge:
    descend OK / climb blocked**, route around ledge via the right, unreachable
    → None, `blocked` edge forces a detour.
  - `GridSnapshot`: `classify_at` out-of-bounds = WALL; `from_reader` None when
    map not ready.
  - `MapMemory.remember_grid`/`grid_for`.
- **Live (fakes):** `navigate_grid` crosses a ledge in the correct direction,
  refuses the wrong direction, and **detours around a fake NPC** (a planned FREE
  tile whose press does not move → transient-block set → reroute, no timeout);
  `explore_grid` does not thrash and records a border portal.
- **ONE gated ROM smoke** on route_101: load `states/post_starter.state`,
  `explore_grid` reveals the grid and crosses the ledge northward — where the
  old `map_map` stalled at `budget_exhausted`.

## Scope / non-goals

- **In:** the five units; the migration cascade (drop `WallMap` from
  `travel_to`/`execute_order`/`run_campaign`, rewire `survey_world` onto
  `explore_grid`); a public `WorldReader.grid_reader`; a per-run transient-block
  set for live NPC detours; removal of unreferenced bump-nav; pure+live+one
  gated ROM smoke.
- **Deferred:** dynamic obstacles (PNJ/objects) — the grid is static-only for
  now (user: "on commence en A on fera le B plus tard"). The multi-hour live
  cross-map run to route_103 → manual runbook (same pattern as prior briques).
