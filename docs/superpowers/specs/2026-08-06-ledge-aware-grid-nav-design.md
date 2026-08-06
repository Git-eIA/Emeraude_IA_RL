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

### Unit 1 — `plan_path_grid` (pure pathfinder)

`env/grid_navigator.py`

```python
def plan_path_grid(
    grid: GridSnapshot,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[str] | None:
    """A* over a GridSnapshot; list of directions start→goal, or None.

    Reuses DELTAS/DIRECTIONS from local_navigator (DRY). Bounded by the finite
    grid (no MAX_EXPANSIONS needed: the node set is width*height).
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
    """Capture MapGridReader.grid(); None if the map is not ready."""
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
2. `plan_path_grid(snap, pos, target)`.
3. press first move (`probe_step`), classify (moved/blocked/transition).

Because the plan already knows walls+ledges from the RAM grid, it learns **zero
false walls** and respects one-way ledges by construction. Reuses
`handle_battle_interruption`, portal recording, heal/grass observation. Same
return set as `navigate_to`: `arrived | unreachable | left_map | timeout |
battle_lost | battle_timeout | battle_interrupted`.

### Unit 5 — grid-based discovery loop

`env/grid_explorer.py`

`explore_grid(...)` replaces `map_map`. Instead of frontier-BFS over unproven
edges, it reads the current map's full RAM grid at once, remembers it
(`remember_grid`), then routes (`plan_path_grid`) toward each reachable,
unvisited `FREE`/`GRASS` cell to reveal border portals. The RAM grid kills the
thrash: no infinite re-proposal of blocked edges. Returns
`complete | budget_exhausted | left_map` + battle outcomes.

## Error handling

- Map not ready (`from_reader` None) → idle a beat and retry, bounded (mirrors
  `snapshot_settled`).
- Goal unreachable (`plan_path_grid` None) → `unreachable`.
- Unexpected map transition mid-move → `left_map`.

## Testing strategy

- **Pure (no ROM):**
  - `plan_path_grid`: straight path, wall routed around, **one-way ledge:
    descend OK / climb blocked**, route around ledge via the right, unreachable
    → None.
  - `GridSnapshot`: `classify_at` out-of-bounds = WALL; `from_reader` None when
    map not ready.
  - `MapMemory.remember_grid`/`grid_for`.
- **Live (fakes):** `navigate_grid` crosses a ledge in the correct direction,
  refuses the wrong direction; `explore_grid` does not thrash.
- **ONE gated ROM smoke** on route_101: load `states/post_starter.state`,
  `explore_grid` reveals the grid and crosses the ledge northward — where the
  old `map_map` stalled at `budget_exhausted`.

## Scope / non-goals

- **In:** the five units, consumer migration (orders.py/campaign.py), removal of
  unreferenced bump-nav, pure+live+one gated ROM smoke.
- **Deferred:** dynamic obstacles (PNJ/objects) — the grid is static-only for
  now (user: "on commence en A on fera le B plus tard"). The multi-hour live
  cross-map run to route_103 → manual runbook (same pattern as prior briques).
