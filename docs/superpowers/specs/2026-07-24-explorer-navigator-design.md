# Explorer Navigator (P2) — design

**Date:** 2026-07-24
**Status:** Approved (brainstorming), pending implementation plan
**Milestone:** P2 — the Explorer becomes a navigator (routing + local movement)

## Goal

Give the Explorer the ability to reach an ordered destination. Two layers:

- **Level A — inter-map routing:** on the `MapMemory` graph (nodes = visited
  maps, edges = walked transitions), compute the sequence of maps to cross from
  the current map to a target map. Shortest path on a small graph.
- **Level B — intra-map movement:** inside a single map, move toward a target
  `(x, y)` cell, detecting walls **by observed collision** (pressed a direction,
  `(x, y)` did not change → wall), without reading the tile-behavior byte from
  RAM.

This is the second palier of the hierarchical-RL Explorer. It is the pure
*navigation brick*: no training, no reward, no probe. Testable in isolation on
synthetic graphs/grids, exactly like P1.

## Context

The locked architecture is hierarchical (manager-worker):

- **Strategist** (manager) holds the story and issues orders: a named
  destination + mode (advance/grind/heal) + battle directive.
- **Explorer** (navigation worker + world perception) navigates to the ordered
  destination and self-builds its map of the world. **P1** already gave it
  perception: `WorldReader` (map_id, pos, tile_behavior=TODO-probe) and
  `MapMemory` (observational graph of places, edges, experience labels).
- **Fighter** (battle worker) wins per the directive.

P1 shipped the perception + memory. P2 adds *how to get somewhere* on top of
that memory. It does not wire into the live emulator loop yet — that is P3.

## Scope (P2)

Two isolated, unit-testable components:

1. `env/route_planner.py` — Level A. Shortest path over the `MapMemory` graph.
2. `env/local_navigator.py` — Level B. Observed-collision detector + a learned
   wall grid per map + A* over that grid.

Both operate on plain data (a `MapMemory` graph, a `WallMap`, snapshots). No
emulator, no ROM, no Gym dependency in the planners themselves.

## Non-goals (deferred)

- The live A↔B loop and real emulator execution (entering a map, knowing which
  cell you land on, stepping the emulator, feeding snapshots back) — that is
  **P3** (hierarchical live loop).
- The `tile_behavior` probe. Navigation needs walkability, and walkability is
  learned observationally by bumping. Semantic tile types (grass for grinding,
  water for Surf) are a later probe session, when actually needed.
- Reward, training, capture, items. Emerald-only (game-specific RAM used freely).

## Architecture

### Level A — inter-map routing (`env/route_planner.py`)

The `MapMemory` graph has directed edges (A→B recorded when the player walked
from map A into map B). Routing = breadth-first search over those edges.

```python
def plan_route(
    memory: MapMemory,
    start_map: tuple[int, int],
    goal_map: tuple[int, int],
) -> list[tuple[int, int]] | None:
    ...
```

- Returns the list of map_ids from `start_map` to `goal_map` **inclusive**
  (`[start_map]` when `start_map == goal_map`).
- Returns `None` when no known path exists: `goal_map` never visited, or the
  graph is disconnected between start and goal.
- BFS (not Dijkstra): edges are unweighted (one map hop = one step). Shortest in
  number of map transitions, which is what we want.
- Reads the graph through `MapMemory`'s public surface (`nodes`, `edges()`). Does
  not mutate the memory.

### Level B — intra-map movement (`env/local_navigator.py`)

Three bricks.

**a) Collision detector (observational).**
Compares two snapshots taken around one key press.

```python
DIRECTIONS = ("up", "down", "left", "right")  # or an enum

def resolve_move(
    before: WorldSnapshot,
    after: WorldSnapshot,
    direction: str,
) -> str:  # "moved" | "blocked" | "transition"
    ...
```

- Same `map_id`, `pos` changed → `"moved"`.
- Same `map_id`, `pos` unchanged → `"blocked"` (a wall in `direction` from
  `before.pos`).
- `map_id` changed → `"transition"` (a map edge; Level A's concern, not a wall).

**b) Learned wall grid (`WallMap`, self-built).**
Per `map_id`, a set of blocked directed edges `{((x, y), direction)}`.

- Fills up as collisions are observed.
- **Bidirectional:** bumping `(2, 3)` going up means the wall sits between
  `(2, 3)` and `(2, 2)`, so `(2, 2)` going down is also blocked — record both.
- **Unknown ≠ blocked:** an edge never tested is assumed walkable (optimistic).
  A* re-plans when it later discovers a real wall (replan-on-bump).

```python
class WallMap:
    def block(self, map_id, cell: tuple[int, int], direction: str) -> None: ...
    def is_blocked(self, map_id, cell: tuple[int, int], direction: str) -> bool: ...
```

**c) A* over the learned grid.**

```python
def plan_path(
    wallmap: WallMap,
    map_id: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[str] | None:  # list of directions
    ...
```

- Shortest path in cells, Manhattan-distance heuristic.
- Steps between adjacent cells cost 1; an edge marked blocked in `wallmap` is
  impassable.
- Returns the list of directions from `start` to `goal`, `[]` when
  `start == goal`, `None` when the goal is unreachable given known walls.
- The caller replans (`plan_path` again) whenever a new wall is discovered.

## Data flow (target, wired in P3)

```
MapMemory.graph ──> plan_route (A) ──> list of map_ids to cross
                                            │
                        for each map in the route:
                                            ▼
WallMap (per map) ──> plan_path A* (B) ──> list of directions
                                            │
                    execute one direction, snapshot before/after
                                            ▼
             resolve_move ──> WallMap.block ──> replan if new wall
```

The A↔B junction (knowing which cell you enter the next map on) is live
plumbing = **P3**. P2 delivers A and B testable in isolation on synthetic
graphs/grids.

## Components / files

- `env/route_planner.py` — `plan_route` (BFS over `MapMemory`).
- `env/local_navigator.py` — `resolve_move`, `WallMap`, `plan_path` (A*).
- `tests/test_route_planner.py` — direct path, multi-hop, `None` when
  disconnected, `None` when goal unknown, `start == goal`.
- `tests/test_local_navigator.py` — detector moved/blocked/transition; `WallMap`
  bidirectional blocking + unknown-optimistic; A* simple path, wall detour,
  replan-on-bump, `None` unreachable, `start == goal`.

## Testing strategy

Pure Python, no ROM, no emulator. `plan_route` runs on a `MapMemory` built by
calling `observe(...)` with synthetic snapshots (the P1 API). The navigator runs
on hand-built `WallMap`s and `WorldSnapshot` pairs. Everything deterministic.

## Future (out of scope, noted)

- P3 live loop: wire A→B against the real emulator, snapshot each step, feed
  collisions back into `WallMap`, cross map edges from `plan_route`.
- `tile_behavior` probe when semantic tiles (grass/water) are needed.
- Order-conditioned Explorer reward and training (P3+).
