# P3 Step 2 — Inter-Map Navigation Design

## Context

P3 wires the Explorer's static bricks (P1 perception, P2 pathfinding) to the
real emulator. Step 1 built `navigate_to`: walk to a target cell **on the
current map**, learning walls by collision and replanning on each bump.

Step 2 answers the next question: **how do we walk from one map to another?**

The Explorer already knows *that* maps connect (`MapMemory.edges()` records
`(map_A, map_B)` when the player crosses). But it does **not** know *where*
the crossing happens — which cell on map A, pressed in which direction, leads
to map B. Without that, we cannot walk to the door. This step adds that missing
piece and the travel loop that chains door-to-door.

## Scope

**In scope (this step):**
- Extend `MapMemory` to remember **portals**: `(from_map, from_cell, direction)
  → to_map`.
- Record portals **live**, the moment the player crosses a map boundary.
- `travel_to(...)`: given a goal map + goal cell, walk map-by-map to it, using
  known portals, reusing `navigate_to` for each intra-map leg.
- Travel over **known territory only** (maps and doors already crossed at least
  once).

**Out of scope (deferred to later P3 steps):**
- **Mapping mode** — wandering to *discover* new doors. If a portal needed for
  the route is unknown, `travel_to` gives up cleanly (`"unknown_route"`); it
  does not explore.
- Strategist order interface, moving story out of `PokemonEmeraldEnv`, the full
  hierarchical loop.
- Tile-behavior probe / cell semantics (still `TODO(probe)` from P1).

## The problem, concretely

To go from map A to map C where the graph is A → B → C:
1. On map A, walk to the exact cell that borders map B, press the crossing
   direction → land on B.
2. On map B, walk to the cell that borders map C, press → land on C.
3. On map C, walk to the final goal cell.

P2's `plan_route` gives the map sequence `[A, B, C]`. What was missing is step
1 and 2's "exact cell + direction". That is the **portal**.

## Design

### 1. Portals in `MapMemory` (extends P1)

A portal is one directed border crossing:

```python
@dataclass(frozen=True)
class Portal:
    from_cell: tuple[int, int]   # cell on from_map you step off of
    direction: str               # d-pad direction pressed to cross
    to_map: tuple[int, int]      # map you land on
```

New state and API on `MapMemory`:

```python
# (from_map, to_map) -> Portal   (one remembered crossing per map pair)
self._portals: dict[
    tuple[tuple[int, int], tuple[int, int]], Portal
] = {}

def record_portal(
    self,
    from_map: tuple[int, int],
    from_cell: tuple[int, int],
    direction: str,
    to_map: tuple[int, int],
) -> None:
    """Remember that leaving from_map at from_cell going `direction` reaches to_map."""

def portal(
    self, from_map: tuple[int, int], to_map: tuple[int, int]
) -> Portal | None:
    """The known crossing from from_map to to_map, or None if never crossed."""
```

Notes:
- **Idempotent / last-write-wins:** re-crossing the same border overwrites the
  stored portal. One crossing per map pair is enough to travel; we do not model
  multiple doors between the same two maps in this step (YAGNI — a later mapping
  step can generalize if needed).
- Portals are stored **only** in `MapMemory`; it already owns the map graph.
  `record_portal` also ensures the edge exists (so `edges()` and `portal()`
  never disagree).
- `observe()` is left as-is (it still records edges from the passive
  `_prev_map_id` transition). The live layer is responsible for calling
  `record_portal` with the richer cell+direction info that `observe` does not
  have. This keeps `observe` a pure passive recorder and avoids guessing the
  crossing direction from position deltas.

### 2. Recording portals live

The live navigator already detects a crossing: in `navigate_to`, when
`_press_until_moved` returns `"transition"` we return `"left_map"`. At that
exact moment we know everything a portal needs:
- `before.map_id` = the map we left (`from_map`)
- `before.pos` = the cell we stepped off (`from_cell`)
- `direction` = the key we pressed
- the map we landed on = the **next** snapshot's `map_id` (`to_map`)

The one place that owns all four facts (`from_map`, `from_cell`, `direction`,
and — after one more snapshot — `to_map`) is inside `navigate_to`, at the moment
it sees a transition. So that is where recording belongs.

**Decision:** give `navigate_to` an optional `memory: MapMemory | None = None`
argument. When a transition is detected and `memory is not None`, read one
settled snapshot to learn `to_map` and call `memory.record_portal(before.map_id,
before.pos, direction, to_map)` before returning `"left_map"`. When `memory is
None` (step-1 callers, existing tests), behavior is unchanged: no snapshot, no
recording, same `str` return type. This adds portal recording without changing
the return type and without a second public function.

### 3. `travel_to` — the door-to-door loop (`env/map_traveler.py`)

```python
def travel_to(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    goal_map: tuple[int, int],
    goal_cell: tuple[int, int],
    max_hops: int = 20,
) -> str:
    """Walk map-by-map to goal_cell on goal_map. Known territory only.

    Returns:
      "arrived"        — reached goal_cell on goal_map
      "unknown_route"  — no known route, or a portal on the route is unknown
      "unreachable"    — a portal cell is walled off on its map
      "lost"           — landed on an unexpected map (route diverged)
      "timeout"        — exceeded max_hops
    """
```

Algorithm (bounded, no recursion — code-safety rule #2):

1. Snapshot current position. If `map_id == goal_map`, delegate straight to
   `navigate_to(target=goal_cell)` and return its result mapped to travel
   outcomes.
2. `route = plan_route(memory, current_map, goal_map)`. If `None` →
   `"unknown_route"`.
3. Loop over the route hops, at most `max_hops` times:
   - Current map = first element; next map = second.
   - `p = memory.portal(current_map, next_map)`. If `None` → `"unknown_route"`
     (door not yet discovered — mapping mode's job, deferred).
   - **Reach the door:** `navigate_to(target=p.from_cell)`.
     - `"unreachable"` → return `"unreachable"`.
     - `"timeout"` → return `"timeout"`.
     - `"arrived"` → we are standing on the door cell; continue to cross.
   - **Cross the door:** the target that makes `navigate_to` press `p.direction`
     once is the neighbour cell `p.from_cell + DELTAS[p.direction]`. That
     neighbour is off the current map, so the first press transitions.
     Call `navigate_to(target=<that neighbour>, memory=memory)`; it presses,
     detects the transition, records the portal, and returns `"left_map"`, which
     the traveler treats as a successful crossing.
   - After crossing, snapshot. If new map != expected next map → `"lost"`.
   - Recompute the remaining route from the new position (re-plan each hop keeps
     it robust if the graph or our position surprises us).
4. If the loop exits without arriving → `"timeout"`.

### File structure

- **`env/map_memory.py`** (modify): add `Portal`, `self._portals`,
  `record_portal`, `portal`.
- **`env/live_navigator.py`** (modify): add optional `memory` param to
  `navigate_to`; record portal on transition.
- **`env/map_traveler.py`** (new): `travel_to` + the door-to-door loop. Imports
  `plan_route` (P2 route planner), `navigate_to` (P3 step 1), `DELTAS` (P2
  local navigator).

Keeping `travel_to` in its own file keeps `live_navigator` focused on the
single-map loop and `map_memory` focused on storage.

## Testing strategy

Same philosophy as P1/P2/P3-step-1: fast unit tests with a fake, plus one
ROM-gated smoke test.

### Unit tests — extend `FakeWorld` to multiple maps

`tests/test_live_navigator.py`'s `FakeWorld` already models one hidden grid with
walls, turns, map-flips, and None-frames. Extend it (in a shared or copied
helper) to a **multi-map** fake: a dict of grids keyed by `map_id`, with border
cells that flip `map_id` and drop the player onto a start cell of the next map
when a given direction is pressed. This lets the traveler actually walk
door-to-door.

**`tests/test_map_memory.py`** (portals):
- `record_portal` then `portal` returns the stored `Portal`.
- `portal` for an unrecorded pair returns `None`.
- `record_portal` also creates the edge (`(from_map, to_map) in edges()`).
- last-write-wins: recording twice overwrites.

**`tests/test_live_navigator.py`** (portal recording):
- `navigate_to` with `memory=` set: after a map transition, the portal is
  recorded with the correct `from_cell` / `direction` / `to_map`.
- `navigate_to` with `memory=None`: no portal recorded, return unchanged
  (regression guard for step-1 behavior).

**`tests/test_map_traveler.py`** (new):
- Same map: `goal_map == current` → delegates to `navigate_to`, returns
  `"arrived"`.
- Two-map hop with a **pre-injected** portal in `MapMemory`: walks to the door,
  crosses, reaches the goal cell → `"arrived"`.
- Three-map chain A→B→C with pre-injected portals → `"arrived"`.
- Route exists in `edges()` but a portal is missing → `"unknown_route"`.
- No route at all (goal map never visited) → `"unknown_route"`.
- Door cell walled off on its map → `"unreachable"`.
- Landing on an unexpected map → `"lost"`.

### ROM smoke test — `tests/test_map_traveler_rom.py`

`@pytest.mark.skipif(not ROM)`. This is where step 1's Minor M4 (the truck
fixture is a poor open-map) gets addressed: pick a savestate on an **open map
with a real doorway** so the crossing is load-bearing. Load it, build
`WorldReader` + `MapMemory` + `WallMap`, walk the player through one real door
once (recording the portal live via `navigate_to(memory=...)`), then assert
`travel_to` back returns a legal outcome (`"arrived"` if we target the start
cell, or at least not a crash/`"lost"`). If no suitable open-map savestate
exists yet, capture one (`states/open_map.state`) as part of the plan.

## Bounded-loops audit (code-safety rule #2)

- `travel_to`'s hop loop is bounded by `max_hops`.
- Each `navigate_to` leg is bounded by its own `max_steps` (step 1).
- No recursion; `plan_route` (BFS) and `plan_path` (A*, `MAX_EXPANSIONS`) are
  already bounded.

## Open questions / risks

- **`plan_route` return shape.** Confirm it returns `list[tuple[int,int]]` of
  map ids (the plan will verify against `env/route_planner.py`).
- **Door neighbour target trick.** Relies on `navigate_to` pressing toward an
  off-map neighbour and getting a transition on the first press. If the emulator
  needs the player to already *face* the door (turn-first), step 1's
  `TURN_RETRIES` handles it. The unit fake must model the same turn-first
  behavior at borders to keep the test honest.
- **`"lost"` recovery.** This step only *reports* `"lost"`; recovering (re-plan
  from wherever we landed) is deferred. The loop already re-plans each hop, so a
  benign one-cell surprise self-corrects; `"lost"` is reserved for landing on a
  genuinely unexpected map.
