# P3 Step 3 — Mapping Mode (`map_map`) Design

## Context

P3 wires the Explorer's static bricks (P1 perception, P2 pathfinding) to the
real emulator.

- **Step 1** built `navigate_to`: walk to a target cell **on the current map**,
  learning walls by collision, replanning on each bump.
- **Step 2** added **portals** (`MapMemory.record_portal` / `portal`) and
  `travel_to`, which chains known doors from map to map.

Both steps only ever learn **passively**: they record a wall or a portal when
they happen to bump into it while going somewhere else. `travel_to` explicitly
refuses to explore — an unknown door on the route returns `"unknown_route"`.

Step 3 adds the missing **active** behavior: **deliberately wander one map to
discover everything about it** — every wall, and every door leading out. This is
what fills `MapMemory` with the portals that `travel_to` later relies on.

## Scope

**In scope (this step):**
- `map_map(...)`: explore **one** map to exhaustion by frontier search, learning
  its walls into a `WallMap` and recording every door it finds as a portal in
  `MapMemory`.
- Reuse P2 (`WallMap`, `OPPOSITE`, `DELTAS`) and P3 step-1 primitives (button
  choreography, snapshot settling) rather than duplicating them.
- Produce a real open-map savestate (`states/open_map.state`) **automatically**
  (a small tool driving the trained Explorer), so this step's ROM smoke — and
  step 1's deferred Minor M4 — become load-bearing.

**Out of scope (deferred to later P3 steps):**
- **Multi-map sweeping** — mapping a whole connected region in one call.
  `map_map` maps exactly one map and stops; the "for each new door, `travel_to`
  + `map_map`" orchestration belongs to a later step / the Strategist.
- **Building-warp round-tripping** — non-reversible doors (house entrances) are
  recorded but not re-entered; discovering them ends the run cleanly. Only
  reversible **border connections** (walking off a route/town edge) are fully
  handled this step.
- Tile-behavior probe / cell semantics (still `TODO(probe)` from P1), reward,
  training, the Strategist order interface.

## The idea, concretely

The Explorer has no pre-loaded map. It perceives via RAM (where am I, which
cell) and **builds knowledge by walking**. `map_map` is the "go survey this
place now" behavior:

1. Stand on a cell. It is now *reached*.
2. Look at its four directions. Any direction not yet *tried* and not a known
   wall is a **frontier** — the edge of the unknown.
3. Path (over cells we already walked) to the nearest frontier cell, then
   **probe** its unknown direction with a single press:
   - **moved** → a new walkable cell; it joins *reached* and grows the frontier.
   - **blocked** → a wall; record it in `WallMap`.
   - **transition** → a **door**; record the portal, then step back through the
     (reversible) border to keep surveying this map.
4. Repeat until no frontier remains (`"complete"`) or the step budget is spent
   (`"budget_exhausted"`).

This is the textbook frontier-based exploration of an unknown grid, specialized
to Emerald's "learn walkability by bumping" model.

## Design

### Shared primitive extraction (small refactor in `env/live_navigator.py`)

`navigate_to` already contains exactly the "press one direction, tell
turn/wall/move apart, classify" logic (`_press_until_moved`) and the
"snapshot skipping relocation None frames" logic (`_snapshot_settled`).
`map_map` needs both. To avoid duplicating the GBA button-debounce choreography
(`STEP_FRAMES` / `RELEASE_FRAMES` / `TURN_RETRIES`), promote these two helpers to
public names and have both modules import them:

- `_press_until_moved` → **`probe_step(emulator, reader, before, direction) -> str`**
  (returns `"moved" | "blocked" | "transition"`).
- `_snapshot_settled` → **`snapshot_settled(reader) -> WorldSnapshot | None`**.

`navigate_to`'s behavior is unchanged; only the two internal call sites are
renamed. The constants (`STEP_FRAMES`, `RELEASE_FRAMES`, `TURN_RETRIES`,
`SETTLE_TRIES`) stay in `live_navigator` and are the single source of timing.

### `map_map` (new file `env/map_explorer.py`)

```python
def map_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    target_map: tuple[int, int],
    max_steps: int = 2000,
) -> str:
    """Survey `target_map` by frontier search: learn every wall and record
    every door as a portal. Known-walkable repositioning only.

    Returns:
      "complete"          — frontier exhausted; the map is fully known
      "budget_exhausted"  — hit max_steps before the frontier emptied
      "left_map"          — crossed a non-reversible door and could not return
    """
```

**Internal state (frontier bookkeeping):**
- `reached: set[tuple[int, int]]` — cells the player has stood on.
- `tried: set[tuple[tuple[int, int], str]]` — `(cell, direction)` edges already
  resolved (moved, blocked, or door).

A direction `d` from a reached `cell` is **frontier** iff
`(cell, d) not in tried` **and** `not wallmap.is_blocked(target_map, cell, d)`.

**Main loop (bounded by `max_steps` — code-safety rule #2):**

Each iteration = "reposition to one frontier cell, probe one unknown edge".

1. `here = snapshot_settled(reader)`; if `None`, idle one beat and continue.
2. If `here.map_id != target_map` → `"left_map"` (safety; the survey should
   never wander off the target map except via a door it failed to return from).
3. `reached.add(here.pos)`.
4. **Pick the next frontier** via a BFS **restricted to `reached` cells**
   (`_nearest_frontier`, below): returns `(route, frontier_cell, direction)` —
   the list of directions from `here.pos` to `frontier_cell` over already-walked
   cells, plus the unexplored `direction` to probe there. If there is no
   frontier anywhere in `reached` → `"complete"`.
5. **Reposition:** press each direction in `route` (all known-walkable, so each
   is expected to `move`). If a press unexpectedly fails to move — the world
   surprised us — abandon this iteration and re-loop (the next snapshot re-grounds
   us). Repositioning never steps onto an unknown or door cell, so it can never
   accidentally cross a portal.
6. **Probe the frontier edge:** `outcome = probe_step(emulator, reader, before,
   direction)`; add `(frontier_cell, direction)` to `tried`.
   - `"moved"` → nothing to record here; the new cell is picked up as `reached`
     on the next iteration's snapshot.
   - `"blocked"` → `wallmap.block(target_map, frontier_cell, direction)`.
   - `"transition"` → `landed = snapshot_settled(reader)`; if not `None`,
     `memory.record_portal(target_map, frontier_cell, direction, landed.map_id)`.
     Then **step back**: `probe_step(..., landed, OPPOSITE[direction])` and
     `returned = snapshot_settled(reader)`. If `returned` is `None` or
     `returned.map_id != target_map`, the door was not reversible → `"left_map"`.
     Otherwise continue surveying.
7. After `max_steps` iterations without emptying the frontier → `"budget_exhausted"`.

**Why BFS over `reached` (not `plan_path`):** `plan_path` (P2) plans over the
*optimistic* infinite grid, treating unknown cells as walkable. Repositioning
over that could route the player through an unknown cell that turns out to be a
door, crossing it by accident. Restricting repositioning to cells we have
actually walked guarantees the only step into the unknown is the deliberate
frontier probe.

```python
def _nearest_frontier(
    reached: set[tuple[int, int]],
    tried: set[tuple[tuple[int, int], str]],
    wallmap: WallMap,
    target_map: tuple[int, int],
    start: tuple[int, int],
) -> tuple[list[str], tuple[int, int], str] | None:
    """BFS over reached cells from `start`; return (route, cell, direction) for
    the nearest reached cell that still has an unexplored, non-walled direction,
    or None if the frontier is empty."""
```

### File structure

- **`env/live_navigator.py`** (modify): rename `_press_until_moved` → `probe_step`
  and `_snapshot_settled` → `snapshot_settled` (public); update the two internal
  call sites. No behavior change.
- **`env/map_explorer.py`** (new): `map_map` + `_nearest_frontier`. Imports
  `probe_step`, `snapshot_settled`, `RELEASE_FRAMES` from `live_navigator`;
  `WallMap`, `DELTAS`, `OPPOSITE`, `DIRECTIONS` from `local_navigator`;
  `MapMemory` from `map_memory`.
- **`tools/capture_open_map.py`** (new): drive the trained Explorer
  (`checkpoints/ppo_emerald_final.zip`) from `states/initial.state` a handful of
  steps until it stands on an overworld map, then save `states/open_map.state`.
  One-time artifact generation; run once, committed.

## Testing strategy

Same philosophy as P1/P2/P3 steps 1–2: fast unit tests with a fake, plus one
ROM-gated smoke.

### Unit tests — `tests/test_map_explorer.py`

Reuse the single-map-with-borders fake pattern (mirror `MultiMapWorld` /
`FakeWorld`): a hidden grid with `walls` and optional reversible `borders`
`(cell, direction) -> (to_map, entry_cell)` that flip `map_id` and step back on
the opposite press. Cases:

- **Complete survey of a sealed room:** a small fully-walled grid → `"complete"`;
  `reached` equals every walkable cell; `wallmap` holds every boundary wall.
- **Frontier is not re-probed:** after `"complete"`, assert the total number of
  probes equals the number of edges (each `(cell, direction)` tried at most once)
  — no wasted re-probing of a resolved edge.
- **Door discovered and recorded, survey continues:** a reversible border →
  `map_map` records the portal (`memory.portal(target, to_map)` is set with the
  right `from_cell`/`direction`), steps back, and still finishes `"complete"`
  with the rest of the room mapped.
- **Non-reversible door ends the run:** a border whose opposite press does *not*
  return to `target_map` → `"left_map"`, but the portal is recorded first.
- **Budget exhausted:** a grid larger than a tiny `max_steps` → `"budget_exhausted"`.

### ROM smoke — `tests/test_map_explorer_rom.py`

`@pytest.mark.skipif(not POKEMON_EMERALD_ROM)`. Load `states/open_map.state`
(produced by `tools/capture_open_map.py`), record the start cell, build
`WorldReader` + `MapMemory` + `WallMap`, run `map_map(target_map=<the open map>,
max_steps=<small, e.g. 40>)`. Assert a **load-bearing** result: the run returns a
legal outcome without crashing **and** learning actually happened, checked only
through externally visible state (`reached` is private to `map_map`):
`snapshot_settled(reader).pos != start_cell` (the player moved) **or** at least
one wall was learned (`wallmap.is_blocked(...)` true for some probed edge) **or**
a portal was recorded (`memory.portal(...)` non-`None`). On a real open map this
exercises real presses, real collisions, and real snapshots, unlike the in-truck
`initial.state`. This is the fixture step 1's Minor M4 was waiting for.

## Bounded-loops audit (code-safety rule #2)

- `map_map`'s main loop is bounded by `max_steps`.
- `_nearest_frontier` is BFS over the finite `reached` set — terminates.
- Repositioning presses `len(route)` times, `route` ≤ `|reached|` — bounded.
- `probe_step` is bounded by `TURN_RETRIES`; `snapshot_settled` by `SETTLE_TRIES`.
- No recursion.

## Open questions / risks

- **Ledges (one-way tiles).** `WallMap.block` is bidirectional (P2 decision), so
  a one-way ledge would be mislearned as a two-way wall, and BFS-over-`reached`
  could treat a ledge-adjacent edge as walkable both ways. Emerald's early routes
  have few ledges; this is a known limitation, deferred with the tile-behavior
  probe. Not blocking for the first mappable towns/routes.
- **Reversible-return assumption.** Border connections are reversible by stepping
  back; the survey relies on that to keep mapping after finding a door. Building
  warps are not, and correctly end the run as `"left_map"` with the portal saved.
- **`capture_open_map.py` depends on the (gitignored) checkpoint.** The tool is
  run once locally where `ppo_emerald_final.zip` exists; its output
  `states/open_map.state` is committed, so the ROM smoke does not need the
  checkpoint afterwards.
- **`_nearest_frontier` tie-breaking.** BFS returns the nearest frontier; ties
  are broken by `DIRECTIONS` order for determinism (keeps unit tests stable).
