# World Surveyor (P3 step 4) — Design

## Goal

Give the Explorer an orchestrator that maps the **overworld** map by map:
starting from wherever the player currently stands, it repeatedly travels to a
not-yet-surveyed map and surveys it, discovering new maps through the reversible
border portals it finds, until the reachable overworld is fully charted.

This is the fourth P3 brick. Steps 1-3 built the pieces:
- `navigate_to` (step 1): walk to a cell on the current map, learning walls.
- `travel_to` (step 2): chain known portals from map A to map B.
- `map_map` (step 3): actively survey ONE map, recording every wall and portal.

The surveyor loops `travel_to` + `map_map` to extend `MapMemory` across the
world. It does **not** enter buildings, train the Explorer, drive the
Strategist, or fight. Those are later paliers.

## Scope

**In scope:**
- Overworld-only sweep (maps linked by reversible border connections).
- Iterative BFS over the map graph, bounded by `max_maps`.
- Robust log-and-continue: a failed leg is recorded, the sweep goes on.
- A `SurveyReport` result listing surveyed maps and failures with reasons.
- `Portal` gains two fields (`reversible`, `to_cell`) so the surveyor can filter
  overworld portals and resolve the entry cell for `travel_to`.

**Out of scope (deferred):**
- Entering building warps and mapping interiors (non-reversible round-trip).
- Strategist order interface / moving the story out of `PokemonEmeraldEnv`.
- Tile-behavior probe / cell semantics (grass, water, ledges).
- Any reward or training change.

## Architecture & components

### New file `env/world_surveyor.py`

One public function plus two private helpers and a frozen result type.

```python
def survey_world(
    emulator,
    reader,
    memory,
    wallmap,
    max_maps: int = 50,
) -> SurveyReport
```

Dependencies are injected (duck-typed), matching every other P3 module, so the
surveyor is fully testable without a ROM. The starting map is read live from the
emulator (no `start_map` parameter): the sweep begins wherever the player is.

```python
@dataclass(frozen=True)
class SurveyReport:
    surveyed: tuple[tuple[int, int], ...]                    # maps charted, in visit order
    failed: tuple[tuple[tuple[int, int], str], ...]          # (map_id, reason)
```

### Modified `env/map_memory.py`

`Portal` gains two fields:

```python
@dataclass(frozen=True)
class Portal:
    from_cell: tuple[int, int]
    direction: str
    to_map: tuple[int, int]
    reversible: bool
    to_cell: tuple[int, int]
```

- `reversible`: `True` when the crossing is a reversible border connection
  (route/town edge), `False` for a non-reversible building warp. Only the
  surveyor's overworld filter uses it, but it is useful groundwork for the
  interiors palier and the Strategist.
- `to_cell`: the cell the player lands on in `to_map` after crossing. Map
  coordinates do NOT continue across a boundary (crossing Route 101's north edge
  lands in Oldale at unrelated coordinates), so the landing cell must be stored;
  it cannot be derived from `from_cell + DELTAS[direction]`.

`record_portal(from_map, from_cell, direction, to_map, reversible, to_cell)`
gains the two new arguments. Last-write-wins keyed on `(from_map, to_map)`
is unchanged.

### Modified `env/map_explorer.py`

When `map_map` records a portal it now supplies both new fields:
- Reversible border (it performs the step-back and continues): `reversible=True`,
  `to_cell = landed.pos` (observed before stepping back).
- Non-reversible warp (it ends the run as `left_map`): `reversible=False`,
  `to_cell = landed.pos`.

`map_map` already observes `landed.pos` in the transition branch, so no new
observation is needed — only threading the values into `record_portal`.

### Modified `env/live_navigator.py`

`navigate_to` already records portals when it detects a live transition
(P3 step 2). It observes `landed.pos` (after one more snapshot to read
`to_map`), so `to_cell = landed.pos`. A live crossing is not re-tested with a
step-back, so it cannot prove reversibility: it passes `reversible=False`
(the cautious default). This keeps `navigate_to` behavior identical to step 2
apart from the two extra recorded fields.

## Algorithm & data flow

Iterative BFS over the map graph, bounded by `max_maps` (code-safety rule #2):

```python
def survey_world(emulator, reader, memory, wallmap, max_maps=50) -> SurveyReport:
    start = _current_map(reader)
    if start is None:
        return SurveyReport((), (("unknown", "no_start"),))

    pending: deque = deque([start])
    queued: set = {start}
    surveyed: list = []
    failed: list = []

    for _ in range(max_maps):
        if not pending:
            break
        target = pending.popleft()

        # 1. reach the target map (unless already there)
        here = _current_map(reader)
        if here != target:
            outcome = travel_to(
                emulator, reader, memory, wallmap,
                target, _entry_cell(memory, target),
            )
            if outcome != "arrived":
                failed.append((target, f"travel:{outcome}"))
                continue

        # 2. survey it
        result = map_map(emulator, reader, memory, wallmap, target)
        if result in ("left_map", "budget_exhausted"):
            failed.append((target, f"map:{result}"))
        surveyed.append(target)

        # 3. enqueue not-yet-seen overworld neighbours
        for portal in _overworld_portals(memory, target):
            nxt = portal.to_map
            if nxt not in queued and nxt not in surveyed:
                queued.add(nxt)
                pending.append(nxt)

    return SurveyReport(tuple(surveyed), tuple(failed))
```

Two private helpers:

- `_current_map(reader)`: `snapshot_settled(reader)` then return `.map_id`
  (or `None` if the snapshot never settles).
- `_overworld_portals(memory, map_id)`: the outgoing `Portal`s of `map_id` whose
  `reversible` is `True`. This is the overworld-only filter; building warps
  (`reversible=False`) are skipped, so their target maps are never enqueued.
- `_entry_cell(memory, target)`: find any recorded portal whose `to_map ==
  target` and return its `to_cell`. `travel_to` routes to that portal's door,
  crosses, lands on `to_cell`, and its final `navigate_to(to_cell)` returns
  `"arrived"` immediately (already there).

**Chicken-and-egg resolved:** a map is enqueued only once a portal leading to it
is already recorded, so `travel_to` always has a route. The start map is the one
map with no incoming portal, but the player is already on it, so no `travel_to`
is attempted for it.

**Bounded loops (code-safety #2):** the sweep loop is bounded by `max_maps`;
`travel_to`, `map_map`, `snapshot_settled` are each independently bounded. No
recursion (BFS is iterative with an explicit queue).

## Error handling

Log-and-continue keeps a local failure from killing the whole sweep:
- `travel_to` returns anything but `"arrived"` → append `(target,
  f"travel:{outcome}")` and move to the next pending map. The map is not
  surveyed and its neighbours are not enqueued (we could not reach it).
- `map_map` returns `"left_map"` or `"budget_exhausted"` → keep whatever it
  learned (walls/portals already written to `memory`/`wallmap`), append
  `(target, f"map:{result}")`, still count the map as surveyed, and enqueue its
  discovered overworld neighbours.
- No start map (`snapshot_settled` never settles) → immediate
  `SurveyReport((), (("unknown", "no_start"),))`.

The `SurveyReport` is the single source of truth for what was charted and what
was missed. A partially explored world is still useful.

## Testing

### `tests/test_world_surveyor.py` (no ROM)

A unified fake `WorldGrid` plays both emulator and reader for a multi-map world:
hidden per-map grids, per-map walls, and border connections
(reversible/non-reversible). It merges the ideas of `MultiMapWorld`
(map_traveler tests) and `ExploreWorld` (map_explorer tests) into one fake that
supports `travel_to` AND `map_map`. It is test-only and lives in this file
(no cross-test-module imports).

Scenarios:
1. Two maps linked by a reversible border → both in `surveyed`, `failed` empty.
2. Chain of three maps → all three surveyed in BFS order.
3. An overworld map whose far side is sealed (unreachable) → appears in `failed`
   with a `travel:...` reason; the sweep still finishes.
4. A building warp (`reversible=False`) → its target map is never enqueued
   (overworld-only respected).
5. `max_maps` reached mid-sweep → stops cleanly, returns a partial report.

### `tests/test_map_memory.py` (extended)

Portal construction and `record_portal` calls updated for the two new fields;
add assertions that `reversible` and `to_cell` round-trip through
`record_portal`/`portal`.

### `tests/test_world_surveyor_rom.py` (1, ROM-gated)

Smoke on `states/open_map.state`: `survey_world(max_maps=2)`, assert the report
is coherent (surveyed non-empty OR failed explains why) and that learning
happened, observed through externally-visible state only (a portal recorded or a
wall learned). Double skipif: `POKEMON_EMERALD_ROM` unset or the state file
missing.

### Call-site updates

`Portal(...)` / `record_portal(...)` calls in `tests/test_map_explorer.py`,
`tests/test_live_navigator.py`, and `tests/test_map_traveler.py` updated for the
new fields so the full suite stays green.

## Files touched

- Create: `env/world_surveyor.py`
- Create: `tests/test_world_surveyor.py`
- Create: `tests/test_world_surveyor_rom.py`
- Modify: `env/map_memory.py` (Portal +2 fields, `record_portal` signature)
- Modify: `env/map_explorer.py` (thread `reversible` + `to_cell` into record)
- Modify: `env/live_navigator.py` (thread `reversible=False` + `to_cell`)
- Modify: `tests/test_map_memory.py`, `tests/test_map_explorer.py`,
  `tests/test_live_navigator.py`, `tests/test_map_traveler.py` (call-site updates)
