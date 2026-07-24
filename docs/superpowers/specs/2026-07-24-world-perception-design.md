# World Perception + Map Memory (P1) — design

**Date:** 2026-07-24
**Status:** Approved (brainstorming), pending implementation plan
**Milestone:** P1 — the Explorer's world perception + self-built map memory

## Goal

Give the Explorer a perception layer that reads *where it is* and *what is here*
from RAM, and a memory that records the places it discovers as it walks — a
self-built graph of the world. No pixels for this: perception is RAM-based.

## Context

The project runs three decoupled brains (Explorer / Fighter / Strategist). We
are moving toward a hierarchical (manager-worker) architecture where:

- **Strategist** = manager: holds the plan, issues orders (destination + mode).
- **Explorer** = navigation worker + world perception: perceives the world,
  recognizes places, navigates to the ordered destination.
- **Fighter** = battle worker.

Today the Explorer conflates navigation with *story knowledge* (the milestone
chain in `env/milestones.py` bakes "where to go" into its reward). The long-term
plan moves the story to the Strategist. Before the Explorer can *navigate to an
ordered place*, it needs two things it does not have yet: a clean **perception**
of its location and surroundings, and a **memory** of the places it has found.

P1 builds exactly those two pieces — nothing more. No navigation, no pathfinding,
no training, no reward changes. This is the perception substrate the later
paliers (P2 navigator, P3 hierarchical loop) build on.

## Scope (P1)

Two isolated modules:

1. **`WorldReader`** — a RAM snapshot of the current world state, each step.
2. **`MapMemory`** — an observational graph of discovered places, updated from a
   stream of snapshots + events.

Kept isolated on purpose (like the emulator wrapper) so the perception layer is
a single swappable unit. For now the target is **Pokémon Émeraude (BPEF) only** —
no cross-game portability constraint. We use game-specific RAM freely.

## Non-goals (deferred)

- Navigation / pathfinding to a destination (P2).
- The hierarchical live loop Strategist→Explorer→Fighter (P3).
- Any change to the Explorer's reward or training.
- Full tile radar ("what tile behavior is in every direction"). P1 reads the
  *current* tile behavior only, and that field ships as `TODO(probe)` (see
  Calibration) until a probe session finds the address.
- Cross-game portability (FireRed and other GBA games) — explicitly dropped for
  now; we optimize for Emerald.

## Architecture

### `WorldReader` — the perception snapshot

`WorldReader` wraps the existing `EmeraldReader` (which already reads map id and
coordinates from validated RAM addresses) and returns an immutable snapshot each
step. It reads only RAM — no pixels.

```python
@dataclass(frozen=True)
class WorldSnapshot:
    map_id: tuple[int, int]        # (map_group, map_num) — which map
    pos: tuple[int, int]           # (x, y) — position within the map
    tile_behavior: int | None      # current-tile behavior id; None until probed
```

```python
class WorldReader:
    def __init__(self, reader: EmeraldReader) -> None: ...
    def snapshot(self) -> WorldSnapshot: ...
```

- `map_id` and `pos` come straight from `EmeraldReader` (already validated:
  SaveBlock1-based addresses, signed s16 coords). Reliable, zero new probing.
- `tile_behavior` is the metatile-behavior byte of the tile the player stands on
  (tall grass, water, wall, door, ...). Its RAM address on BPEF is not yet
  known, so `WorldReader` returns `None` for it until the probe lands (the same
  `TODO(probe)` pattern used for the battle addresses). Everything else in P1
  works and is testable without it.

### `MapMemory` — the self-built graph

`MapMemory` consumes a stream of `(snapshot, events)` and builds a graph of the
world by observation only — never by parsing the ROM.

- **Nodes** = maps visited, keyed by `map_id`. Each node carries:
  - `place_type`: `"pokemon_center" | "outdoor" | "indoor" | "unknown"`,
    resolved from a small known-id catalog (below), defaulting to `"unknown"`.
  - `labels: set[str]` = facts accumulated from the vécu, e.g. `"healing_spot"`
    (HP was restored to full here), `"has_grass"` (a wild encounter started
    here). Labels are additive and monotonic within a run.
- **Edges** = observed transitions between maps. When the current snapshot's
  `map_id` differs from the previous one, `MapMemory` records a directed edge
  `previous_map_id -> current_map_id` (a walked passage: door, route link, ...).

```python
@dataclass
class PlaceNode:
    map_id: tuple[int, int]
    place_type: str
    labels: set[str]

@dataclass(frozen=True)
class WorldEvent:
    healed: bool = False              # HP restored to full this step
    encounter_started: bool = False   # a wild battle began this step

class MapMemory:
    def observe(self, snapshot: WorldSnapshot, event: WorldEvent) -> None: ...
    def node(self, map_id: tuple[int, int]) -> PlaceNode | None: ...
    def edges(self) -> set[tuple[tuple[int, int], tuple[int, int]]]: ...
```

`observe` is the single entry point:

1. Ensure a node exists for `snapshot.map_id` (create with catalog-resolved
   `place_type` on first sight).
2. If a previous `map_id` is tracked and differs, add the directed edge.
3. Apply event labels: `healed` → add `"healing_spot"` to the node;
   `encounter_started` → add `"has_grass"` to the node.
4. Update the tracked previous `map_id` to the current one.

`WorldEvent` is produced by the caller (later, the Explorer env, which knows HP
and battle state). In P1 we only define the interface and drive it with scripted
events in tests.

### Place identification — hybrid-lite

A tiny catalog maps known map ids to a place type:

```python
KNOWN_PLACES: dict[tuple[int, int], str] = {
    # (map_group, map_num): place_type
    # e.g. Oldale Pokémon Center -> "pokemon_center"
}
```

Catalog resolves the *a priori* type; observed labels confirm/enrich it (a node
in the catalog as `"pokemon_center"` also gets `"healing_spot"` the first time a
heal is observed there). The catalog is small and BPEF-specific by design — the
one place that would change for another game. It starts with the handful of ids
we already know and grows as we identify more.

## Components / files

- `env/world_reader.py` — `WorldSnapshot` + `WorldReader` (wraps
  `EmeraldReader`; `tile_behavior` is `TODO(probe)`, returns `None` for now).
- `env/map_memory.py` — `PlaceNode`, `WorldEvent`, `MapMemory`, and the
  `KNOWN_PLACES` catalog.
- `tests/test_world_reader.py` — snapshot reads map id + coords from a fake
  reader; `tile_behavior` is `None`.
- `tests/test_map_memory.py` — graph construction: node creation, edge on
  transition, label application (heal, encounter), catalog resolution, idempotent
  re-visits.

All tests use `FakeEmulator` / a fake reader (scripted RAM snapshots). No ROM, no
training, deterministic.

## Calibration

`tile_behavior` ships as `TODO(probe)`: `WorldReader` returns `None` until a
manual probe session on the BPEF ROM finds the metatile-behavior RAM address and
the behavior-id constants (tall grass, water, door, ...). This mirrors how the
battle addresses were validated. When the probe lands, only `WorldReader`
changes; `MapMemory` and its interface do not.

## Testing strategy

Pure-Python, no ROM, deterministic. `WorldReader` is tested against a fake
`EmeraldReader` returning scripted map/coords. `MapMemory` is tested by feeding a
scripted sequence of `(snapshot, event)` and asserting the resulting nodes,
edges, and labels. The `tile_behavior` probe is validated later against the real
ROM, out of band.

## Future (out of scope, noted)

- P2: Explorer navigator — pathfinding across the `MapMemory` graph to reach an
  ordered destination.
- P3: hierarchical live loop — Strategist issues (destination, mode); Explorer
  navigates; Fighter battles on encounter.
- Tile radar: read behavior in all four facing directions (needs the probed
  address) so the navigator can avoid/seek grass, water, ledges.
