# World Perception + Map Memory (P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Explorer a RAM-based perception snapshot (`WorldReader`) and a self-built observational map graph (`MapMemory`) of the places it discovers.

**Architecture:** Two isolated modules under `env/`. `WorldReader` wraps the existing `EmeraldReader` and returns an immutable `WorldSnapshot` (map id, position, tile behavior). `MapMemory` consumes a stream of `(snapshot, event)` and builds a graph: nodes = visited maps (typed via a small catalog), edges = observed transitions, labels = facts from the vécu (healing_spot, has_grass). No pixels, no navigation, no training, no reward changes.

**Tech Stack:** Python ≥3.12, dataclasses (`frozen=True` where immutable), pytest, ruff (line-length 100). Tests use `FakeEmulator` from `tests/conftest.py` — no ROM.

**Reference — existing API (do not change):**
- `env/game_state.py`: `EmeraldReader(read: ReadFn)` with `player_state() -> PlayerState | None`. `PlayerState` fields: `x, y, map_group, map_num, badges, party_count, clock_set, town_state`. `player_state()` returns `None` while save blocks relocate.
- `tests/conftest.py`: `FakeEmulator` exposes `read_bytes(address, length) -> bytes` and attributes `x, y, map_group, map_num`. Build a reader with `EmeraldReader(fake.read_bytes)`.

---

### Task 1: WorldReader + WorldSnapshot

**Files:**
- Create: `env/world_reader.py`
- Test: `tests/test_world_reader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_world_reader.py
"""WorldReader: RAM snapshot of the Explorer's world state (no navigation)."""
from __future__ import annotations

from env.game_state import EmeraldReader
from env.world_reader import WorldReader, WorldSnapshot
from tests.conftest import FakeEmulator


def _reader(emu: FakeEmulator) -> WorldReader:
    return WorldReader(EmeraldReader(emu.read_bytes))


def test_snapshot_reads_map_id_and_position() -> None:
    emu = FakeEmulator()
    emu.map_group, emu.map_num = 0, 16  # Route 101
    emu.x, emu.y = 3, 7
    snap = _reader(emu).snapshot()
    assert isinstance(snap, WorldSnapshot)
    assert snap.map_id == (0, 16)
    assert snap.pos == (3, 7)


def test_tile_behavior_is_none_until_probed() -> None:
    snap = _reader(FakeEmulator()).snapshot()
    assert snap is not None
    assert snap.tile_behavior is None


def test_snapshot_is_none_while_save_blocks_relocate() -> None:
    emu = FakeEmulator()
    emu._sb1 = 0x00000000  # out of EWRAM range -> player_state() returns None
    assert _reader(emu).snapshot() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_world_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.world_reader'`

- [ ] **Step 3: Write minimal implementation**

```python
# env/world_reader.py
"""WorldReader: RAM-based perception snapshot for the Explorer.

Reads only where the player is (map id, position) and what tile it stands on.
No pixels. Isolated on purpose so the perception layer is a single swappable
unit. Emerald (BPEF) only — game-specific RAM is used freely.
"""
from __future__ import annotations

from dataclasses import dataclass

from env.game_state import EmeraldReader


@dataclass(frozen=True)
class WorldSnapshot:
    map_id: tuple[int, int]        # (map_group, map_num) — which map
    pos: tuple[int, int]           # (x, y) — position within the map
    tile_behavior: int | None      # current-tile behavior id; None until probed


class WorldReader:
    """Wraps EmeraldReader and returns an immutable WorldSnapshot each step."""

    def __init__(self, reader: EmeraldReader) -> None:
        self._reader = reader

    def snapshot(self) -> WorldSnapshot | None:
        """Snapshot the world, or None while the save blocks relocate."""
        ps = self._reader.player_state()
        if ps is None:
            return None
        return WorldSnapshot(
            map_id=(ps.map_group, ps.map_num),
            pos=(ps.x, ps.y),
            tile_behavior=self._tile_behavior(),
        )

    def _tile_behavior(self) -> int | None:
        # TODO(probe): read the metatile-behavior byte of the tile the player
        # stands on (tall grass, water, wall, door, ...). Its RAM address on
        # BPEF is not yet known; returns None until a probe session finds it.
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_world_reader.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check env/world_reader.py tests/test_world_reader.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add env/world_reader.py tests/test_world_reader.py
git commit -m "feat(world): WorldReader RAM snapshot (map id, pos, tile TODO-probe)"
```

---

### Task 2: MapMemory — nodes, catalog, node creation

**Files:**
- Create: `env/map_memory.py`
- Test: `tests/test_map_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_map_memory.py
"""MapMemory: self-built observational graph of discovered places."""
from __future__ import annotations

from env.map_memory import KNOWN_PLACES, MapMemory, PlaceNode, WorldEvent
from env.world_reader import WorldSnapshot


def _snap(map_id: tuple[int, int], pos: tuple[int, int] = (0, 0)) -> WorldSnapshot:
    return WorldSnapshot(map_id=map_id, pos=pos, tile_behavior=None)


def test_first_sight_creates_a_node() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent())
    node = mem.node((0, 16))
    assert isinstance(node, PlaceNode)
    assert node.map_id == (0, 16)
    assert node.labels == set()


def test_unknown_map_defaults_to_unknown_place_type() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent())
    assert mem.node((0, 16)).place_type == "unknown"


def test_catalogued_map_resolves_its_place_type() -> None:
    # Pick any id present in the catalog; skip cleanly if the catalog is empty.
    if not KNOWN_PLACES:
        return
    known_id, known_type = next(iter(KNOWN_PLACES.items()))
    mem = MapMemory()
    mem.observe(_snap(known_id), WorldEvent())
    assert mem.node(known_id).place_type == known_type


def test_revisiting_a_map_does_not_duplicate_the_node() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16), (1, 1)), WorldEvent())
    mem.observe(_snap((0, 16), (2, 2)), WorldEvent())
    assert mem.node((0, 16)) is not None
    assert len(mem.nodes) == 1


def test_node_returns_none_for_unseen_map() -> None:
    assert MapMemory().node((9, 9)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_map_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.map_memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# env/map_memory.py
"""MapMemory: a self-built graph of the places the Explorer discovers.

Built by observation only — never by parsing the ROM. Nodes are maps the player
has stood on; edges are transitions actually walked; labels are facts learned
from experience (a heal happened here, a wild battle started here). Emerald
(BPEF) only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from env.world_reader import WorldSnapshot

# Small, game-specific catalog: known map ids -> a priori place type. The one
# place that would change for another game. Grows as we identify more maps.
KNOWN_PLACES: dict[tuple[int, int], str] = {
    # (map_group, map_num): place_type
    # e.g. a Pokémon Center map id -> "pokemon_center"
}


@dataclass
class PlaceNode:
    map_id: tuple[int, int]
    place_type: str
    labels: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class WorldEvent:
    healed: bool = False              # HP restored to full this step
    encounter_started: bool = False   # a wild battle began this step


class MapMemory:
    """Observational graph: nodes = visited maps, edges = walked transitions."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[int, int], PlaceNode] = {}

    def observe(self, snapshot: WorldSnapshot, event: WorldEvent) -> None:
        self._ensure_node(snapshot.map_id)

    def node(self, map_id: tuple[int, int]) -> PlaceNode | None:
        return self.nodes.get(map_id)

    def _ensure_node(self, map_id: tuple[int, int]) -> PlaceNode:
        node = self.nodes.get(map_id)
        if node is None:
            node = PlaceNode(map_id=map_id, place_type=KNOWN_PLACES.get(map_id, "unknown"))
            self.nodes[map_id] = node
        return node
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_map_memory.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add env/map_memory.py tests/test_map_memory.py
git commit -m "feat(world): MapMemory nodes + place-type catalog"
```

---

### Task 3: MapMemory — edges on observed transitions

**Files:**
- Modify: `env/map_memory.py`
- Test: `tests/test_map_memory.py` (add tests)

- [ ] **Step 1: Write the failing test (append to tests/test_map_memory.py)**

```python
def test_transition_between_maps_adds_a_directed_edge() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9)), WorldEvent())    # Littleroot
    mem.observe(_snap((0, 16)), WorldEvent())   # step onto Route 101
    assert ((0, 9), (0, 16)) in mem.edges()


def test_staying_on_the_same_map_adds_no_edge() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9), (1, 1)), WorldEvent())
    mem.observe(_snap((0, 9), (1, 2)), WorldEvent())
    assert mem.edges() == set()


def test_edges_are_directional() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9)), WorldEvent())
    mem.observe(_snap((0, 16)), WorldEvent())
    assert ((0, 16), (0, 9)) not in mem.edges()


def test_first_observation_adds_no_edge() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 9)), WorldEvent())
    assert mem.edges() == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_map_memory.py -v`
Expected: FAIL with `AttributeError: 'MapMemory' object has no attribute 'edges'`

- [ ] **Step 3: Update implementation**

In `env/map_memory.py`, change `__init__`, `observe`, and add `edges()`:

```python
    def __init__(self) -> None:
        self.nodes: dict[tuple[int, int], PlaceNode] = {}
        self._edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self._prev_map_id: tuple[int, int] | None = None

    def observe(self, snapshot: WorldSnapshot, event: WorldEvent) -> None:
        self._ensure_node(snapshot.map_id)
        if self._prev_map_id is not None and self._prev_map_id != snapshot.map_id:
            self._edges.add((self._prev_map_id, snapshot.map_id))
        self._prev_map_id = snapshot.map_id

    def edges(self) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        return set(self._edges)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_map_memory.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add env/map_memory.py tests/test_map_memory.py
git commit -m "feat(world): MapMemory directed edges from observed transitions"
```

---

### Task 4: MapMemory — event labels (healing_spot, has_grass)

**Files:**
- Modify: `env/map_memory.py`
- Test: `tests/test_map_memory.py` (add tests)

- [ ] **Step 1: Write the failing test (append to tests/test_map_memory.py)**

```python
def test_heal_event_labels_the_current_place_as_healing_spot() -> None:
    mem = MapMemory()
    mem.observe(_snap((1, 5)), WorldEvent(healed=True))
    assert "healing_spot" in mem.node((1, 5)).labels


def test_encounter_event_labels_the_current_place_as_has_grass() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent(encounter_started=True))
    assert "has_grass" in mem.node((0, 16)).labels


def test_labels_are_additive_across_observations() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent(encounter_started=True))
    mem.observe(_snap((0, 16)), WorldEvent(healed=True))
    assert mem.node((0, 16)).labels == {"has_grass", "healing_spot"}


def test_no_event_adds_no_label() -> None:
    mem = MapMemory()
    mem.observe(_snap((0, 16)), WorldEvent())
    assert mem.node((0, 16)).labels == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_map_memory.py -v`
Expected: FAIL — `healing_spot`/`has_grass` not in labels

- [ ] **Step 3: Update implementation**

In `env/map_memory.py`, extend `observe` to apply event labels:

```python
    def observe(self, snapshot: WorldSnapshot, event: WorldEvent) -> None:
        node = self._ensure_node(snapshot.map_id)
        if self._prev_map_id is not None and self._prev_map_id != snapshot.map_id:
            self._edges.add((self._prev_map_id, snapshot.map_id))
        self._prev_map_id = snapshot.map_id
        if event.healed:
            node.labels.add("healing_spot")
        if event.encounter_started:
            node.labels.add("has_grass")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_map_memory.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint + full suite**

Run: `.venv/bin/ruff check env/ tests/`
Expected: no errors

Run: `POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba .venv/bin/pytest -q`
Expected: all tests pass (100 prior + 16 new = 116)

- [ ] **Step 6: Commit**

```bash
git add env/map_memory.py tests/test_map_memory.py
git commit -m "feat(world): MapMemory observed labels (healing_spot, has_grass)"
```
