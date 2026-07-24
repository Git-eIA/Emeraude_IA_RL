# Explorer Navigator (P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Explorer a navigation brick — inter-map routing over the
`MapMemory` graph plus intra-map A* movement using observed-collision walls, all
testable in isolation with no emulator, no ROM, no probe.

**Architecture:** Two pure modules. `env/route_planner.py` does BFS over the P1
`MapMemory` graph (Level A). `env/local_navigator.py` holds the collision
detector, a self-built `WallMap`, and A* over that learned grid (Level B). Both
consume plain data; the live A↔B wiring is deferred to P3.

**Tech Stack:** Python ≥3.12, pytest, ruff (line-length 100). Depends only on
the existing `env/map_memory.py` (`MapMemory`, `edges()`, `nodes`) and
`env/world_reader.py` (`WorldSnapshot` with `map_id`, `pos`, `tile_behavior`).

---

## File structure

- `env/route_planner.py` (new) — `plan_route(memory, start_map, goal_map)`; BFS
  over `MapMemory.edges()`. Single responsibility: inter-map shortest path.
- `env/local_navigator.py` (new) — direction constants, `resolve_move`,
  `WallMap`, `plan_path` (A*). Single responsibility: intra-map movement.
- `tests/test_route_planner.py` (new).
- `tests/test_local_navigator.py` (new).

Two modules because routing between maps and moving within a map are distinct
concerns with different inputs (a graph vs a grid of walls).

---

## Task 1: Inter-map routing (`plan_route`)

**Files:**
- Create: `env/route_planner.py`
- Test: `tests/test_route_planner.py`

- [ ] **Step 1: Write the failing tests**

```python
"""plan_route: BFS shortest path over the MapMemory graph (no ROM, no emulator)."""
from __future__ import annotations

from env.map_memory import MapMemory, WorldEvent
from env.route_planner import plan_route
from env.world_reader import WorldSnapshot


def _snap(map_id: tuple[int, int]) -> WorldSnapshot:
    return WorldSnapshot(map_id=map_id, pos=(0, 0), tile_behavior=None)


def _walk(memory: MapMemory, *map_ids: tuple[int, int]) -> None:
    """Feed a sequence of maps so MapMemory records the walked edges."""
    for map_id in map_ids:
        memory.observe(_snap(map_id), WorldEvent())


def test_start_equals_goal_returns_single_map() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9))
    assert plan_route(memory, (0, 9), (0, 9)) == [(0, 9)]


def test_direct_edge() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16))  # (0,9) -> (0,16)
    assert plan_route(memory, (0, 9), (0, 16)) == [(0, 9), (0, 16)]


def test_multi_hop_shortest_path() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16), (0, 17), (0, 18))
    assert plan_route(memory, (0, 9), (0, 18)) == [(0, 9), (0, 16), (0, 17), (0, 18)]


def test_none_when_goal_unknown() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16))
    assert plan_route(memory, (0, 9), (5, 5)) is None


def test_none_when_disconnected() -> None:
    memory = MapMemory()
    _walk(memory, (0, 9), (0, 16))   # component A
    memory._prev_map_id = None       # break the chain so no edge (0,16)->(1,1)
    _walk(memory, (1, 1), (1, 2))    # component B, unreachable from (0,9)
    assert plan_route(memory, (0, 9), (1, 2)) is None


def test_bfs_prefers_fewer_hops() -> None:
    memory = MapMemory()
    # long way: A -> B -> C -> D
    _walk(memory, (0, 0), (0, 1), (0, 2), (0, 3))
    # shortcut: A -> D
    memory._prev_map_id = (0, 0)
    _walk(memory, (0, 3))
    assert plan_route(memory, (0, 0), (0, 3)) == [(0, 0), (0, 3)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_route_planner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.route_planner'`.

- [ ] **Step 3: Write the implementation**

```python
"""plan_route: shortest inter-map path over the MapMemory graph.

BFS over the directed edges MapMemory recorded from walked transitions. Edges are
unweighted (one map hop = one step), so BFS yields the fewest-transitions route.
Pure — no emulator, no ROM. Emerald (BPEF) only.
"""
from __future__ import annotations

from collections import deque

from env.map_memory import MapMemory


def plan_route(
    memory: MapMemory,
    start_map: tuple[int, int],
    goal_map: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Return the map_ids from start to goal inclusive, or None if unreachable."""
    if start_map == goal_map:
        return [start_map]

    # Adjacency from the recorded directed edges.
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for src, dst in memory.edges():
        adjacency.setdefault(src, []).append(dst)

    # BFS carrying the path to each visited map.
    queue: deque[list[tuple[int, int]]] = deque([[start_map]])
    seen: set[tuple[int, int]] = {start_map}
    while queue:
        path = queue.popleft()
        current = path[-1]
        for neighbour in adjacency.get(current, ()):
            if neighbour == goal_map:
                return path + [neighbour]
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(path + [neighbour])
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_route_planner.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/route_planner.py tests/test_route_planner.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add env/route_planner.py tests/test_route_planner.py
git commit -m "feat: P2 inter-map routing (BFS over MapMemory graph)"
```

---

## Task 2: Direction constants + collision detector (`resolve_move`)

**Files:**
- Create: `env/local_navigator.py`
- Test: `tests/test_local_navigator.py`

- [ ] **Step 1: Write the failing tests**

```python
"""local_navigator: observed-collision detector, WallMap, A* (no ROM, no emulator)."""
from __future__ import annotations

from env.local_navigator import (
    DIRECTIONS,
    resolve_move,
)
from env.world_reader import WorldSnapshot


def _snap(map_id: tuple[int, int], pos: tuple[int, int]) -> WorldSnapshot:
    return WorldSnapshot(map_id=map_id, pos=pos, tile_behavior=None)


def test_directions_are_the_four_cardinals() -> None:
    assert set(DIRECTIONS) == {"up", "down", "left", "right"}


def test_moved_when_pos_changes_same_map() -> None:
    before = _snap((0, 9), (2, 3))
    after = _snap((0, 9), (2, 2))
    assert resolve_move(before, after) == "moved"


def test_blocked_when_pos_unchanged_same_map() -> None:
    before = _snap((0, 9), (2, 3))
    after = _snap((0, 9), (2, 3))
    assert resolve_move(before, after) == "blocked"


def test_transition_when_map_changes() -> None:
    before = _snap((0, 9), (2, 3))
    after = _snap((0, 16), (5, 10))
    assert resolve_move(before, after) == "transition"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_local_navigator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'env.local_navigator'`.

- [ ] **Step 3: Write the implementation**

```python
"""local_navigator: intra-map movement from observed collisions.

The Explorer learns which cell-to-cell moves are blocked by bumping into walls
(pressed a direction, position did not change -> wall). Those observations fill a
WallMap, and A* plans a path over the known-walkable grid, replanning whenever a
new wall is discovered. No tile-behavior probe, no emulator, no ROM. Emerald
(BPEF) only.
"""
from __future__ import annotations

from env.world_reader import WorldSnapshot

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")

# Grid convention: x grows right, y grows down. up decreases y.
DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def resolve_move(before: WorldSnapshot, after: WorldSnapshot) -> str:
    """Classify one attempted step: 'moved' | 'blocked' | 'transition'."""
    if before.map_id != after.map_id:
        return "transition"
    if before.pos != after.pos:
        return "moved"
    return "blocked"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_local_navigator.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/local_navigator.py tests/test_local_navigator.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add env/local_navigator.py tests/test_local_navigator.py
git commit -m "feat: P2 collision detector + direction constants"
```

---

## Task 3: Learned wall grid (`WallMap`)

**Files:**
- Modify: `env/local_navigator.py`
- Test: `tests/test_local_navigator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_local_navigator.py`:

```python
from env.local_navigator import WallMap


def test_wallmap_records_blocked_edge() -> None:
    walls = WallMap()
    walls.block((0, 9), (2, 3), "up")
    assert walls.is_blocked((0, 9), (2, 3), "up") is True


def test_wallmap_unknown_edge_is_optimistically_open() -> None:
    walls = WallMap()
    assert walls.is_blocked((0, 9), (2, 3), "up") is False


def test_wallmap_blocking_is_bidirectional() -> None:
    walls = WallMap()
    walls.block((0, 9), (2, 3), "up")   # wall between (2,3) and (2,2)
    # the neighbour going back the opposite way is also blocked
    assert walls.is_blocked((0, 9), (2, 2), "down") is True


def test_wallmap_is_per_map() -> None:
    walls = WallMap()
    walls.block((0, 9), (2, 3), "up")
    assert walls.is_blocked((0, 16), (2, 3), "up") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_local_navigator.py -q`
Expected: FAIL — `ImportError: cannot import name 'WallMap'`.

- [ ] **Step 3: Write the implementation**

Add to `env/local_navigator.py` (after the module constants):

```python
class WallMap:
    """Per-map set of blocked directed edges, learned from observed collisions.

    An edge never observed is assumed walkable (optimistic); A* replans when a
    real wall is later discovered. Blocking is bidirectional: a wall between two
    cells blocks the move both ways.
    """

    def __init__(self) -> None:
        # map_id -> set of (cell, direction) that are known blocked.
        self._blocked: dict[
            tuple[int, int], set[tuple[tuple[int, int], str]]
        ] = {}

    def block(
        self, map_id: tuple[int, int], cell: tuple[int, int], direction: str
    ) -> None:
        """Record a wall in `direction` from `cell`, and its mirror from the neighbour."""
        edges = self._blocked.setdefault(map_id, set())
        edges.add((cell, direction))
        dx, dy = DELTAS[direction]
        neighbour = (cell[0] + dx, cell[1] + dy)
        edges.add((neighbour, OPPOSITE[direction]))

    def is_blocked(
        self, map_id: tuple[int, int], cell: tuple[int, int], direction: str
    ) -> bool:
        """True only if this edge was observed blocked; unknown edges are open."""
        return (cell, direction) in self._blocked.get(map_id, ())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_local_navigator.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Lint**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/local_navigator.py tests/test_local_navigator.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add env/local_navigator.py tests/test_local_navigator.py
git commit -m "feat: P2 WallMap — bidirectional learned wall grid"
```

---

## Task 4: A* over the learned grid (`plan_path`)

**Files:**
- Modify: `env/local_navigator.py`
- Test: `tests/test_local_navigator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_local_navigator.py`:

```python
from env.local_navigator import plan_path


def test_path_start_equals_goal_is_empty() -> None:
    walls = WallMap()
    assert plan_path(walls, (0, 9), (2, 3), (2, 3)) == []


def test_path_straight_line_no_walls() -> None:
    walls = WallMap()
    # (0,0) -> (2,0): two steps right
    assert plan_path(walls, (0, 9), (0, 0), (2, 0)) == ["right", "right"]


def test_path_detours_around_a_wall() -> None:
    walls = WallMap()
    # Block the direct step right from (0,0); A* must go around via down.
    walls.block((0, 9), (0, 0), "right")
    path = plan_path(walls, (0, 9), (0, 0), (1, 0))
    assert path is not None
    # any valid detour reaches the goal; verify by walking it
    x, y = 0, 0
    for d in path:
        assert walls.is_blocked((0, 9), (x, y), d) is False
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
        x, y = x + dx, y + dy
    assert (x, y) == (1, 0)


def test_path_none_when_fully_walled_in() -> None:
    walls = WallMap()
    for d in ("up", "down", "left", "right"):
        walls.block((0, 9), (0, 0), d)
    assert plan_path(walls, (0, 9), (0, 0), (5, 5)) is None


def test_path_replans_after_new_wall_discovered() -> None:
    walls = WallMap()
    # first plan goes straight right
    first = plan_path(walls, (0, 9), (0, 0), (2, 0))
    assert first == ["right", "right"]
    # discover a wall on the direct route, replan
    walls.block((0, 9), (1, 0), "right")
    second = plan_path(walls, (0, 9), (0, 0), (2, 0))
    assert second is not None
    assert second != first  # forced to detour
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_local_navigator.py -q`
Expected: FAIL — `ImportError: cannot import name 'plan_path'`.

- [ ] **Step 3: Write the implementation**

Add to `env/local_navigator.py`:

```python
import heapq


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_path(
    wallmap: WallMap,
    map_id: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[str] | None:
    """A* over the learned grid; list of directions from start to goal, or None.

    Unknown edges are treated as walkable (optimistic). The infinite grid is
    explored lazily; the Manhattan heuristic keeps A* focused toward the goal.
    """
    if start == goal:
        return []

    # Priority queue of (f_score, cell); came_from records (prev_cell, direction).
    open_heap: list[tuple[int, tuple[int, int]]] = [(_manhattan(start, goal), start)]
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        for direction in DIRECTIONS:
            if wallmap.is_blocked(map_id, current, direction):
                continue
            dx, dy = DELTAS[direction]
            neighbour = (current[0] + dx, current[1] + dy)
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbour, 1 << 30):
                g_score[neighbour] = tentative
                came_from[neighbour] = (current, direction)
                f_score = tentative + _manhattan(neighbour, goal)
                heapq.heappush(open_heap, (f_score, neighbour))
    return None


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[tuple[int, int], str]],
    goal: tuple[int, int],
) -> list[str]:
    directions: list[str] = []
    cell = goal
    while cell in came_from:
        prev, direction = came_from[cell]
        directions.append(direction)
        cell = prev
    directions.reverse()
    return directions
```

Move the `import heapq` to the top of the file with the other imports (keep
`from __future__ import annotations` first).

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_local_navigator.py -q`
Expected: PASS (13 passed).

- [ ] **Step 5: Full suite + lint**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=$PWD /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: all prior tests still pass (P1 + P2 green).

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/local_navigator.py tests/test_local_navigator.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add env/local_navigator.py tests/test_local_navigator.py
git commit -m "feat: P2 A* pathfinding over the learned wall grid"
```

---

## Self-review notes

- Spec coverage: Task 1 = Level A (`plan_route`). Tasks 2-4 = Level B
  (`resolve_move`, `WallMap`, `plan_path`). All spec components covered.
- The `test_none_when_disconnected` / `test_bfs_prefers_fewer_hops` tests reset
  `memory._prev_map_id` to control which edges get recorded — the P1 `observe`
  API only creates an edge between consecutive `observe` calls on different maps.
- Types are consistent across tasks: `map_id`/`pos`/`cell` are all
  `tuple[int, int]`; directions are `str` from `DIRECTIONS`; `plan_route`
  returns `list[tuple[int, int]] | None`; `plan_path` returns `list[str] | None`.
- A* explores an unbounded grid; every test has a reachable goal or is fully
  walled at the start cell, so the search terminates. The live loop (P3) will
  bound the goal to a real in-map cell.
