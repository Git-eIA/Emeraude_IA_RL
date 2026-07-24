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
