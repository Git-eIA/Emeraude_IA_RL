"""MapMemory: a self-built graph of the places the Explorer discovers.

Built by observation only — never by parsing the ROM. Nodes are maps the player
has stood on; edges are transitions actually walked; labels are facts learned
from experience (a heal happened here, a wild battle started here). Emerald
(BPEF) only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from env.world_reader import WorldSnapshot

if TYPE_CHECKING:
    from env.grid_snapshot import GridSnapshot

# Small, game-specific catalog: known map ids -> a priori place type. The one
# place that would change for another game. Grows as we identify more maps.
KNOWN_PLACES: dict[tuple[int, int], str] = {
    # (map_group, map_num): place_type
    # e.g. a Pokémon Center map id -> "pokemon_center"
}


@dataclass(frozen=True)
class Portal:
    """One directed border crossing: leave `from_cell` going `direction` to reach `to_map`."""
    from_cell: tuple[int, int]
    direction: str
    to_map: tuple[int, int]
    reversible: bool          # True = reversible overworld border, False = building warp
    to_cell: tuple[int, int]  # cell landed on in to_map (coords do NOT continue across a border)


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
        self._edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self._prev_map_id: tuple[int, int] | None = None
        # (from_map, to_map) -> the remembered crossing between them.
        self._portals: dict[
            tuple[tuple[int, int], tuple[int, int]], Portal
        ] = {}
        # recognition label -> {map_id: cell} (last-write-wins per map).
        self._labeled_cells: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
        # map_id -> remembered classified grid (last-write-wins).
        self._grids: dict[tuple[int, int], "GridSnapshot"] = {}

    def observe(self, snapshot: WorldSnapshot, event: WorldEvent) -> None:
        node = self._ensure_node(snapshot.map_id)
        if self._prev_map_id is not None and self._prev_map_id != snapshot.map_id:
            self._edges.add((self._prev_map_id, snapshot.map_id))
        self._prev_map_id = snapshot.map_id
        if event.healed:
            node.labels.add("healing_spot")
            self._labeled_cells.setdefault("healing_spot", {})[snapshot.map_id] = snapshot.pos
        if event.encounter_started:
            node.labels.add("has_grass")
            self._labeled_cells.setdefault("has_grass", {})[snapshot.map_id] = snapshot.pos

    def edges(self) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        return set(self._edges)

    def record_portal(
        self,
        from_map: tuple[int, int],
        from_cell: tuple[int, int],
        direction: str,
        to_map: tuple[int, int],
        reversible: bool,
        to_cell: tuple[int, int],
    ) -> None:
        """Remember the door from from_map to to_map; also ensures the edge exists."""
        self._ensure_node(from_map)
        self._ensure_node(to_map)
        self._edges.add((from_map, to_map))
        self._portals[(from_map, to_map)] = Portal(
            from_cell, direction, to_map, reversible, to_cell
        )

    def portal(
        self, from_map: tuple[int, int], to_map: tuple[int, int]
    ) -> Portal | None:
        """The known crossing from from_map to to_map, or None if never recorded."""
        return self._portals.get((from_map, to_map))

    def outgoing_portals(self, from_map: tuple[int, int]) -> tuple[Portal, ...]:
        """Every recorded portal leaving `from_map`."""
        return tuple(p for (f, _t), p in self._portals.items() if f == from_map)

    def incoming_portals(self, to_map: tuple[int, int]) -> tuple[Portal, ...]:
        """Every recorded portal arriving at `to_map`."""
        return tuple(p for (_f, t), p in self._portals.items() if t == to_map)

    def cells_labeled(
        self, label: str
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """All (map_id, cell) pairs remembered under the given recognition label."""
        return [(map_id, cell) for map_id, cell in self._labeled_cells.get(label, {}).items()]

    def healing_spots(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Known healing locations as (map_id, cell); shortcut for cells_labeled('healing_spot')."""
        return self.cells_labeled("healing_spot")

    def remember_grid(self, snap: "GridSnapshot") -> None:
        """Store the snapshot keyed by its map_id; last-write-wins."""
        self._grids[snap.map_id] = snap

    def grid_for(self, map_id: tuple[int, int]) -> "GridSnapshot | None":
        """Return the remembered grid for map_id, or None if never seen."""
        return self._grids.get(map_id)

    def node(self, map_id: tuple[int, int]) -> PlaceNode | None:
        return self.nodes.get(map_id)

    def _ensure_node(self, map_id: tuple[int, int]) -> PlaceNode:
        node = self.nodes.get(map_id)
        if node is None:
            node = PlaceNode(map_id=map_id, place_type=KNOWN_PLACES.get(map_id, "unknown"))
            self.nodes[map_id] = node
        return node
