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
