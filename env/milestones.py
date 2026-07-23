"""Story milestone rewards: one-time bonuses for scripted progress events.

Extending the story chain = appending rows to starter_milestones() (or a
future table). Conditions read PlayerState only; flags-based conditions can
close over EmeraldReader.read_flag when needed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from env.game_state import PlayerState

# Map IDs from pret/pokeemerald data/maps/map_groups.json
LITTLEROOT = (0, 9)
ROUTE_101 = (0, 16)
# Brendan's and May's house ground floors; the player spawns in one of them.
PLAYER_HOUSES_1F = frozenset({(1, 0), (1, 2)})

# The rival's (May's) house. The Pokeball cutscene in the upstairs bedroom sets
# VAR_LITTLEROOT_TOWN_STATE to 1, which makes the twin guarding the Route 101
# exit step aside — without it the agent is pushed back at the north edge.
MAYS_HOUSE_1F = (1, 2)
MAYS_HOUSE_2F = (1, 3)

# Littleroot's exit to Route 101 is at the top edge; this milestone pays for
# committing northward through the boundary.
NORTH_LITTLEROOT_MAX_Y = 1


@dataclass(frozen=True)
class Milestone:
    name: str
    condition: Callable[[PlayerState], bool]
    points: float
    terminal: bool = False


def starter_milestones() -> tuple[Milestone, ...]:
    """M6 chain: intro, rival's Pokeball cutscene, leave town north, get the starter.

    Total reward: 190 points across 10 milestones.
    """
    return (
        Milestone(
            "exit_truck",
            lambda s: (s.map_group, s.map_num) == LITTLEROOT,
            5.0,
        ),
        Milestone(
            "enter_house",
            lambda s: (s.map_group, s.map_num) in PLAYER_HOUSES_1F,
            5.0,
        ),
        Milestone(
            "clock_set",
            lambda s: s.clock_set,
            15.0,
        ),
        Milestone(
            "back_outside",
            lambda s: s.clock_set and (s.map_group, s.map_num) == LITTLEROOT,
            10.0,
        ),
        Milestone(
            "enter_rival_house",
            lambda s: s.clock_set and (s.map_group, s.map_num) == MAYS_HOUSE_1F,
            5.0,
        ),
        Milestone(
            "rival_upstairs",
            lambda s: s.clock_set and (s.map_group, s.map_num) == MAYS_HOUSE_2F,
            5.0,
        ),
        Milestone(
            # clock_set + the 1..4 range guard against transient garbage RAM
            # reads during map warps (observed: town_state=40 for one step).
            "meet_rival",
            lambda s: s.clock_set and 1 <= s.town_state <= 4,
            15.0,
        ),
        Milestone(
            "north_littleroot",
            lambda s: s.clock_set
            and (s.map_group, s.map_num) == LITTLEROOT
            and s.y <= NORTH_LITTLEROOT_MAX_Y,
            10.0,
        ),
        Milestone(
            "reach_route_101",
            lambda s: (s.map_group, s.map_num) == ROUTE_101,
            20.0,
        ),
        Milestone(
            "starter_obtained",
            lambda s: s.party_count >= 1,
            100.0,
            terminal=True,
        ),
    )


class MilestoneTracker:
    """Evaluates milestones each step; each fires at most once per episode."""

    def __init__(self, milestones: tuple[Milestone, ...]) -> None:
        self._milestones = milestones
        self._fired: set[str] = set()

    @property
    def fired(self) -> frozenset[str]:
        return frozenset(self._fired)

    def reset(self) -> None:
        self._fired.clear()

    def update(self, state: PlayerState | None) -> tuple[float, bool]:
        """Returns (reward, terminated) for this step."""
        if state is None:
            return 0.0, False
        reward = 0.0
        terminated = False
        for milestone in self._milestones:
            if milestone.name in self._fired or not milestone.condition(state):
                continue
            self._fired.add(milestone.name)
            reward += milestone.points
            terminated = terminated or milestone.terminal
        return reward, terminated
