"""Story milestone rewards: one-time bonuses for scripted progress events.

Extending the story chain = appending rows to starter_milestones() (or a
future table). Conditions read PlayerState only; flags-based conditions can
close over EmeraldReader.read_flag when needed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from env.game_state import PlayerState

# MAP_ROUTE101 from pret/pokeemerald include/constants/map_groups.h
ROUTE_101 = (0, 16)


@dataclass(frozen=True)
class Milestone:
    name: str
    condition: Callable[[PlayerState], bool]
    points: float
    terminal: bool = False


def starter_milestones() -> tuple[Milestone, ...]:
    """M5 chain: leave town northward, then obtain the starter."""
    return (
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
