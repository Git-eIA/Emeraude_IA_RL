"""encounter_detector: recognise grass by its effect (a wild battle started here).

Pure logic, no emulator: a boolean 'am I in battle now' goes in, a single
boolean 'a wild battle just started' comes out. Structural twin of
heal_detector.HealWatcher, but watches the battle flag instead of party HP.
Reused both to LEARN a grass cell (during movement/cartography) and to KNOW
when grind's walk-loop has triggered a battle.
"""
from __future__ import annotations


class EncounterWatcher:
    """Fires once on the step where a wild battle transitions from absent to present."""

    def __init__(self) -> None:
        # Start optimistic so an already-in-battle first read is not a spurious start.
        self._was_in_battle = True

    def observe(self, in_battle: bool) -> bool:
        """Feed the current battle flag; returns True only on the absent -> present edge."""
        started = in_battle and not self._was_in_battle
        self._was_in_battle = in_battle
        return started
