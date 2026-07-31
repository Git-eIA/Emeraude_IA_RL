"""heal_detector: recognise a heal by its effect (HP refilled to max).

Pure logic, no emulator: the party's (current, max) HP per member goes in, a
single boolean 'a heal just happened' comes out. Reused both to LEARN a healing
spot (during movement) and to KNOW when a heal interaction is done.
"""
from __future__ import annotations


def party_is_full(hp: list[tuple[int, int]]) -> bool:
    """True if the party is non-empty and every member is at max HP."""
    if not hp:
        return False
    return all(cur >= mx for cur, mx in hp)


def party_needs_heal(hp: list[tuple[int, int]], threshold: float) -> bool:
    """True if any member has fainted (0 HP) OR the party's total HP fraction is
    below `threshold`. False for an empty party. Mixed trigger: a KO forces a
    heal even when the totals look fine; a low total forces a heal even with
    nobody KO'd."""
    if not hp:
        return False
    if any(cur == 0 for cur, _ in hp):
        return True
    total_cur = sum(cur for cur, _ in hp)
    total_max = sum(mx for _, mx in hp)
    return total_max > 0 and total_cur / total_max < threshold


class HealWatcher:
    """Fires once on the step where the party goes from not-full to full HP."""

    def __init__(self) -> None:
        # Start optimistic so an already-full first read is not a spurious heal.
        self._was_full = True

    def observe(self, hp: list[tuple[int, int]]) -> bool:
        """Feed current party HP; returns True only on the not-full -> full edge."""
        full = party_is_full(hp)
        healed = full and not self._was_full
        self._was_full = full
        return healed
