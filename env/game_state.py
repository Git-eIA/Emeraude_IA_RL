"""Typed readers for Pokémon Emerald RAM.

Emerald relocates its save blocks (anti-cheat DMA), so all SaveBlock1 fields
are reached through the pointer at SAVE_BLOCK1_PTR. Addresses cross-checked
against pret/pokeemerald and pokebot-gen3.

BPEF (French Emerald) address verification:
  Source: pokebot-gen3 modules/data/symbols/pokeemerald.sym (base table) +
          modules/data/symbols/patches/language/pokeemerald.yml (language patches).
  Neither gSaveBlock1Ptr nor gPlayerPartyCount have an 'F:' entry in the YAML
  patch file, confirming BPEF uses the same addresses as BPEE (US/Europe).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

ReadFn = Callable[[int, int], bytes]

# IWRAM pointer to the relocated SaveBlock1 struct.
# Verified identical for BPEE and BPEF via pokebot-gen3 symbol tables.
SAVE_BLOCK1_PTR = 0x03005D8C

# EWRAM address of gPlayerPartyCount (1 byte).
# Verified identical for BPEE and BPEF via pokebot-gen3 symbol tables.
PARTY_COUNT_ADDR = 0x020244E9

# EWRAM address of gPlayerParty: 6 x 100-byte struct Pokemon, directly after
# gPlayerPartyCount. No F: override in pokebot-gen3 language patches -> BPEF = BPEE.
PARTY_ADDR = 0x020244EC
PARTY_MON_SIZE = 100
PARTY_LEVEL_OFFSET = 84  # u8 level, unencrypted battle section (pret include/pokemon.h)

# offsetof(struct SaveBlock1, ...) from pret/pokeemerald
_POS_OFFSET = 0x0000  # Coords16 pos: s16 x, s16 y
_LOCATION_OFFSET = 0x0004  # WarpData location: s8 mapGroup, s8 mapNum
_FLAGS_OFFSET = 0x1270  # u8 flags[]
_FIRST_BADGE_FLAG = 0x867  # FLAG_BADGE01_GET .. FLAG_BADGE08_GET are contiguous

_EWRAM_START = 0x02000000
_EWRAM_END = 0x02040000


@dataclass(frozen=True)
class PlayerState:
    x: int
    y: int
    map_group: int
    map_num: int
    badges: int
    party_count: int


class EmeraldReader:
    """Parses Emerald game state through an injected raw-memory reader."""

    def __init__(self, read: ReadFn) -> None:
        self._read = read

    def player_state(self) -> PlayerState | None:
        """Current player state, or None while save blocks are relocating."""
        sb1 = self._save_block1()
        if sb1 is None:
            return None
        pos = self._read(sb1 + _POS_OFFSET, 4)
        location = self._read(sb1 + _LOCATION_OFFSET, 2)
        return PlayerState(
            x=int.from_bytes(pos[0:2], "little", signed=True),
            y=int.from_bytes(pos[2:4], "little", signed=True),
            map_group=location[0],
            map_num=location[1],
            badges=self._badge_count(sb1),
            party_count=self._read(PARTY_COUNT_ADDR, 1)[0],
        )

    def read_flag(self, flag_id: int) -> bool:
        """True if the event flag is set; False while save blocks relocate."""
        sb1 = self._save_block1()
        if sb1 is None:
            return False
        return self._flag(sb1, flag_id)

    def party_levels(self) -> list[int]:
        """Levels of the party Pokémon in slot order; empty list when no party."""
        count = min(self._read(PARTY_COUNT_ADDR, 1)[0], 6)
        return [
            self._read(PARTY_ADDR + slot * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET, 1)[0]
            for slot in range(count)
        ]

    def _save_block1(self) -> int | None:
        sb1 = int.from_bytes(self._read(SAVE_BLOCK1_PTR, 4), "little")
        if not _EWRAM_START <= sb1 < _EWRAM_END:
            return None
        return sb1

    def _flag(self, sb1: int, flag_id: int) -> bool:
        byte_index, bit_index = divmod(flag_id, 8)
        raw = self._read(sb1 + _FLAGS_OFFSET + byte_index, 1)[0]
        return bool(raw >> bit_index & 1)

    def _badge_count(self, sb1: int) -> int:
        # FLAG_BADGE01_GET..FLAG_BADGE08_GET are contiguous from _FIRST_BADGE_FLAG.
        return sum(self._flag(sb1, _FIRST_BADGE_FLAG + i) for i in range(8))
