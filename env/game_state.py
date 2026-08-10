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
# IWRAM pointer to the relocated SaveBlock2 struct (holds the item encryption key).
# Candidate — probe-confirmed on BPEF before the ROM smoke goes load-bearing.
SAVE_BLOCK2_PTR = 0x03005D90

# EWRAM address of gPlayerPartyCount (1 byte).
# Verified identical for BPEE and BPEF via pokebot-gen3 symbol tables.
PARTY_COUNT_ADDR = 0x020244E9

# EWRAM address of gPlayerParty: 6 x 100-byte struct Pokemon, directly after
# gPlayerPartyCount. No F: override in pokebot-gen3 language patches -> BPEF = BPEE.
PARTY_ADDR = 0x020244EC
PARTY_MON_SIZE = 100
PARTY_LEVEL_OFFSET = 84  # u8 level, unencrypted battle section (pret include/pokemon.h)
PARTY_HP_OFFSET = 86      # 0x56, u16 current HP (unencrypted; confirmed heatz123/pokeagent + pret)
PARTY_MAX_HP_OFFSET = 88  # 0x58, u16 max HP     (unencrypted; confirmed)

# offsetof(struct SaveBlock1, ...) from pret/pokeemerald
_POS_OFFSET = 0x0000  # Coords16 pos: s16 x, s16 y
_LOCATION_OFFSET = 0x0004  # WarpData location: s8 mapGroup, s8 mapNum
_FLAGS_OFFSET = 0x1270  # u8 flags[]
_FIRST_BADGE_FLAG = 0x867  # FLAG_BADGE01_GET .. FLAG_BADGE08_GET are contiguous
# FLAG_SET_WALL_CLOCK from pret/pokeemerald include/constants/flags.h.
# The intro is complete only once the bedroom wall clock has been set.
FLAG_SET_WALL_CLOCK = 0x51
# FLAG_SYS_POKEDEX_GET / FLAG_RECEIVED_RUNNING_SHOES from pret/pokeemerald
# include/constants/flags.h. Candidate ids — the probe (tools/probe_phase2_facts.py)
# confirms or replaces them on BPEF before the ROM smoke becomes load-bearing.
FLAG_SYS_POKEDEX_GET = 0x801
FLAG_RECEIVED_RUNNING_SHOES = 0x86F

# Event vars array inside SaveBlock1. Offset verified empirically on BPEF
# (2026-07-23 probe): VAR_LITTLEROOT_INTRO_STATE stepped 0->7 during the intro.
_VARS_OFFSET = 0x139C  # offsetof(struct SaveBlock1, vars)
_VARS_START = 0x4000  # first var id (pret include/constants/vars.h)
# The twin guarding Littleroot's north exit steps aside once this var is >= 1
# (set by the Pokeball cutscene in the rival's bedroom).
VAR_LITTLEROOT_TOWN_STATE = 0x4050

_EWRAM_START = 0x02000000
_EWRAM_END = 0x02040000

# Bag Items pocket: contiguous ItemSlot array inside SaveBlock1. Item quantities
# are XOR-encrypted with the low 16 bits of SaveBlock2.encryptionKey. Offsets are
# candidates from pret/pokeemerald, confirmed by the Phase 2 probe.
POKE_BALL_ITEM_ID = 0x4
_ITEMS_POCKET_OFFSET = 0x560   # offsetof(SaveBlock1, bagPocket_Items) — candidate
_SECURITY_KEY_OFFSET = 0xAC    # offsetof(SaveBlock2, encryptionKey)
_ITEM_SLOT_SIZE = 4            # u16 itemId + u16 quantity
_ITEMS_POCKET_CAPACITY = 30    # bound the scan (code-safety #2)


@dataclass(frozen=True)
class PlayerState:
    x: int
    y: int
    map_group: int
    map_num: int
    badges: int
    party_count: int
    clock_set: bool = False
    town_state: int = 0


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
            clock_set=self._flag(sb1, FLAG_SET_WALL_CLOCK),
            town_state=self._var(sb1, VAR_LITTLEROOT_TOWN_STATE),
        )

    def read_flag(self, flag_id: int) -> bool:
        """True if the event flag is set; False while save blocks relocate."""
        sb1 = self._save_block1()
        if sb1 is None:
            return False
        return self._flag(sb1, flag_id)

    def has_pokedex(self) -> bool:
        """True once the Pokédex has been received (Birch lab cutscene)."""
        return self.read_flag(FLAG_SYS_POKEDEX_GET)

    def has_running_shoes(self) -> bool:
        """True once the running shoes have been received (Route 101 cutscene)."""
        return self.read_flag(FLAG_RECEIVED_RUNNING_SHOES)

    def party_levels(self) -> list[int]:
        """Levels of the party Pokémon in slot order; empty list when no party."""
        count = min(self._read(PARTY_COUNT_ADDR, 1)[0], 6)
        return [
            self._read(PARTY_ADDR + slot * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET, 1)[0]
            for slot in range(count)
        ]

    def party_hp(self) -> list[tuple[int, int]]:
        """(current_hp, max_hp) per party Pokémon in slot order; empty when no party."""
        count = min(self._read(PARTY_COUNT_ADDR, 1)[0], 6)
        out: list[tuple[int, int]] = []
        for slot in range(count):
            base = PARTY_ADDR + slot * PARTY_MON_SIZE
            cur = int.from_bytes(self._read(base + PARTY_HP_OFFSET, 2), "little")
            mx = int.from_bytes(self._read(base + PARTY_MAX_HP_OFFSET, 2), "little")
            out.append((cur, mx))
        return out

    def has_item(self, item_id: int, min_qty: int = 1) -> bool:
        """True if the Items pocket holds >= min_qty of item_id (qty XOR-decrypted)."""
        sb1 = self._save_block1()
        sb2 = self._save_block2()
        if sb1 is None or sb2 is None:
            return False
        key = int.from_bytes(self._read(sb2 + _SECURITY_KEY_OFFSET, 4), "little") & 0xFFFF
        base = sb1 + _ITEMS_POCKET_OFFSET
        for slot in range(_ITEMS_POCKET_CAPACITY):
            entry = self._read(base + slot * _ITEM_SLOT_SIZE, _ITEM_SLOT_SIZE)
            slot_id = int.from_bytes(entry[0:2], "little")
            if slot_id != item_id:
                continue
            qty = int.from_bytes(entry[2:4], "little") ^ key
            return qty >= min_qty
        return False

    def _save_block1(self) -> int | None:
        sb1 = int.from_bytes(self._read(SAVE_BLOCK1_PTR, 4), "little")
        if not _EWRAM_START <= sb1 < _EWRAM_END:
            return None
        return sb1

    def _save_block2(self) -> int | None:
        sb2 = int.from_bytes(self._read(SAVE_BLOCK2_PTR, 4), "little")
        if not _EWRAM_START <= sb2 < _EWRAM_END:
            return None
        return sb2

    def _flag(self, sb1: int, flag_id: int) -> bool:
        byte_index, bit_index = divmod(flag_id, 8)
        raw = self._read(sb1 + _FLAGS_OFFSET + byte_index, 1)[0]
        return bool(raw >> bit_index & 1)

    def _var(self, sb1: int, var_id: int) -> int:
        raw = self._read(sb1 + _VARS_OFFSET + (var_id - _VARS_START) * 2, 2)
        return int.from_bytes(raw, "little")

    def _badge_count(self, sb1: int) -> int:
        # FLAG_BADGE01_GET..FLAG_BADGE08_GET are contiguous from _FIRST_BADGE_FLAG.
        return sum(self._flag(sb1, _FIRST_BADGE_FLAG + i) for i in range(8))


# --- Battle RAM (M7 Fighter) ---
# Base addresses confirmed by tools/probe_battle.py on BPEF (Task 1).
# TODO(probe): replace with the values printed by the probe if they differ.
GBATTLE_MONS_ADDR = 0x02024084
GBATTLE_TYPE_FLAGS_ADDR = 0x02022FEC
GBATTLE_OUTCOME_ADDR = 0x0202433A
GMOVE_RESULT_FLAGS_ADDR = 0x0202427C
BATTLE_MON_SIZE = 0x58

# Action-selection state: this byte equals 4 only while the battle waits for the
# player to pick an action (ATTAQUE/SAC/POKEMON/FUITE). It is 0 during the move
# submenu, move animations, and result dialogue. Found empirically by diffing
# RAM at the menu vs mid-animation across turns, and validated by driving 5/5
# wild battles to a win with it.
GBATTLE_ACTION_MENU_ADDR = 0x02023266
ACTION_MENU_VALUE = 4

# Per-BattlePokemon offsets (pret/pokeemerald, region-stable).
_BM_SPECIES = 0x00
_BM_MOVES = 0x0C
# 0x20 is `ability`; type1/type2 follow it (confirmed empirically: reads at
# 0x20 returned ability IDs like 66=Blaze/53=Pickup, out of the 0..17 range).
_BM_TYPE1 = 0x21
_BM_TYPE2 = 0x22
_BM_PP = 0x24
_BM_HP = 0x28
_BM_LEVEL = 0x2A
_BM_MAXHP = 0x2C

_MOVE_RESULT_SUPER_EFFECTIVE = 0x0008

# gBattleTypeFlags bit that distinguishes trainer battles from wild encounters.
# Source: pret/pokeemerald include/constants/battle.h BATTLE_TYPE_TRAINER.
BATTLE_TYPE_TRAINER = 0x0008


@dataclass(frozen=True)
class MoveInfo:
    move_id: int
    pp: int


@dataclass(frozen=True)
class BattleState:
    in_battle: bool
    my_hp: int
    my_max_hp: int
    my_level: int
    my_types: tuple[int, int]
    my_moves: tuple[MoveInfo, MoveInfo, MoveInfo, MoveInfo]
    opp_hp: int
    opp_max_hp: int
    opp_level: int
    opp_types: tuple[int, int]
    opp_species: int
    outcome: int
    last_move_super_effective: bool


class BattleReader:
    """Parses battle RAM into an immutable BattleState.

    Takes the same read_bytes(addr, size) callable as EmeraldReader so it is
    testable with crafted bytes and needs no ROM.
    """

    def __init__(self, read_bytes: ReadFn) -> None:
        self._read = read_bytes

    def battle_state(self) -> BattleState:
        flags = self._u16(GBATTLE_TYPE_FLAGS_ADDR)
        my = self._read_mon(GBATTLE_MONS_ADDR)
        opp = self._read_mon(GBATTLE_MONS_ADDR + BATTLE_MON_SIZE)
        outcome = self._u8(GBATTLE_OUTCOME_ADDR)
        # In battle iff flags set AND opponent has a real max HP AND the battle
        # has no terminal outcome yet. gBattleTypeFlags/gBattleMons are NOT
        # cleared when a battle ends, so a savestate captured just after one
        # (e.g. the forced starter battle) keeps them set — gating on
        # gBattleOutcome == 0 rejects that residual state. Every consumer already
        # treats outcome != 0 as "battle over", so this only tightens in_battle.
        in_battle = flags != 0 and opp["max_hp"] > 0 and outcome == 0
        move_result = self._u16(GMOVE_RESULT_FLAGS_ADDR)
        return BattleState(
            in_battle=in_battle,
            my_hp=my["hp"],
            my_max_hp=my["max_hp"],
            my_level=my["level"],
            my_types=my["types"],
            my_moves=my["moves"],
            opp_hp=opp["hp"],
            opp_max_hp=opp["max_hp"],
            opp_level=opp["level"],
            opp_types=opp["types"],
            opp_species=opp["species"],
            outcome=outcome,
            last_move_super_effective=bool(move_result & _MOVE_RESULT_SUPER_EFFECTIVE),
        )

    def battle_starting(self) -> bool:
        """True while battle flags are set with no terminal outcome yet.

        Covers the intro window — flags set before gBattleMons populates, so
        battle_state().in_battle is still False — and the active battle. Residual
        flags from a loaded post-battle savestate carry outcome != 0 and read
        False, so a freshly loaded savestate does not hang the navigator.
        """
        flags = self._u16(GBATTLE_TYPE_FLAGS_ADDR)
        outcome = self._u8(GBATTLE_OUTCOME_ADDR)
        return flags != 0 and outcome == 0

    def is_trainer_battle(self) -> bool:
        """True when the current battle is against a trainer (not wild)."""
        return bool(self._u16(GBATTLE_TYPE_FLAGS_ADDR) & BATTLE_TYPE_TRAINER)

    def _read_mon(self, base: int) -> dict:
        moves = tuple(
            MoveInfo(
                move_id=self._u16(base + _BM_MOVES + 2 * i),
                pp=self._u8(base + _BM_PP + i),
            )
            for i in range(4)
        )
        return {
            "species": self._u16(base + _BM_SPECIES),
            "hp": self._u16(base + _BM_HP),
            "max_hp": self._u16(base + _BM_MAXHP),
            "level": self._u8(base + _BM_LEVEL),
            "types": (self._u8(base + _BM_TYPE1), self._u8(base + _BM_TYPE2)),
            "moves": moves,
        }

    def at_action_menu(self) -> bool:
        """True iff the battle is waiting for the player to choose an action."""
        return self._u8(GBATTLE_ACTION_MENU_ADDR) == ACTION_MENU_VALUE

    def _u8(self, addr: int) -> int:
        return self._read(addr, 1)[0]

    def _u16(self, addr: int) -> int:
        b = self._read(addr, 2)
        return b[0] | (b[1] << 8)
