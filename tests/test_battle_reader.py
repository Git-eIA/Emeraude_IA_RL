"""BattleReader parses battle RAM into an immutable BattleState (no ROM)."""
from __future__ import annotations

from env.game_state import (
    BATTLE_MON_SIZE,
    GBATTLE_MONS_ADDR,
    GBATTLE_OUTCOME_ADDR,
    GBATTLE_TYPE_FLAGS_ADDR,
    GMOVE_RESULT_FLAGS_ADDR,
    BattleReader,
    BattleState,
)

MOVE_RESULT_SUPER_EFFECTIVE = 0x0008


def _u16(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def _mon_bytes(
    *,
    species: int,
    hp: int,
    max_hp: int,
    level: int,
    types: tuple[int, int],
    moves: tuple[int, int, int, int],
    pp: tuple[int, int, int, int],
) -> bytearray:
    buf = bytearray(BATTLE_MON_SIZE)
    buf[0x00:0x02] = _u16(species)
    for i, m in enumerate(moves):
        buf[0x0C + 2 * i : 0x0C + 2 * i + 2] = _u16(m)
    buf[0x21] = types[0]
    buf[0x22] = types[1]
    for i, p in enumerate(pp):
        buf[0x24 + i] = p
    buf[0x28:0x2A] = _u16(hp)
    buf[0x2A] = level
    buf[0x2C:0x2E] = _u16(max_hp)
    return buf


def _make_reader(
    *,
    in_battle: bool = True,
    outcome: int = 0,
    move_result: int = 0,
    player: bytearray | None = None,
    opponent: bytearray | None = None,
) -> BattleReader:
    player = player or _mon_bytes(
        species=1, hp=19, max_hp=19, level=5, types=(12, 12),
        moves=(33, 45, 0, 0), pp=(35, 40, 0, 0),
    )
    opponent = opponent or _mon_bytes(
        species=4, hp=18, max_hp=18, level=5, types=(10, 10),
        moves=(10, 43, 0, 0), pp=(35, 30, 0, 0),
    )

    def read(addr: int, size: int) -> bytes:
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16(1 if in_battle else 0) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16(move_result)
        if GBATTLE_MONS_ADDR <= addr < GBATTLE_MONS_ADDR + BATTLE_MON_SIZE:
            offset = addr - GBATTLE_MONS_ADDR
            return bytes(player[offset : offset + size])
        if (
            GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
            <= addr
            < GBATTLE_MONS_ADDR + 2 * BATTLE_MON_SIZE
        ):
            offset = addr - (GBATTLE_MONS_ADDR + BATTLE_MON_SIZE)
            return bytes(opponent[offset : offset + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")

    return BattleReader(read)


def test_reads_hp_level_types_and_moves() -> None:
    state = _make_reader().battle_state()
    assert isinstance(state, BattleState)
    assert state.in_battle is True
    assert state.my_hp == 19 and state.my_max_hp == 19 and state.my_level == 5
    assert state.my_types == (12, 12)
    assert state.opp_hp == 18 and state.opp_max_hp == 18 and state.opp_level == 5
    assert state.opp_types == (10, 10)
    assert state.opp_species == 4
    assert state.outcome == 0
    assert len(state.my_moves) == 4
    assert state.my_moves[0].move_id == 33
    assert state.my_moves[0].pp == 35


def test_not_in_battle_when_flags_zero_and_opp_maxhp_zero() -> None:
    opp = _mon_bytes(
        species=0, hp=0, max_hp=0, level=0, types=(0, 0),
        moves=(0, 0, 0, 0), pp=(0, 0, 0, 0),
    )
    state = _make_reader(in_battle=False, opponent=opp).battle_state()
    assert state.in_battle is False


def test_super_effective_flag_detected() -> None:
    state = _make_reader(move_result=MOVE_RESULT_SUPER_EFFECTIVE).battle_state()
    assert state.last_move_super_effective is True
    state2 = _make_reader(move_result=0).battle_state()
    assert state2.last_move_super_effective is False


def test_outcome_nonzero_when_battle_ends() -> None:
    state = _make_reader(outcome=1).battle_state()
    assert state.outcome == 1
