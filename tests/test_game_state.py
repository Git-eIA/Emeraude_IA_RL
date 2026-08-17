from __future__ import annotations

from pathlib import Path

import pytest

from env.game_state import (
    PARTY_COUNT_ADDR,
    POKE_BALL_ITEM_ID,
    SAVE_BLOCK1_PTR,
    SAVE_BLOCK2_PTR,
    _FLAGS_OFFSET,
    _ITEM_SLOT_SIZE,
    _ITEMS_POCKET_OFFSET,
    _POKEBALLS_POCKET_OFFSET,
    _SECURITY_KEY_OFFSET,
    EmeraldReader,
    FLAG_SYS_POKEDEX_GET,
    FLAG_RECEIVED_RUNNING_SHOES,
    PlayerState,
)
from tests.conftest import requires_rom


def make_fake_read(memory: dict[int, bytes]):
    """Fake reader backed by a sparse address->bytes dict."""

    def read(address: int, length: int) -> bytes:
        for base, blob in memory.items():
            if base <= address and address + length <= base + len(blob):
                offset = address - base
                return blob[offset : offset + length]
        return b"\x00" * length

    return read


def build_memory(
    *,
    x: int,
    y: int,
    map_group: int,
    map_num: int,
    badge_bits: int,
    party_count: int,
    clock_set: bool = False,
    town_state: int = 0,
) -> dict[int, bytes]:
    sb1 = 0x02025A00  # arbitrary but valid EWRAM address for the fake
    save_block1 = bytearray(0x1600)  # must fit the vars array up to 0x159C
    save_block1[0:2] = x.to_bytes(2, "little", signed=True)
    save_block1[2:4] = y.to_bytes(2, "little", signed=True)
    save_block1[4] = map_group
    save_block1[5] = map_num
    # Badge flags start at flag 0x867 -> byte 0x10C bit 7 of the flags array
    flags_value = badge_bits << 7
    save_block1[0x1270 + 0x10C : 0x1270 + 0x10E] = flags_value.to_bytes(2, "little")
    if clock_set:
        # FLAG_SET_WALL_CLOCK = 0x51 -> byte 10, bit 1 of the flags array
        save_block1[0x1270 + 10] |= 0b10
    # VAR_LITTLEROOT_TOWN_STATE = 0x4050 -> vars index 0x50, u16 LE
    save_block1[0x139C + 0xA0 : 0x139C + 0xA2] = town_state.to_bytes(2, "little")
    return {
        SAVE_BLOCK1_PTR: sb1.to_bytes(4, "little"),
        sb1: bytes(save_block1),
        PARTY_COUNT_ADDR: bytes([party_count]),
    }


def test_reads_player_state():
    memory = build_memory(x=12, y=7, map_group=3, map_num=1, badge_bits=0b00000111, party_count=2)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state == PlayerState(x=12, y=7, map_group=3, map_num=1, badges=3, party_count=2)


def test_negative_coords_are_signed():
    # Coords16 is s16; some map transitions use negative tile coordinates.
    memory = build_memory(x=-3, y=-1, map_group=0, map_num=9, badge_bits=0, party_count=1)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert (state.x, state.y) == (-3, -1)


def test_invalid_save_block_pointer_returns_none():
    memory = {SAVE_BLOCK1_PTR: (0x00000000).to_bytes(4, "little")}
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.player_state() is None


def test_zero_badges():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.badges == 0


def test_all_badges():
    memory = build_memory(x=1, y=1, map_group=1, map_num=1, badge_bits=0xFF, party_count=6)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.badges == 8


def add_party(memory: dict[int, bytes], levels: list[int]) -> None:
    """Add gPlayerParty structs with the given levels to fake memory."""
    from env.game_state import PARTY_ADDR, PARTY_LEVEL_OFFSET, PARTY_MON_SIZE

    party = bytearray(PARTY_MON_SIZE * 6)
    for slot, level in enumerate(levels):
        party[slot * PARTY_MON_SIZE + PARTY_LEVEL_OFFSET] = level
    memory[PARTY_ADDR] = bytes(party)


def test_read_flag_set_and_unset():
    # badge_bits=0b1 sets flag 0x867; flag 0x868 stays clear
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0b1, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.read_flag(0x867) is True
    assert reader.read_flag(0x868) is False


def test_read_flag_invalid_pointer_is_false():
    memory = {SAVE_BLOCK1_PTR: (0x00000000).to_bytes(4, "little")}
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.read_flag(0x867) is False


def test_clock_set_flag_read_into_state():
    memory = build_memory(
        x=0, y=0, map_group=1, map_num=1, badge_bits=0, party_count=0, clock_set=True
    )
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.clock_set is True


def test_clock_not_set_by_default():
    memory = build_memory(x=0, y=0, map_group=1, map_num=1, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.clock_set is False


def test_town_state_read_into_state():
    memory = build_memory(
        x=0, y=0, map_group=0, map_num=9, badge_bits=0, party_count=0, town_state=1
    )
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.town_state == 1


def test_town_state_zero_by_default():
    memory = build_memory(x=0, y=0, map_group=0, map_num=9, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    state = reader.player_state()
    assert state is not None
    assert state.town_state == 0


def test_party_levels_empty():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=0)
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.party_levels() == []


def test_party_levels_two_pokemon():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=2)
    add_party(memory, [5, 12])
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.party_levels() == [5, 12]


def test_party_levels_count_clamped_to_six():
    memory = build_memory(x=0, y=0, map_group=0, map_num=0, badge_bits=0, party_count=200)
    add_party(memory, [5, 5, 5, 5, 5, 5])
    reader = EmeraldReader(make_fake_read(memory))
    assert len(reader.party_levels()) == 6


@requires_rom
def test_real_rom_state_after_initial_savestate(rom_path):
    """Sanity check against the real game: truck-start savestate is readable."""
    from emulator.gba import GbaEmulator
    from env.game_state import FLAG_SET_WALL_CLOCK

    state_file = Path("states/initial.state")
    if not state_file.is_file():
        pytest.skip("states/initial.state not created yet")
    emu = GbaEmulator(rom_path)
    emu.load_state(state_file.read_bytes())
    emu.step(frames=10)
    reader = EmeraldReader(emu.read_bytes)
    state = reader.player_state()
    assert state is not None
    assert (state.map_group, state.map_num) == (25, 40)  # InsideOfTruck
    assert state.party_count == 0
    assert state.badges == 0
    assert state.clock_set is False
    assert state.town_state == 0
    assert reader.party_levels() == []
    assert reader.read_flag(FLAG_SET_WALL_CLOCK) is False


def test_party_hp_reads_current_and_max_per_member() -> None:
    from env.game_state import (
        EmeraldReader, PARTY_ADDR, PARTY_COUNT_ADDR, PARTY_MON_SIZE,
        PARTY_HP_OFFSET, PARTY_MAX_HP_OFFSET,
    )

    mem: dict[int, int] = {PARTY_COUNT_ADDR: 2}
    for slot, (cur, mx) in enumerate([(12, 34), (5, 5)]):
        base = PARTY_ADDR + slot * PARTY_MON_SIZE
        mem[base + PARTY_HP_OFFSET] = cur & 0xFF
        mem[base + PARTY_HP_OFFSET + 1] = cur >> 8
        mem[base + PARTY_MAX_HP_OFFSET] = mx & 0xFF
        mem[base + PARTY_MAX_HP_OFFSET + 1] = mx >> 8

    def read(addr: int, length: int) -> bytes:
        return bytes(mem.get(addr + i, 0) for i in range(length))

    reader = EmeraldReader(read)
    assert reader.party_hp() == [(12, 34), (5, 5)]


def test_party_hp_empty_when_no_party() -> None:
    from env.game_state import EmeraldReader

    def read(addr: int, length: int) -> bytes:
        return bytes(0 for _ in range(length))  # count = 0

    assert EmeraldReader(read).party_hp() == []


def _reader_with_flag(flag_id: int, value: bool) -> EmeraldReader:
    """A reader whose SaveBlock1 has exactly `flag_id` set to `value`."""
    sb1 = 0x02025734
    byte_index, bit_index = divmod(flag_id, 8)
    flag_addr = sb1 + _FLAGS_OFFSET + byte_index
    flag_byte = (1 << bit_index) if value else 0

    def read(addr: int, size: int) -> bytes:
        if addr == SAVE_BLOCK1_PTR and size == 4:
            return sb1.to_bytes(4, "little")
        if addr == flag_addr and size == 1:
            return bytes([flag_byte])
        return bytes(size)

    return EmeraldReader(read)


def test_has_pokedex_true_when_flag_set() -> None:
    assert _reader_with_flag(FLAG_SYS_POKEDEX_GET, True).has_pokedex() is True


def test_has_pokedex_false_when_flag_clear() -> None:
    assert _reader_with_flag(FLAG_SYS_POKEDEX_GET, False).has_pokedex() is False


def test_has_running_shoes_true_when_flag_set() -> None:
    assert _reader_with_flag(FLAG_RECEIVED_RUNNING_SHOES, True).has_running_shoes() is True


def test_has_running_shoes_false_when_flag_clear() -> None:
    assert _reader_with_flag(FLAG_RECEIVED_RUNNING_SHOES, False).has_running_shoes() is False


def _reader_with_item(
    item_id: int, real_qty: int, security_key: int,
    pocket_offset: int = _ITEMS_POCKET_OFFSET,
) -> EmeraldReader:
    """A reader whose pocket (default Items) slot 0 holds (item_id, real_qty) encrypted."""
    sb1 = 0x02025734
    sb2 = 0x02027000
    stored_qty = real_qty ^ (security_key & 0xFFFF)
    slot_addr = sb1 + pocket_offset
    key_addr = sb2 + _SECURITY_KEY_OFFSET

    def read(addr: int, size: int) -> bytes:
        if addr == SAVE_BLOCK1_PTR and size == 4:
            return sb1.to_bytes(4, "little")
        if addr == SAVE_BLOCK2_PTR and size == 4:
            return sb2.to_bytes(4, "little")
        if addr == key_addr and size == 4:
            return security_key.to_bytes(4, "little")
        if addr == slot_addr and size == _ITEM_SLOT_SIZE:
            return item_id.to_bytes(2, "little") + stored_qty.to_bytes(2, "little")
        return bytes(size)

    return EmeraldReader(read)


def test_has_item_decrypts_quantity_and_meets_threshold() -> None:
    reader = _reader_with_item(POKE_BALL_ITEM_ID, real_qty=5, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=5) is True


def test_has_item_below_threshold_is_false() -> None:
    reader = _reader_with_item(POKE_BALL_ITEM_ID, real_qty=3, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=5) is False


def test_has_item_absent_is_false() -> None:
    reader = _reader_with_item(item_id=0x0, real_qty=0, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=1) is False


def _reader_with_poke_balls(real_qty: int, security_key: int) -> EmeraldReader:
    """A reader whose Balls pocket slot 0 holds (POKE_BALL, real_qty) encrypted."""
    return _reader_with_item(
        POKE_BALL_ITEM_ID, real_qty, security_key,
        pocket_offset=_POKEBALLS_POCKET_OFFSET,
    )


def test_has_poke_balls_decrypts_quantity_and_meets_threshold() -> None:
    reader = _reader_with_poke_balls(real_qty=5, security_key=0x1234ABCD)
    assert reader.has_poke_balls(min_qty=5) is True


def test_has_poke_balls_below_threshold_is_false() -> None:
    reader = _reader_with_poke_balls(real_qty=3, security_key=0x1234ABCD)
    assert reader.has_poke_balls(min_qty=5) is False


def test_has_poke_balls_absent_is_false() -> None:
    reader = _reader_with_item(item_id=0x0, real_qty=0, security_key=0x1234ABCD)
    assert reader.has_poke_balls(min_qty=1) is False


def test_poke_balls_not_visible_via_has_item() -> None:
    """Regression: Poke Balls live in the Balls pocket, not the Items pocket that
    has_item scans — the bug this fix corrects. has_item must not see them."""
    reader = _reader_with_poke_balls(real_qty=5, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=1) is False
