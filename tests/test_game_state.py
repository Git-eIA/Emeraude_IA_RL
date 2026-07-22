from __future__ import annotations

from pathlib import Path

import pytest

from env.game_state import (
    PARTY_COUNT_ADDR,
    SAVE_BLOCK1_PTR,
    EmeraldReader,
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
) -> dict[int, bytes]:
    sb1 = 0x02025A00  # arbitrary but valid EWRAM address for the fake
    save_block1 = bytearray(0x1400)  # must fit flags region up to 0x137E
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
    assert reader.party_levels() == []
    assert reader.read_flag(FLAG_SET_WALL_CLOCK) is False
