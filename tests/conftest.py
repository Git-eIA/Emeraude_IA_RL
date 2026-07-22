from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROM_ENV_VAR = "POKEMON_EMERALD_ROM"

requires_rom = pytest.mark.skipif(
    not os.environ.get(ROM_ENV_VAR),
    reason=f"{ROM_ENV_VAR} not set",
)


@pytest.fixture
def rom_path() -> Path:
    return Path(os.environ[ROM_ENV_VAR])


class FakeEmulator:
    """Emulator double: moving player, deterministic frames, no ROM needed."""

    def __init__(self) -> None:
        self.x = 5
        self.y = 5
        self.map_group = 0
        self.map_num = 10  # Oldale: neutral map, fires no milestone
        self.clock_set = False
        self.party_count = 0
        self.party_levels: list[int] = []
        self.loaded_states: list[bytes] = []
        self._sb1 = 0x02025A00

    def step(self, keys: int = 0, frames: int = 1) -> None:
        from emulator import buttons

        if keys & buttons.KEY_RIGHT:
            self.x += 1
        if keys & buttons.KEY_LEFT:
            self.x -= 1
        if keys & buttons.KEY_UP:
            self.y -= 1
        if keys & buttons.KEY_DOWN:
            self.y += 1

    def read_bytes(self, address: int, length: int) -> bytes:
        from env import game_state

        if address == game_state.SAVE_BLOCK1_PTR:
            return self._sb1.to_bytes(4, "little")[:length]
        if address == self._sb1:
            # Coordinates are s16 in Emerald; use signed=True to handle negative values.
            return self.x.to_bytes(2, "little", signed=True) + self.y.to_bytes(2, "little", signed=True)
        if address == self._sb1 + 4:
            return bytes([self.map_group, self.map_num])[:length]
        if address == game_state.PARTY_COUNT_ADDR:
            return bytes([self.party_count])[:length]
        for slot, level in enumerate(self.party_levels):
            slot_addr = game_state.PARTY_ADDR + slot * game_state.PARTY_MON_SIZE
            if address == slot_addr + game_state.PARTY_LEVEL_OFFSET:
                return bytes([level])[:length]
        # Flags byte 10 holds FLAG_SET_WALL_CLOCK (0x51) at bit 1.
        if address == self._sb1 + 0x1270 + 10:
            return (b"\x02" if self.clock_set else b"\x00") * length
        return b"\x00" * length

    def load_state(self, state: bytes) -> None:
        self.loaded_states.append(state)
        self.x, self.y = 5, 5
        self.clock_set = False
        self.party_count = 0
        self.party_levels = []

    def screenshot(self) -> np.ndarray:
        # numpy seeds must be non-negative; abs() handles negative coordinates.
        rng = np.random.default_rng(seed=abs(self.x * 1000 + self.y))
        return rng.integers(0, 255, size=(160, 240, 3), dtype=np.uint8)
