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
            return self.x.to_bytes(2, "little") + self.y.to_bytes(2, "little")
        if address == self._sb1 + 4:
            return bytes([1, 2])[:length]
        return b"\x00" * length

    def load_state(self, state: bytes) -> None:
        self.loaded_states.append(state)
        self.x, self.y = 5, 5

    def screenshot(self) -> np.ndarray:
        rng = np.random.default_rng(seed=self.x * 1000 + self.y)
        return rng.integers(0, 255, size=(160, 240, 3), dtype=np.uint8)
