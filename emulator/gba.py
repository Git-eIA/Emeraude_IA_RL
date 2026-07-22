"""Headless mGBA wrapper: frame stepping, inputs, RAM reads, savestates."""
from __future__ import annotations

from pathlib import Path

import mgba.core
import mgba.image
import mgba.log
import numpy as np


class GbaEmulator:
    """Game-agnostic control of one headless mGBA core instance."""

    def __init__(self, rom_path: Path | str) -> None:
        rom_path = Path(rom_path)
        if not rom_path.is_file():
            raise FileNotFoundError(rom_path)
        mgba.log.silence()
        core = mgba.core.load_path(str(rom_path))
        if core is None:
            raise ValueError(f"mGBA could not load ROM: {rom_path}")
        self._core = core
        width, height = core.desired_video_dimensions()
        self._video = mgba.image.Image(width, height)
        core.set_video_buffer(self._video)
        core.reset()

    def step(self, keys: int = 0, frames: int = 1) -> None:
        """Hold `keys` (bitmask from emulator.buttons) for `frames` frames."""
        self._core.set_keys(raw=keys)
        for _ in range(frames):
            self._core.run_frame()

    def read_bytes(self, address: int, length: int) -> bytes:
        return bytes(self._core.memory.u8[address : address + length])

    def save_state(self) -> bytes:
        state = self._core.save_raw_state()
        if state is None:
            raise RuntimeError("mGBA failed to save state")
        return bytes(state)

    def load_state(self, state: bytes) -> None:
        if not self._core.load_raw_state(state):
            raise RuntimeError("mGBA failed to load state")

    def screenshot(self) -> np.ndarray:
        """Current frame as (160, 240, 3) uint8 RGB array."""
        # mGBA's video buffer is RGBX (4 channels); convert to RGB explicitly.
        return np.asarray(self._video.to_pil().convert("RGB"), dtype=np.uint8)
