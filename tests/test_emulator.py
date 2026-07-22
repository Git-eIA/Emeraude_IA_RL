from __future__ import annotations

import numpy as np

from tests.conftest import requires_rom


@requires_rom
def test_boot_and_screenshot(rom_path):
    from emulator.gba import GbaEmulator

    emu = GbaEmulator(rom_path)
    emu.step(frames=60)
    frame = emu.screenshot()
    assert frame.shape == (160, 240, 3)
    assert frame.dtype == np.uint8


@requires_rom
def test_rom_header_readable(rom_path):
    from emulator.gba import GbaEmulator

    emu = GbaEmulator(rom_path)
    # Game code lives at 0x080000AC in the cartridge header; Emerald FR = BPEF
    assert emu.read_bytes(0x080000AC, 4) == b"BPEF"


@requires_rom
def test_savestate_roundtrip_is_deterministic(rom_path):
    from emulator.gba import GbaEmulator

    emu = GbaEmulator(rom_path)
    emu.step(frames=120)
    state = emu.save_state()
    # Run one extra frame so the video buffer is fully rendered at the saved point,
    # then take screenshot A. Same operation after load_state: one frame to repaint.
    emu.step(frames=1)
    a = emu.screenshot().copy()
    emu.step(frames=60)
    emu.load_state(state)
    emu.step(frames=1)
    b = emu.screenshot()
    np.testing.assert_array_equal(a, b)


def test_missing_rom_raises(tmp_path):
    import pytest

    from emulator.gba import GbaEmulator

    with pytest.raises(FileNotFoundError):
        GbaEmulator(tmp_path / "nope.gba")
