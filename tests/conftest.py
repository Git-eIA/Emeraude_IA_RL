from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM_ENV_VAR = "POKEMON_EMERALD_ROM"

requires_rom = pytest.mark.skipif(
    not os.environ.get(ROM_ENV_VAR),
    reason=f"{ROM_ENV_VAR} not set",
)


@pytest.fixture
def rom_path() -> Path:
    return Path(os.environ[ROM_ENV_VAR])
