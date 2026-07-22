# Pokémon Emerald RL Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working pipeline where a PPO agent trains on Pokémon Emerald through a headless mGBA emulator (milestones M1-M4 of the design spec).

**Architecture:** Three decoupled layers: `emulator/` (game-agnostic mGBA wrapper), `env/` (Gymnasium env + Emerald RAM readers + exploration reward), `agent/` (PPO training on MPS). The env receives the emulator by injection so unit tests run with a fake emulator, no ROM needed.

**Tech Stack:** Python 3.12, libmgba-py (mGBA Python bindings), Gymnasium, stable-baselines3 (PPO), PyTorch (MPS), NumPy, Pillow, pygame (interactive tool), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-22-pokemon-emerald-rl-design.md`

**ROM policy:** The user's Emerald ROM is NEVER committed. Tests needing a real ROM read the `POKEMON_EMERALD_ROM` env var and skip when unset. Savestates in `states/` are gitignored too.

**ROM version:** The user's ROM is **French Emerald (game code BPEF)** at
`roms/pokemon_emerald_fr.gba` (gitignored). All RAM addresses documented for
the US version (BPEE) MUST be verified against the French symbol tables before
use — pokebot-gen3 ships per-language symbol files (`modules/data/symbols/` in
https://github.com/40Cakes/pokebot-gen3): look up `gSaveBlock1Ptr` and
`gPlayerPartyCount` for BPEF. The SaveBlock1 struct offsets (pos, location,
flags) come from the game's own code and are identical across languages; only
the absolute EWRAM/IWRAM symbol addresses may shift. Use
`POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba` everywhere the plan says
`<user-rom-path>`.

---

## File Structure

```
Emu/
├── pyproject.toml            # project metadata, deps, ruff/pytest config
├── .gitignore                # ROMs, states, venv, runs, checkpoints
├── README.md                 # quickstart
├── docs/architecture/        # README.md + modules.md (module index)
├── emulator/
│   ├── __init__.py
│   ├── buttons.py            # GBA key bitmask constants
│   └── gba.py                # GbaEmulator: boot/step/read_bytes/states/screenshot
├── env/
│   ├── __init__.py
│   ├── game_state.py         # Emerald RAM readers (position, map, badges, party)
│   ├── rewards.py            # ExplorationTracker
│   └── pokemon_env.py        # PokemonEmeraldEnv (Gymnasium)
├── agent/
│   ├── __init__.py
│   └── train.py              # PPO training entrypoint (MPS, parallel envs)
├── tools/
│   ├── run_scripted.py       # M1 check: scripted inputs + PNG dumps
│   └── play_interactive.py   # pygame play; create states/initial.state
└── tests/
    ├── conftest.py           # fixtures: real-ROM skip logic, FakeEmulator
    ├── test_emulator.py
    ├── test_game_state.py
    ├── test_rewards.py
    ├── test_env.py
    └── test_train_smoke.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`
- Create: `emulator/__init__.py`, `env/__init__.py`, `agent/__init__.py`, `tests/__init__.py`
- Create: `docs/architecture/README.md`, `docs/architecture/modules.md`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "emu"
version = "0.1.0"
description = "Deep RL agent for Pokémon Emerald via headless mGBA"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "gymnasium>=1.0",
    "stable-baselines3>=2.4",
    "torch>=2.4",
    "pillow>=10.0",
    "pygame>=2.5",
    "tensorboard>=2.16",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["emulator", "env", "agent"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/

# Never commit game assets
*.gba
*.sav
roms/
states/

# Training artifacts
runs/
checkpoints/
```

- [ ] **Step 3: Write `README.md`**

```markdown
# Emu — Pokémon Emerald RL

A PPO agent that learns to play Pokémon Emerald through a headless mGBA core.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    # libmgba-py: see docs/architecture/README.md (built separately)

Provide your own legally dumped Emerald ROM:

    export POKEMON_EMERALD_ROM=~/roms/pokemon_emerald.gba

## Usage

    python tools/play_interactive.py     # play, create states/initial.state
    python agent/train.py                # train PPO
    tensorboard --logdir runs/

## Tests

    pytest            # ROM-dependent tests skip if POKEMON_EMERALD_ROM unset
```

- [ ] **Step 4: Create empty `__init__.py` files**

Create `emulator/__init__.py`, `env/__init__.py`, `agent/__init__.py`,
`tests/__init__.py` — all empty.

- [ ] **Step 5: Write `docs/architecture/README.md`**

```markdown
# Architecture

Three decoupled layers (see ADR-001 and the design spec):

- `emulator/` — game-agnostic mGBA wrapper (libmgba-py). No Pokémon knowledge.
- `env/` — Pokémon Emerald Gymnasium env: pixel observations, RAM-derived rewards.
- `agent/` — PPO training (stable-baselines3, MPS). Consumes the Gymnasium API only.

## libmgba-py install

libmgba-py is not on PyPI. Two options, in order of preference:

1. Prebuilt wheel from https://github.com/hanzi/libmgba-py/releases
   (pick the cp312 / macOS arm64 asset): `pip install <wheel-url-or-file>`
2. Build from that repo's source following its README (requires `brew install
   cmake libzip libpng ffmpeg`).

Verify with: `python -c "import mgba.core; print('ok')"`

Module index: see `modules.md`.
```

- [ ] **Step 6: Write `docs/architecture/modules.md`**

```markdown
# Module index

| Module | Responsibility | Public API | Depends on |
|--------|---------------|------------|------------|
| emulator/buttons.py | GBA key bitmask constants | KEY_A..KEY_L | — |
| emulator/gba.py | Headless mGBA wrapper | GbaEmulator | libmgba-py, buttons |
| env/game_state.py | Emerald RAM parsing | EmeraldReader, PlayerState | — (reader injected) |
| env/rewards.py | Exploration reward | ExplorationTracker | game_state |
| env/pokemon_env.py | Gymnasium env | PokemonEmeraldEnv | game_state, rewards |
| agent/train.py | PPO training entrypoint | main() | env, sb3, torch |
| tools/run_scripted.py | Scripted-input M1 check | CLI | emulator |
| tools/play_interactive.py | Human play + savestate creation | CLI | emulator, pygame |
```

- [ ] **Step 7: Create venv, install, verify tooling**

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
ruff check . && pytest --collect-only -q
```

Expected: install succeeds; ruff clean; pytest collects 0 tests without error.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore README.md emulator/ env/ agent/ tests/ docs/architecture/
git commit -m "chore: project scaffolding (packages, tooling, architecture docs)"
```

---

### Task 2: Install libmgba-py and prove the core works

**Files:**
- Create: `tools/smoke_mgba.py`

This task has a native-build component. Follow the order below; stop at the
first option that works.

- [ ] **Step 1: Try a prebuilt wheel**

Check https://github.com/hanzi/libmgba-py/releases for a `cp312`
`macosx_arm64` wheel. If present:

```bash
source .venv/bin/activate
pip install <downloaded-wheel-file>
python -c "import mgba.core; print('mgba ok')"
```

Expected: `mgba ok`.

- [ ] **Step 2 (only if Step 1 fails): Build from source**

```bash
brew install cmake libzip libpng ffmpeg
git clone https://github.com/hanzi/libmgba-py /tmp/libmgba-py
# Follow that repo's README build instructions for macOS, then:
pip install /tmp/libmgba-py  # or the wheel it produces
python -c "import mgba.core; print('mgba ok')"
```

- [ ] **Step 3: Discover the exact bindings API**

The wrapper in Task 3 uses `load_path`, `run_frame`, key setting, memory
access, raw savestates, and a video buffer. Confirm the exact names first:

```bash
python - <<'EOF'
import inspect
import mgba.core
import mgba.image
print([m for m in dir(mgba.core.Core) if not m.startswith("_")])
print(inspect.signature(mgba.core.load_path))
EOF
```

Note the actual names for: setting keys (`set_keys`/`set_keys_raw`), memory
(`core.memory.u8[...]` or similar), savestates (`save_raw_state`/
`load_raw_state`), video (`set_video_buffer`, `mgba.image.Image`,
`desired_video_dimensions`). **If any name differs from the code in Task 3,
adapt the Task 3 code to the real API — the tests, not the plan text, are the
source of truth.**

- [ ] **Step 4: Write `tools/smoke_mgba.py`**

```python
"""Boot a ROM headless, run 300 frames, dump a screenshot. Proves the core works."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mgba.core
import mgba.image


def main() -> int:
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM to a .gba file")
        return 1
    core = mgba.core.load_path(rom)
    if core is None:
        print(f"mGBA could not load: {rom}")
        return 1
    width, height = core.desired_video_dimensions()
    image = mgba.image.Image(width, height)
    core.set_video_buffer(image)
    core.reset()
    for _ in range(300):
        core.run_frame()
    out = Path("smoke_frame.png")
    image.to_pil().convert("RGB").save(out)
    print(f"OK — {core.game_title if hasattr(core, 'game_title') else 'ROM'} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the smoke script**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> python tools/smoke_mgba.py
open smoke_frame.png
```

Expected: `smoke_frame.png` shows the GBA boot/intro screen (not black).
Delete the PNG afterwards (`rm smoke_frame.png`).

- [ ] **Step 6: Commit**

```bash
git add tools/smoke_mgba.py
git commit -m "feat: mgba bindings installed + headless smoke script"
```

---

### Task 3: Game-agnostic emulator wrapper

**Files:**
- Create: `emulator/buttons.py`
- Create: `emulator/gba.py`
- Test: `tests/conftest.py`, `tests/test_emulator.py`

- [ ] **Step 1: Write `emulator/buttons.py`**

```python
"""GBA key bitmask constants, matching the KEYINPUT register bit order."""
from __future__ import annotations

KEY_A = 1 << 0
KEY_B = 1 << 1
KEY_SELECT = 1 << 2
KEY_START = 1 << 3
KEY_RIGHT = 1 << 4
KEY_LEFT = 1 << 5
KEY_UP = 1 << 6
KEY_DOWN = 1 << 7
KEY_R = 1 << 8
KEY_L = 1 << 9
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 3: Write failing tests in `tests/test_emulator.py`**

```python
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
    a = emu.screenshot().copy()
    emu.step(frames=60)
    emu.load_state(state)
    b = emu.screenshot()
    np.testing.assert_array_equal(a, b)


def test_missing_rom_raises(tmp_path):
    import pytest

    from emulator.gba import GbaEmulator

    with pytest.raises(FileNotFoundError):
        GbaEmulator(tmp_path / "nope.gba")
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> pytest tests/test_emulator.py -v
```

Expected: FAIL / ERROR with `ModuleNotFoundError: emulator.gba`.

- [ ] **Step 5: Write `emulator/gba.py`**

Adapt method names to the real API discovered in Task 2 Step 3 if they differ.

```python
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
        self._core.set_keys_raw(keys)
        for _ in range(frames):
            self._core.run_frame()

    def read_bytes(self, address: int, length: int) -> bytes:
        return bytes(self._core.memory.u8[address : address + length])

    def save_state(self) -> bytes:
        return bytes(self._core.save_raw_state())

    def load_state(self, state: bytes) -> None:
        self._core.load_raw_state(state)

    def screenshot(self) -> np.ndarray:
        """Current frame as (160, 240, 3) uint8 RGB."""
        return np.asarray(self._video.to_pil().convert("RGB"), dtype=np.uint8)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> pytest tests/test_emulator.py -v && ruff check .
```

Expected: 4 passed (or 3 passed + 1 skipped without ROM); ruff clean.

- [ ] **Step 7: Update `docs/architecture/modules.md`**

The rows for `emulator/buttons.py` and `emulator/gba.py` already exist from
Task 1 — verify the public API column still matches (`GbaEmulator:
step/read_bytes/save_state/load_state/screenshot`). Fix if drifted.

- [ ] **Step 8: Commit**

```bash
git add emulator/ tests/
git commit -m "feat: game-agnostic headless mGBA wrapper with savestates"
```

---

### Task 4: Manual tools — scripted run and interactive play

**Files:**
- Create: `tools/run_scripted.py`
- Create: `tools/play_interactive.py`

No unit tests: these are thin manual tools; correctness is verified by eye
(M1 acceptance) and by the savestate they produce.

- [ ] **Step 1: Write `tools/run_scripted.py`**

```python
"""M1 acceptance: run a scripted input sequence, dump numbered screenshots."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

from emulator import buttons
from emulator.gba import GbaEmulator

# (keys, frames) pairs: mash through intro, then walk around
SCRIPT: list[tuple[int, int]] = [
    (0, 600),
    (buttons.KEY_A, 10), (0, 60),
    (buttons.KEY_A, 10), (0, 60),
    (buttons.KEY_START, 10), (0, 60),
    (buttons.KEY_A, 10), (0, 120),
    (buttons.KEY_DOWN, 32), (buttons.KEY_LEFT, 32),
    (buttons.KEY_UP, 32), (buttons.KEY_RIGHT, 32),
]


def main() -> int:
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM")
        return 1
    out_dir = Path("scripted_frames")
    out_dir.mkdir(exist_ok=True)
    emu = GbaEmulator(rom)
    for i, (keys, frames) in enumerate(SCRIPT):
        emu.step(keys, frames)
        Image.fromarray(emu.screenshot()).save(out_dir / f"{i:03d}.png")
    print(f"Wrote {len(SCRIPT)} frames to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and check the frames**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> python tools/run_scripted.py && open scripted_frames/
```

Expected: successive PNGs show intro screens advancing. Clean up with
`rm -rf scripted_frames/` (it is not gitignored on purpose — it should never
be committed, delete it).

- [ ] **Step 3: Write `tools/play_interactive.py`**

```python
"""Play the ROM with the keyboard; press S to save states/initial.state.

Keys: arrows = D-pad, X = A, Z = B, Enter = Start, Backspace = Select,
A = L, S(hift) keys per MAPPING below. Press F5 to save state, Esc to quit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pygame

from emulator import buttons
from emulator.gba import GbaEmulator

MAPPING = {
    pygame.K_UP: buttons.KEY_UP,
    pygame.K_DOWN: buttons.KEY_DOWN,
    pygame.K_LEFT: buttons.KEY_LEFT,
    pygame.K_RIGHT: buttons.KEY_RIGHT,
    pygame.K_x: buttons.KEY_A,
    pygame.K_z: buttons.KEY_B,
    pygame.K_RETURN: buttons.KEY_START,
    pygame.K_BACKSPACE: buttons.KEY_SELECT,
    pygame.K_a: buttons.KEY_L,
    pygame.K_s: buttons.KEY_R,
}
SCALE = 3
STATE_PATH = Path("states/initial.state")


def main() -> int:
    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        print("Set POKEMON_EMERALD_ROM")
        return 1
    emu = GbaEmulator(rom)
    pygame.init()
    screen = pygame.display.set_mode((240 * SCALE, 160 * SCALE))
    pygame.display.set_caption("Emerald — F5 saves states/initial.state")
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 0
                if event.key == pygame.K_F5:
                    STATE_PATH.parent.mkdir(exist_ok=True)
                    STATE_PATH.write_bytes(emu.save_state())
                    print(f"Saved {STATE_PATH}")
        pressed = pygame.key.get_pressed()
        keys = 0
        for pg_key, gba_key in MAPPING.items():
            if pressed[pg_key]:
                keys |= gba_key
        emu.step(keys, frames=1)
        frame = np.transpose(emu.screenshot(), (1, 0, 2))  # pygame wants (w, h)
        surface = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Play through the intro and create the initial savestate**

Ask the USER to do this (it is their ROM and their save):

```bash
POKEMON_EMERALD_ROM=<user-rom-path> python tools/play_interactive.py
```

Play until the character is standing in the truck/bedroom after the intro
cutscene, then press F5. Expected: `states/initial.state` exists.

- [ ] **Step 5: Update `docs/architecture/modules.md`**

Verify the `tools/` rows match reality (they were pre-filled in Task 1).

- [ ] **Step 6: Commit**

```bash
git add tools/run_scripted.py tools/play_interactive.py
git commit -m "feat: scripted-run and interactive-play tools (M1 acceptance)"
```

---

### Task 5: Emerald RAM readers

**Files:**
- Create: `env/game_state.py`
- Test: `tests/test_game_state.py`

Addresses below come from the pret/pokeemerald decompilation (Emerald US,
BPEE). **The target ROM is French (BPEF)** — before implementing, look up
`gSaveBlock1Ptr` and `gPlayerPartyCount` in pokebot-gen3's BPEF symbol table
(https://github.com/40Cakes/pokebot-gen3, `modules/data/symbols/`) and use
those values for `SAVE_BLOCK1_PTR` / `PARTY_COUNT_ADDR`. Struct offsets
(_POS_OFFSET, _LOCATION_OFFSET, _FLAGS_OFFSET, badge flag IDs) are identical
across languages. The real-ROM integration test (Step 5) is the ultimate
check: wrong addresses ⇒ absurd party_count/coords.

Design: parsing is pure — `EmeraldReader` takes a `read(address, length) ->
bytes` callable. Unit tests inject a fake reader; no ROM needed.

- [ ] **Step 1: Write failing tests in `tests/test_game_state.py`**

```python
from __future__ import annotations

import pytest

from env.game_state import (
    PARTY_COUNT_ADDR,
    SAVE_BLOCK1_PTR,
    EmeraldReader,
    PlayerState,
)


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
    *, x: int, y: int, map_group: int, map_num: int, badge_bits: int, party_count: int
) -> dict[int, bytes]:
    sb1 = 0x02025A00  # arbitrary but valid EWRAM address for the fake
    save_block1 = bytearray(0x1290)
    save_block1[0:2] = x.to_bytes(2, "little")
    save_block1[2:4] = y.to_bytes(2, "little")
    save_block1[4] = map_group
    save_block1[5] = map_num
    # Badge flags start at flag 0x867 -> byte 0x10C bit 7 of the flags array
    flags_value = badge_bits << 7
    save_block1[0x1270 + 0x10C : 0x1270 + 0x10E] = flags_value.to_bytes(2, "little")
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_game_state.py -v
```

Expected: FAIL with `ModuleNotFoundError: env.game_state`.

- [ ] **Step 3: Write `env/game_state.py`**

```python
"""Typed readers for Pokémon Emerald (US, BPEE) RAM.

Emerald relocates its save blocks (anti-cheat DMA), so all SaveBlock1 fields
are reached through the pointer at SAVE_BLOCK1_PTR. Addresses cross-checked
against pret/pokeemerald and pokebot-gen3.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

ReadFn = Callable[[int, int], bytes]

SAVE_BLOCK1_PTR = 0x03005D8C
PARTY_COUNT_ADDR = 0x020244E9

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
        sb1 = int.from_bytes(self._read(SAVE_BLOCK1_PTR, 4), "little")
        if not _EWRAM_START <= sb1 < _EWRAM_END:
            return None
        pos = self._read(sb1 + _POS_OFFSET, 4)
        location = self._read(sb1 + _LOCATION_OFFSET, 2)
        return PlayerState(
            x=int.from_bytes(pos[0:2], "little"),
            y=int.from_bytes(pos[2:4], "little"),
            map_group=location[0],
            map_num=location[1],
            badges=self._badge_count(sb1),
            party_count=self._read(PARTY_COUNT_ADDR, 1)[0],
        )

    def _badge_count(self, sb1: int) -> int:
        byte_index, bit_index = divmod(_FIRST_BADGE_FLAG, 8)
        raw = int.from_bytes(self._read(sb1 + _FLAGS_OFFSET + byte_index, 2), "little")
        return ((raw >> bit_index) & 0xFF).bit_count()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_game_state.py -v && ruff check .
```

Expected: 4 passed; ruff clean.

- [ ] **Step 5: Add a real-ROM integration test**

Append to `tests/test_game_state.py`:

```python
from tests.conftest import requires_rom  # noqa: E402  (top of file in practice)


@requires_rom
def test_real_rom_state_after_initial_savestate(rom_path):
    """Sanity check against the real game: state is readable post-intro."""
    from pathlib import Path

    from emulator.gba import GbaEmulator

    state_file = Path("states/initial.state")
    if not state_file.is_file():
        import pytest

        pytest.skip("states/initial.state not created yet (Task 4)")
    emu = GbaEmulator(rom_path)
    emu.load_state(state_file.read_bytes())
    emu.step(frames=10)
    state = EmeraldReader(emu.read_bytes).player_state()
    assert state is not None
    assert state.party_count <= 6
```

Put the import at the top of the file with the others.

- [ ] **Step 6: Run the full test file**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> pytest tests/test_game_state.py -v
```

Expected: all pass (integration test skips if the savestate does not exist yet).

- [ ] **Step 7: Commit**

```bash
git add env/game_state.py tests/test_game_state.py
git commit -m "feat: emerald RAM readers (position, map, badges, party)"
```

---

### Task 6: Exploration reward

**Files:**
- Create: `env/rewards.py`
- Test: `tests/test_rewards.py`

- [ ] **Step 1: Write failing tests in `tests/test_rewards.py`**

```python
from __future__ import annotations

from env.game_state import PlayerState
from env.rewards import ExplorationTracker


def state(x: int, y: int, group: int = 0, num: int = 0) -> PlayerState:
    return PlayerState(x=x, y=y, map_group=group, map_num=num, badges=0, party_count=0)


def test_new_tile_rewards_once():
    tracker = ExplorationTracker()
    assert tracker.update(state(1, 1)) == 1.0
    assert tracker.update(state(1, 1)) == 0.0
    assert tracker.update(state(2, 1)) == 1.0


def test_same_coords_on_different_map_are_distinct():
    tracker = ExplorationTracker()
    assert tracker.update(state(1, 1, group=0, num=0)) == 1.0
    assert tracker.update(state(1, 1, group=0, num=1)) == 1.0


def test_none_state_gives_zero():
    tracker = ExplorationTracker()
    assert tracker.update(None) == 0.0


def test_visited_count():
    tracker = ExplorationTracker()
    tracker.update(state(1, 1))
    tracker.update(state(2, 1))
    tracker.update(state(2, 1))
    assert tracker.visited_count == 2


def test_reset_clears_history():
    tracker = ExplorationTracker()
    tracker.update(state(1, 1))
    tracker.reset()
    assert tracker.visited_count == 0
    assert tracker.update(state(1, 1)) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rewards.py -v
```

Expected: FAIL with `ModuleNotFoundError: env.rewards`.

- [ ] **Step 3: Write `env/rewards.py`**

```python
"""Reward shaping. v1: reward discovery of never-visited tiles (map-qualified)."""
from __future__ import annotations

from env.game_state import PlayerState


class ExplorationTracker:
    """Gives +1.0 the first time each (map_group, map_num, x, y) tile is seen."""

    def __init__(self) -> None:
        self._visited: set[tuple[int, int, int, int]] = set()

    @property
    def visited_count(self) -> int:
        return len(self._visited)

    def reset(self) -> None:
        self._visited.clear()

    def update(self, state: PlayerState | None) -> float:
        if state is None:
            return 0.0
        tile = (state.map_group, state.map_num, state.x, state.y)
        if tile in self._visited:
            return 0.0
        self._visited.add(tile)
        return 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rewards.py -v && ruff check .
```

Expected: 5 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add env/rewards.py tests/test_rewards.py
git commit -m "feat: exploration reward tracker"
```

---

### Task 7: Gymnasium environment

**Files:**
- Create: `env/pokemon_env.py`
- Modify: `tests/conftest.py` (add FakeEmulator)
- Test: `tests/test_env.py`

- [ ] **Step 1: Add `FakeEmulator` to `tests/conftest.py`**

Append:

```python
import numpy as np


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
```

- [ ] **Step 2: Write failing tests in `tests/test_env.py`**

```python
from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.pokemon_env import OBS_SHAPE, PokemonEmeraldEnv
from tests.conftest import FakeEmulator


def make_env(max_steps: int = 50) -> PokemonEmeraldEnv:
    return PokemonEmeraldEnv(FakeEmulator(), initial_state=b"fake", max_steps=max_steps)


def test_gymnasium_api_compliance():
    check_env(make_env(), skip_render_check=True)


def test_reset_loads_initial_state_and_returns_obs():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.emulator.loaded_states == [b"fake"]
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.uint8


def test_moving_to_new_tile_gives_positive_reward():
    env = make_env()
    env.reset(seed=0)
    right = env.ACTIONS.index("right")
    _, reward, _, _, _ = env.step(right)
    assert reward > 0.0


def test_staying_put_gives_zero_reward_after_first_visit():
    env = make_env()
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    env.step(noop)
    _, reward, _, _, _ = env.step(noop)
    assert reward == 0.0


def test_truncates_at_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    noop = env.ACTIONS.index("noop")
    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(noop)
    assert truncated
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_env.py -v
```

Expected: FAIL with `ModuleNotFoundError: env.pokemon_env`.

- [ ] **Step 4: Write `env/pokemon_env.py`**

```python
"""Gymnasium environment for Pokémon Emerald over a (real or fake) GBA emulator."""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from emulator import buttons
from env.game_state import EmeraldReader
from env.rewards import ExplorationTracker

FRAME_SIZE = (84, 84)  # (width, height) after downscale
FRAME_STACK = 3
OBS_SHAPE = (FRAME_SIZE[1], FRAME_SIZE[0], FRAME_STACK)
FRAMES_PER_ACTION = 24  # ~0.4 s of game time; one walking step per action

_ACTION_KEYS: dict[str, int] = {
    "noop": 0,
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
    "a": buttons.KEY_A,
    "b": buttons.KEY_B,
    "start": buttons.KEY_START,
}


class PokemonEmeraldEnv(gym.Env):
    """Pixels in, exploration reward out. Episodes start from a fixed savestate."""

    metadata = {"render_modes": ["rgb_array"]}
    ACTIONS = list(_ACTION_KEYS)

    def __init__(
        self,
        emulator: Any,
        initial_state: bytes,
        max_steps: int = 2048,
    ) -> None:
        super().__init__()
        self.emulator = emulator
        self._initial_state = initial_state
        self._max_steps = max_steps
        self._reader = EmeraldReader(emulator.read_bytes)
        self._tracker = ExplorationTracker()
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)
        self._step_count = 0
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.emulator.load_state(self._initial_state)
        self._tracker.reset()
        self._frames.clear()
        self._step_count = 0
        frame = self._current_frame()
        for _ in range(FRAME_STACK):
            self._frames.append(frame)
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        keys = _ACTION_KEYS[self.ACTIONS[action]]
        self.emulator.step(keys, FRAMES_PER_ACTION)
        self._frames.append(self._current_frame())
        self._step_count += 1
        reward = self._tracker.update(self._reader.player_state())
        truncated = self._step_count >= self._max_steps
        return self._observation(), reward, False, truncated, self._info()

    def render(self) -> np.ndarray:
        return self.emulator.screenshot()

    def _current_frame(self) -> np.ndarray:
        image = Image.fromarray(self.emulator.screenshot()).convert("L").resize(FRAME_SIZE)
        return np.asarray(image, dtype=np.uint8)

    def _observation(self) -> np.ndarray:
        return np.stack(self._frames, axis=-1)

    def _info(self) -> dict[str, Any]:
        state = self._reader.player_state()
        return {
            "visited_tiles": self._tracker.visited_count,
            "badges": state.badges if state else 0,
            "map": (state.map_group, state.map_num) if state else None,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_env.py -v && ruff check .
```

Expected: 5 passed; ruff clean.

- [ ] **Step 6: Run the whole suite**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> pytest -v
```

Expected: everything passes (ROM tests skip without the env var).

- [ ] **Step 7: Commit**

```bash
git add env/pokemon_env.py tests/conftest.py tests/test_env.py
git commit -m "feat: gymnasium env with pixel obs and exploration reward"
```

---

### Task 8: PPO training entrypoint

**Files:**
- Create: `agent/train.py`
- Test: `tests/test_train_smoke.py`

- [ ] **Step 1: Write the failing smoke test in `tests/test_train_smoke.py`**

```python
from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.pokemon_env import PokemonEmeraldEnv
from tests.conftest import FakeEmulator


def test_ppo_learns_without_crashing():
    """256 steps of PPO on the fake emulator: wiring works end to end."""
    vec = DummyVecEnv(
        [lambda: PokemonEmeraldEnv(FakeEmulator(), initial_state=b"fake", max_steps=64)]
    )
    model = PPO("CnnPolicy", vec, n_steps=64, batch_size=64, device="cpu", verbose=0)
    model.learn(total_timesteps=256)
```

- [ ] **Step 2: Run it (it should pass already — it exercises Task 7 output)**

```bash
pytest tests/test_train_smoke.py -v
```

Expected: PASS in under ~2 minutes. If it fails, the env/sb3 wiring is broken:
fix before writing train.py.

- [ ] **Step 3: Write `agent/train.py`**

```python
"""Train PPO on Pokémon Emerald. Requires POKEMON_EMERALD_ROM and states/initial.state."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from emulator.gba import GbaEmulator
from env.pokemon_env import PokemonEmeraldEnv

log = logging.getLogger("agent.train")

STATE_PATH = Path("states/initial.state")


def make_env(rom_path: str, initial_state: bytes, max_steps: int):
    def _init() -> PokemonEmeraldEnv:
        return PokemonEmeraldEnv(GbaEmulator(rom_path), initial_state, max_steps=max_steps)

    return _init


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint .zip to resume")
    args = parser.parse_args()

    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        log.error("Set POKEMON_EMERALD_ROM")
        return 1
    if not STATE_PATH.is_file():
        log.error("Missing %s — create it with tools/play_interactive.py", STATE_PATH)
        return 1

    initial_state = STATE_PATH.read_bytes()
    vec = SubprocVecEnv(
        [make_env(rom, initial_state, args.max_steps) for _ in range(args.envs)]
    )
    device = pick_device()
    log.info("Training on device=%s with %d envs", device, args.envs)

    if args.resume:
        model = PPO.load(args.resume, env=vec, device=device)
    else:
        model = PPO(
            "CnnPolicy",
            vec,
            n_steps=512,
            batch_size=512,
            ent_coef=0.01,
            learning_rate=3e-4,
            device=device,
            verbose=1,
            tensorboard_log="runs",
        )
    checkpoints = CheckpointCallback(
        save_freq=max(50_000 // args.envs, 1), save_path="checkpoints", name_prefix="ppo_emerald"
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoints, reset_num_timesteps=False)
    model.save("checkpoints/ppo_emerald_final")
    return 0


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run ruff and the full suite**

```bash
ruff check . && POKEMON_EMERALD_ROM=<user-rom-path> pytest -v
```

Expected: clean, all pass/skip as designed.

- [ ] **Step 5: Launch a short real training run (M4 acceptance)**

```bash
POKEMON_EMERALD_ROM=<user-rom-path> python agent/train.py --envs 4 --timesteps 20000
tensorboard --logdir runs/
```

Expected: training runs without crashing; TensorBoard shows
`rollout/ep_rew_mean` > 0 (the agent discovers tiles). This is the M4
acceptance check — a rising curve is a bonus at 20k steps, not required.

- [ ] **Step 6: Update `docs/architecture/modules.md`**

Verify the `agent/train.py` row (public API: `main()`, deps: env, sb3, torch).

- [ ] **Step 7: Commit**

```bash
git add agent/train.py tests/test_train_smoke.py
git commit -m "feat: PPO training entrypoint with MPS and parallel envs"
```

---

### Task 9: Final verification and docs

**Files:**
- Modify: `README.md` (only if commands drifted from reality)
- Modify: `docs/architecture/modules.md` (only if drifted)

- [ ] **Step 1: Full suite + lint from a clean shell**

```bash
source .venv/bin/activate
ruff check . && POKEMON_EMERALD_ROM=<user-rom-path> pytest -v
```

Expected: 0 failures; ROM tests ran (not skipped).

- [ ] **Step 2: Verify README quickstart is truthful**

Follow README commands top to bottom as written. Fix any drift.

- [ ] **Step 3: Commit (only if fixes were needed)**

```bash
git add README.md docs/architecture/modules.md
git commit -m "docs: align quickstart and module index with implementation"
```

---

## Out of scope (next plan)

- M5: milestone rewards (event flags: starter obtained, rival beaten, badges)
- M6: reward/hyperparameter iteration toward badge 1
- `tools/watch.py` (watch a trained agent play live)
- Battle shaping (v3 rewards)
