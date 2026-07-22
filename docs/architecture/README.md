# Architecture

Three decoupled layers (see ADR-001 and the design spec):

- `emulator/` — game-agnostic mGBA wrapper (libmgba-py). No Pokémon knowledge.
- `env/` — Pokémon Emerald Gymnasium env: pixel observations, RAM-derived rewards.
- `agent/` — PPO training (stable-baselines3, MPS). Consumes the Gymnasium API only.

## libmgba-py install

**Requires Python 3.12** (the prebuilt `.abi3.so` is not compatible with 3.14+).
Recreate the venv if needed: `rm -rf .venv && python3.12 -m venv .venv && pip install -e ".[dev]"`.

libmgba-py is not on PyPI. Steps actually used (macOS arm64):

1. Install the native library: `brew install mgba`
   This provides `/opt/homebrew/lib/libmgba.0.10.dylib` which the `.so` links against.

2. Download the prebuilt package from https://github.com/hanzi/libmgba-py/releases
   (tag `0.2.0-2`, asset `libmgba-py_0.2.0_macos-arm64.zip`) and unzip it
   directly into the venv site-packages:
   ```
   gh release download "0.2.0-2" -R hanzi/libmgba-py \
     --pattern "libmgba-py_0.2.0_macos-arm64.zip" -D /tmp/
   unzip /tmp/libmgba-py_0.2.0_macos-arm64.zip \
     -d "$(python -c 'import site; print(site.getsitepackages()[0])')"
   ```
   Note: this is NOT a wheel — it is a raw Python package with a stable-ABI `.so`.

3. Install cffi (runtime dependency): `pip install cffi`

Verify with: `python -c "import mgba.core; print('ok')"`

Module index: see `modules.md`.
