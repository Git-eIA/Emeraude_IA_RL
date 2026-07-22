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
