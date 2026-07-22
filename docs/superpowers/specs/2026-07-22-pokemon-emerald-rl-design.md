# Pokémon Emerald RL Agent — Design

Date: 2026-07-22
Status: Approved (pending spec review)

## Goal

Build a deep learning agent that plays Pokémon Emerald (GBA) with the long-term
objective of finishing the game. The emulator is a means, not the project: we
use the proven mGBA core through its Python bindings rather than writing an
emulator from scratch.

## Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Emulator | mGBA via `libmgba-py` | Battle-tested core, headless, frame-by-frame control, direct RAM access, savestates. Faster-than-realtime emulation for training. |
| Game | Pokémon Emerald (user-provided ROM) | User choice. Fully decompiled (pret/pokeemerald) and RAM-mapped by the community (pokebot-gen3 project). |
| Observations | Screen pixels (downscaled, stacked frames) | Agent "sees" the game like a human. |
| Reward | Computed from RAM | Exploration (new tiles/maps), then event flags, badges, party progress. Only realistic way to measure progress in Pokémon. |
| DL stack | Python + PyTorch, PPO via stable-baselines3, Gymnasium API | Standard RL ecosystem, MPS support on Apple Silicon (M4 Pro, 24 GB). |

Rejected alternatives: writing a GBA emulator from scratch (months of debugging
before any RL work), stable-retro (fragile GBA/macOS arm64 support, rigid
`data.json` integration), Lua-socket bridge to mGBA GUI (IPC too slow for
training).

## Architecture

```
Emu/
├── emulator/     # Thin wrapper over libmgba-py
│                 #   boot(rom), step(buttons), read_ram(addr, len),
│                 #   save_state()/load_state(), screenshot()
├── env/          # PokemonEmeraldEnv (Gymnasium)
│                 #   obs: downscaled grayscale frames, frame stack
│                 #   action space: discrete buttons (D-pad, A, B, Start, ...)
│                 #   reward: RAM-derived (exploration, milestones)
│                 #   game_state.py: typed readers for Emerald RAM
│                 #     (player position, map ID, party, badges, event flags)
├── agent/        # PPO training (stable-baselines3), MPS device config,
│                 #   checkpoints, TensorBoard logs, parallel envs
└── tools/        # Live viewer (watch the agent play), RAM inspector,
                  #   savestate manager, replay
```

Component boundaries:
- `emulator/` knows nothing about Pokémon. Reusable for any GBA ROM.
- `env/` knows Pokémon Emerald (RAM addresses, rewards) but nothing about the
  learning algorithm.
- `agent/` consumes the Gymnasium API only. Swappable algorithm.

## Data flow

```
agent (PPO) --action--> env.step() --buttons--> emulator (mGBA headless)
emulator --frame + RAM--> env --obs (pixels) + reward (RAM-derived)--> agent
```

Training uses N parallel headless emulator instances (SubprocVecEnv). Episodes
start from savestates (e.g., "post-intro, in bedroom") to skip unskippable
cutscenes and focus learning.

## Reward design (staged)

1. **v1 — Exploration:** reward for each newly visited (map, tile) coordinate.
   Proven to get agents out of the starting town (Pokémon Red RL approach).
2. **v2 — Milestones:** event flags from RAM — got starter, beat rival, badge
   obtained, new town reached. Large one-time rewards.
3. **v3 — Battle shaping:** party XP/levels gained, opponent HP reduced,
   penalty on blackout.

"Finishing the game" is the long-term target; success is measured by the
furthest milestone reached. This is research-grade difficulty — milestones are
the honest metric.

## Milestones (build order)

- **M1:** libmgba-py builds on macOS; boot ROM headless; scripted inputs;
  screenshot to PNG. Verification: script walks the player out of a room.
- **M2:** Minimal Gymnasium env (`reset`/`step`/`render`), random agent runs
  without crash at > realtime speed.
- **M3:** RAM readers for Emerald (position, map ID, badges, party) with tests
  against known savestates. Reuse pokebot-gen3's documented addresses.
- **M4:** Exploration reward + first PPO training run on MPS; TensorBoard
  shows increasing explored-tile count.
- **M5:** Milestone rewards; agent reliably gets starter and leaves Littleroot.
- **M6+:** Iterate on rewards/hyperparameters toward badge 1 and beyond.

## Error handling

- Emulator wrapper validates ROM path and detects bad ROM/BIOS at boot.
- Env guards against RAM reads returning garbage during map transitions
  (retry/latch last known-good values).
- Training: periodic checkpoints; runs resumable after crash.

## Testing

- `emulator/`: unit tests with a homebrew/test ROM (no commercial ROM in repo);
  savestate round-trip, deterministic stepping.
- `env/`: RAM reader tests against committed savestates (not the ROM);
  Gymnasium API compliance (`check_env`).
- `agent/`: smoke test — 1k steps of PPO on 1 env completes without error.
- The user's ROM is never committed (gitignored).

## Constraints

- macOS Apple Silicon (M4 Pro, 24 GB) — PyTorch MPS, no CUDA.
- Python ≥ 3.12, ruff, pytest (per user global standards).
- User provides their own legally dumped Emerald ROM.
