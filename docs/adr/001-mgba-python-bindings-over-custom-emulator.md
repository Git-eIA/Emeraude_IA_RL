# ADR-001: mGBA Python bindings instead of a custom emulator

Date: 2026-07-22
Status: Accepted

## Context

The project goal is a deep learning agent that plays Pokémon Emerald (GBA) to
completion. An emulator is required. Options considered:

1. Write a GBA emulator from scratch (ARM7TDMI CPU, PPU, DMA, timers...)
2. mGBA core via `libmgba-py` Python bindings + custom Gymnasium env
3. stable-retro (gym-retro fork) with a custom Emerald integration
4. mGBA GUI + Lua scripting + socket bridge to Python

## Decision

Option 2: mGBA via `libmgba-py`, wrapped in a custom Gymnasium environment.

## Rationale

- The emulator is a means, not the goal; debugging a custom emulator to the
  point of running Emerald end-to-end would delay RL work by months.
- libmgba-py gives frame-by-frame headless control, direct RAM access, and
  savestates — exactly the primitives RL training needs, faster than realtime.
- The pokebot-gen3 project proves this stack works for Emerald on this core
  and documents the relevant RAM layout.
- stable-retro has fragile GBA/macOS arm64 support and a rigid integration
  format; a Lua/socket bridge is too slow for training throughput.

## Consequences

- We depend on building libmgba on macOS (documented, but a native build step).
- The Gymnasium interface is ours to write and maintain (`env/`), which is
  also where the project's interesting logic lives.
- `emulator/` stays game-agnostic, so a from-scratch emulator could later be
  swapped in behind the same interface if desired.
