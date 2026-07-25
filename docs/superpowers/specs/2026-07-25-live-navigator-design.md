# P3 (step 1) — Live Intra-Map Navigator — Design

**Date:** 2026-07-25
**Palier:** P3, first increment (the smallest live loop against the real emulator)
**Status:** approved, ready for planning

## Goal

Drive the **real** GBA emulator so the player character walks from wherever it
is to a target cell `(x, y)` **on the current map**, learning walls by collision
and replanning when it bumps into one. This is the first time the Explorer's
brain (P1 perception + P2 pathfinding) actually presses buttons and moves in the
running game.

One sentence: **wire P1 `snapshot()` + P2 `plan_path()`/`WallMap` to the live
emulator, in a perceive → plan → press → classify → learn → replan loop.**

## Why this increment first

P2 is pure math (given a wall grid, compute a path). Nothing presses buttons.
This step is the smallest thing we can *watch work on screen*: "walk to that
cell without getting stuck." It needs neither the Strategist, nor the Fighter,
nor a populated multi-map graph. Once it works, later P3 increments (crossing
maps, self-mapping, Strategist orders) build on top of it.

## Architecture

### New file: `env/live_navigator.py`

A single control function plus small constants. Pure control glue over injected
dependencies — no Gym env, no training, no reward. It reuses, unchanged:

- **P1** `WorldReader.snapshot() -> WorldSnapshot | None` — where am I now
  (`map_id`, `pos`). Returns `None` during SaveBlock relocation.
- **P2** `plan_path(wallmap, map_id, start, goal) -> list[str] | None` — the
  direction sequence, or `None` if unreachable given known walls.
- **P2** `resolve_move(before, after) -> "moved" | "blocked" | "transition"`.
- **P2** `WallMap.block(map_id, cell, direction)` — record a discovered wall.
- **Emulator** `emulator.step(keys, frames)` with `buttons.KEY_UP/DOWN/LEFT/RIGHT`.

### The public function

```python
def navigate_to(
    emulator,          # anything with .step(keys, frames)
    reader,            # anything with .snapshot() -> WorldSnapshot | None
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
) -> str:
    """Walk the player to `target` on its current map. Returns an outcome string:
    'arrived' | 'unreachable' | 'left_map' | 'timeout'."""
```

### Direction → key map (module constant)

```python
_DIRECTION_KEYS: dict[str, int] = {
    "up": buttons.KEY_UP,
    "down": buttons.KEY_DOWN,
    "left": buttons.KEY_LEFT,
    "right": buttons.KEY_RIGHT,
}
STEP_FRAMES = 24     # hold a d-pad key ~0.4 s = one walking tile (matches env FRAMES_PER_ACTION)
RELEASE_FRAMES = 8   # idle after each press so the GBA doesn't fuse consecutive presses
TURN_RETRIES = 2     # see "turn vs wall" below
```

## Data flow (the live loop)

```
repeat up to max_steps times:
    before = reader.snapshot()
    if before is None:            # SaveBlock relocating this frame
        emulator.step(0, STEP_FRAMES)   # idle one beat, retry
        continue
    if before.pos == target:
        return "arrived"
    path = plan_path(wallmap, before.map_id, before.pos, target)
    if path is None:
        return "unreachable"
    direction = path[0]           # first hop; we replan every iteration (replan-on-bump)
    outcome = _press_until_moved(emulator, reader, before, direction)
    if outcome == "transition":
        return "left_map"         # walked off the map — later P3 handles inter-map
    if outcome == "blocked":
        wallmap.block(before.map_id, before.pos, direction)   # confirmed wall
    # "moved": loop, re-snapshot, keep going
return "timeout"
```

We deliberately **replan from scratch each iteration** rather than executing the
whole `path` blindly: the moment a bump reveals a wall the plan assumed open, the
next `plan_path` routes around it. This is the "replan-on-bump" from P2.

## The one real subtlety: turn vs wall

In Emerald, pressing a d-pad direction while facing a *different* direction makes
the character **turn in place on the first press without moving**. A naive
`resolve_move` would read that first no-move as a wall.

**Resolution:** `_press_until_moved` presses the same direction up to
`TURN_RETRIES` times. A genuine turn resolves into a move on the second press
(now facing the right way); a genuine wall stays `blocked` on every retry. So:

```python
def _press_until_moved(emulator, reader, before, direction) -> str:
    outcome = "blocked"
    for _ in range(TURN_RETRIES):
        emulator.step(_DIRECTION_KEYS[direction], STEP_FRAMES)
        emulator.step(0, RELEASE_FRAMES)     # release so consecutive presses don't fuse
        after = _snapshot_settled(reader)    # skip None frames
        outcome = resolve_move(before, after)
        if outcome != "blocked":
            return outcome                   # moved or transition -> done
    return outcome                           # still blocked after retries -> real wall
```

`RELEASE_FRAMES` (a short idle, e.g. 8) matters: two held presses with no release
between them get fused by the GBA into one press (the Fighter hit this exact
bug). `_snapshot_settled` re-reads `snapshot()`, skipping up to a few `None`
frames during relocation.

The exact `STEP_FRAMES` / `TURN_RETRIES` / `RELEASE_FRAMES` values are tuned
during implementation against the real ROM (the ROM smoke test is where we
confirm one press = one clean tile).

## Testing strategy

### Unit tests (no ROM) — `tests/test_live_navigator.py`

A `FakeWorld` simulates a small grid with hidden walls the navigator must
discover. It plays **both** roles the navigator depends on:

- as the *emulator*: `.step(keys, frames)` decodes the d-pad bit and moves its
  hidden position one tile — unless a hidden wall blocks it (then position is
  unchanged, i.e. a real "blocked"), optionally modelling a first-press "turn"
  (no move) to exercise `TURN_RETRIES`.
- as the *reader*: `.snapshot()` returns a `WorldSnapshot` of its current
  `map_id`/`pos`.

Cases:
1. Straight corridor, no walls → arrives, presses count == distance.
2. A hidden wall on the direct path → bumps once, records it in `WallMap`,
   replans around it, arrives.
3. Goal fully walled off from an open start → returns `"unreachable"` (relies on
   P2's bounded A*), within `max_steps`.
4. First press only turns (no move), second moves → still arrives, does **not**
   record a false wall.
5. Stepping onto a cell that flips `map_id` → returns `"left_map"`.
6. Never-arriving world capped by `max_steps` → returns `"timeout"`.
7. `snapshot()` returns `None` for a frame or two, then a real snapshot → the
   loop tolerates it and still makes progress.

### ROM smoke test (gated) — `tests/test_live_navigator_rom.py`

Marked to run only when `POKEMON_EMERALD_ROM` is set (same gating as existing ROM
tests). Load `states/initial.state` (or a clearer open-map state), pick a target
a few tiles away on the same map, run `navigate_to`, and assert the final
`snapshot().pos == target` (or `"arrived"`). This is the "watch it actually
walk" check; it also validates the frame-timing constants.

## Non-goals (deferred to later P3 increments)

- Crossing between maps (following a `plan_route` chain) — this returns
  `"left_map"` and stops.
- Self-mapping mode (wandering to populate `MapMemory`).
- Strategist order interface and moving the story out of `PokemonEmeraldEnv`.
- The full hierarchical loop (Strategist dispatching navigator + Fighter).
- Tile semantics (grass/water/Surf) — still awaits the `tile_behavior` probe.

## Files

- Create: `env/live_navigator.py`
- Create: `tests/test_live_navigator.py`
- Create: `tests/test_live_navigator_rom.py`
- Reuse unchanged: `env/world_reader.py`, `env/local_navigator.py`, `emulator/gba.py`, `emulator/buttons.py`
