# Interruptible Navigation (auto-battle during travel) — Design Spec

**Status:** draft (2026-08-02)

**Palier:** P4 — Brique 1 of 3 toward "beat the rival on route_103 autonomously"
(Brique 2 = live survey+travel to route_103; Brique 3 = rival trainer battle + campaign milestone).

## Goal

Make `navigate_to` (the shared intra-map movement primitive) survive a wild battle
that starts mid-move: detect the battle, hand it to the trained Fighter via
`play_battle`, win, and resume walking. `travel_to` — and therefore the `advance`
mode of `execute_order` — inherit this for free. This is the prerequisite for any
real overworld traversal, because route_101/route_103 are full of tall grass and a
wild encounter today derails navigation into a `timeout`.

## The bug being fixed

In `env/live_navigator.py::navigate_to`, when a wild battle starts en route:
`enc_watcher.observe(reader.in_battle())` records the cell as `has_grass`, but the
loop then calls `probe_step`, which presses the d-pad while the character is frozen
in battle. `resolve_move` sees no movement → returns `"blocked"` → `wallmap.block`
records a **false wall** → the loop spins (still `in_battle`, cannot move) until
`max_steps` is exhausted → `"timeout"`. Navigation through grass is impossible.

## Architecture

Single-primitive fix. The auto-battle handling lives at the top of `navigate_to`'s
loop, factored into one small helper so it stays under 40 lines (code-safety #4)
and is unit-testable in isolation. The Fighter is injected (`move_type_fn`,
`predict`) — the exact dependency-injection pattern already used by `_execute_grind`
in `env/orders.py` — so `live_navigator` stays free of Stable-Baselines3/torch and
testable without a policy.

### 1. `env/live_navigator.py`

New helper:

```
def _handle_battle_interruption(emulator, reader, move_type_fn, predict) -> str | None:
    """If a wild battle is in progress, hand it to the Fighter and report.

    Returns None when there is no battle (or the battle was won) so the caller
    resumes navigating; returns a terminal outcome string when navigation must
    abort: "battle_lost" | "battle_timeout" (Fighter could not win) or
    "battle_interrupted" (a battle is up but no Fighter was supplied).
    """
```

- Not in battle → `None`.
- In battle, no Fighter (`move_type_fn is None or predict is None`) → `"battle_interrupted"`.
- In battle, Fighter present → `result = play_battle(emulator, move_type_fn, predict)`;
  `"won"` → `None`; `"lost"` → `"battle_lost"`; `"battle_timeout"` → `"battle_timeout"`.

`navigate_to` gains two optional params `move_type_fn: Any = None, predict: Any = None`.
The loop order is: read `before`; if non-None run the existing `has_grass`/heal
recording block (`if memory is not None`); THEN the interruption check; THEN the
target/plan/probe logic:

```
if memory is not None:
    if heal_watcher.observe(reader.party_hp()): memory.observe(before, WorldEvent(healed=True))
    if enc_watcher.observe(reader.in_battle()): memory.observe(before, WorldEvent(encounter_started=True))
interruption = _handle_battle_interruption(emulator, reader, move_type_fn, predict)
if interruption is not None:
    return interruption
if before.pos == target:
    return "arrived"
...
```

**Ordering matters:** the recording block must run BEFORE the interruption handler.
`enc_watcher` fires `has_grass` only on the not-in-battle → in-battle front; if the
handler fought (and cleared) the battle first, `reader.in_battle()` would already
read False and the grass cell would never be learned. Recording first preserves the
learning; fighting second resumes the walk (after a won battle the same iteration
falls through to plan/probe with the character now free to move).

New `navigate_to` return values (added to the existing
`'arrived' | 'unreachable' | 'left_map' | 'timeout'`):
`"battle_lost" | "battle_timeout" | "battle_interrupted"`.

### 2. `env/map_traveler.py`

`travel_to` gains `move_type_fn: Any = None, predict: Any = None` and passes them to
**all three** `navigate_to` calls (goal-cell arrival line 40, portal-cell approach
line 50, door-crossing line 60).

`travel_to` must propagate the battle outcomes rather than mishandle them. Define:

```
BATTLE_OUTCOMES = ("battle_lost", "battle_timeout", "battle_interrupted")
```

After each `navigate_to` call, if its result is in `BATTLE_OUTCOMES`, return it
immediately. This is why `"battle_lost"` is used instead of `"lost"`: `travel_to`
already returns `"lost"` to mean "crossed a door but landed on the wrong map"
(line 66) — reusing `"lost"` would conflate a Fighter loss with a routing error.

### 3. `env/orders.py`

`execute_order`'s `advance` path (the final `travel_to` call) passes
`move_type_fn=move_type_fn, predict=predict` (both already parameters of
`execute_order`). No other mode changes. The docstring's returns list is extended
with the three battle outcomes for `advance`.

## Outcome vocabulary (final)

| Outcome | Meaning |
|---|---|
| `battle_lost` | A wild battle interrupted navigation and the Fighter lost. |
| `battle_timeout` | A wild battle interrupted navigation and the Fighter hit its turn cap. |
| `battle_interrupted` | A wild battle is in progress but no Fighter was supplied. |

All three propagate verbatim up through `travel_to` and `execute_order`, so a future
Strategist/campaign driver sees the specific failure and can react (heal, retry,
abort).

## Scope / non-goals

- **`map_map` (survey) is NOT wired here.** It has its own overworld loop and also
  treads grass, so it will adopt the SAME `_handle_battle_interruption` helper in
  Brique 2, where the survey actually crosses grassy routes. Keeping it out of
  Brique 1 keeps this increment focused and its tests pure.
- No new Fighter capability (still wild-battle "win" only; trainer battles = Brique 3).
- No mid-battle telemetry, no capture/min_loss directives, no nearest-spot.
- No ROM smoke: this is pure composition of `navigate_to` + already-ROM-tested
  `play_battle`; there is no savestate of "walking on grass with a guaranteed
  next-step encounter and a winnable party" to make a deterministic smoke. The
  live proof comes in Brique 2 when survey crosses route_101's grass for real.

## Testing (pure, no ROM, no SB3)

`tests/test_live_navigator.py` — extend the existing `FakeWorld` family:

1. **Win-and-resume:** a fake world whose `in_battle()` returns True for one loop
   iteration on a grass cell, then False after `play_battle` is invoked; a scripted
   `predict`/`move_type_fn` that makes the fake `play_battle` path return `"won"`.
   Assert `navigate_to` reaches the target (does not `timeout`), the false-wall is
   never recorded, and (with `memory` passed) the grass cell IS learned as
   `has_grass` (recording happens before the fight).
2. **Lost propagates:** fake battle that the Fighter loses → `navigate_to` returns
   `"battle_lost"`.
3. **No Fighter → `battle_interrupted`:** `in_battle()` True, `move_type_fn`/`predict`
   omitted → `navigate_to` returns `"battle_interrupted"`.
4. **Regression:** `in_battle()` always False (existing fakes) → behavior unchanged,
   including the `memory=None` path.

`tests/test_map_traveler.py` — a fake where the first `navigate_to` hop hits a lost
battle → `travel_to` returns `"battle_lost"` (propagation, not a routing divergence).

`tests/test_orders.py` — `advance` to a known destination whose path treads a grass
cell with a Fighter wired → the wild battle is won and travel completes to
`"arrived"`; without a Fighter → `"battle_interrupted"`.

Because `play_battle` is invoked inside these paths, the fake worlds must serve the
battle-reader bytes the same way `GrassBattleWorld` already does in the existing
grind tests (reuse that pattern), and `move_type_fn`/`predict` are plain callables.

## Files touched

- `env/live_navigator.py` — new import `from env.battle_player import play_battle`;
  helper `_handle_battle_interruption`; `navigate_to` signature/loop (Create: helper; Modify: navigate_to).
- `env/map_traveler.py` — `travel_to` signature + 3 call sites + propagation.
- `env/orders.py` — `advance` path threads the Fighter deps; docstring returns.
- `tests/test_live_navigator.py`, `tests/test_map_traveler.py`, `tests/test_orders.py`.
