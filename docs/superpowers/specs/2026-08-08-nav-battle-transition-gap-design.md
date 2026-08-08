# Fix — battle-transition detection gap in grid navigation

Date: 2026-08-08
Branch: `fix/nav-battle-transition-gap`
Status: design (awaiting spec review before writing-plans)

## Problem (root cause, probe-confirmed)

Scripted north navigation on grass-dense route_101 stalls and returns a false
`unreachable`. Six throwaway probes isolated the cause; it is NOT terrain, A\*, a
coordinate offset, or a sliding-window grid.

`navigate_grid` reaches (11,10) via a ledge jump, then every planned press reads
`blocked` and the per-run `blocked` set fills until `plan_path_grid` returns None
→ false `unreachable`. `tools/probe_frozen_at_waypoint.py` read the RAW battle RAM
while idling at (11,10):

```
idle+0..80 : flags=0x0000 outcome=1 opp_maxhp=13 in_battle=False  (won-battle residual)
idle+100   : flags=0x0004 outcome=1 opp_maxhp=13 in_battle=False  (NEW battle intro)
idle+120   : flags=0x0004 outcome=0 opp_maxhp=13 in_battle=True   (battle active)
after RIGHT: in_battle=True                                       (input does nothing)
```

Grass-dense route_101 pulls the player into wild battles constantly. The freeze is
invisible to the nav loop because `WorldReader.in_battle()` (game_state.py:219 =
`flags != 0 AND opp max_hp > 0 AND outcome == 0`) reads **False during two battle
transition windows**:

- **intro**: `gBattleTypeFlags` set but `gBattleMons` opponent `max_hp` still 0;
- **end-fade**: a terminal `gBattleOutcome` (won/lost) lingers while the overworld
  fades back in.

During those windows `handle_battle_interruption` returns None (it only acts on
`in_battle()`), so `navigate_grid` proceeds to `probe_step`, which presses a FROZEN
game → `resolve_move` returns `blocked` → the edge is recorded as a wall in the
`blocked` set → `plan_path_grid` eventually returns None → false `unreachable`.

Scope of impact: this freezes navigation on ANY grass traversal, not just (11,10).
Fixing it should unblock the whole route_101 → Oldale → route_103 traversal, which
is the real reach blocker for beating the route_103 rival.

## Goal

`navigate_grid` (and every caller of `handle_battle_interruption`, incl.
`grid_explorer`) must treat a battle transition window as "wait it out / play the
battle", never as a wall. A press is only recorded as blocked when the game is
confirmed to be in overworld control.

Non-goals (explicitly out of scope):
- No change to `play_battle` or the trainer-battle path.
- No change to `in_battle()` semantics (grass-learning / trainer logic depend on it).
- No new navigation capability, no A\* change, no grid-decode change.
- Not solving battle attrition or the Oldale north-exit geometry (separate, downstream).

## Design (option C = robust detection + anti-poison guard)

### 1. New raw signal on the reader: `battle_starting()`

`env/game_state.py` `BattleReader` gains a cheap method (2 small RAM reads, no full
mon parse — safe for the nav hot loop):

```python
def battle_starting(self) -> bool:
    """True while a battle's flags are set with no terminal outcome yet.

    Covers the intro window (flags set before gBattleMons populates, so
    battle_state().in_battle is still False) and the active battle. Residual
    flags from a loaded post-battle savestate carry a terminal outcome != 0 and
    read False, so a freshly loaded post_starter savestate does not hang.
    """
    flags = int.from_bytes(self._read(GBATTLE_TYPE_FLAGS_ADDR, 2), "little")
    outcome = self._read(GBATTLE_OUTCOME_ADDR, 1)[0]
    return flags != 0 and outcome == 0
```

`env/world_reader.py` `WorldReader.battle_starting()` delegates to
`self._battle.battle_starting()` (mirrors the existing `in_battle()` passthrough).

Why `outcome == 0` and not `flags` alone: `gBattleTypeFlags`/`gBattleMons` are not
cleared when a battle ends, so a loaded savestate (e.g. `post_starter.state`) keeps
flags set. Gating on `outcome == 0` distinguishes a *live* intro/active battle from
*residual* flags. This is the same discriminator `in_battle()` already relies on.

### 2. `handle_battle_interruption` guarantees overworld control on return

Rewrite so that when it returns None, the game is confirmed to be in overworld
control (or there was never a battle). Signature unchanged
`(emulator, reader, move_type_fn, predict) -> str | None`, so no caller changes.

```python
def handle_battle_interruption(emulator, reader, move_type_fn, predict) -> str | None:
    if not reader.battle_starting():
        return None
    # Intro window: flags are set but the opponent has not populated yet, so
    # in_battle() is still False. Idle (bounded) until it confirms.
    for _ in range(BATTLE_TRANSITION_SETTLE):
        if reader.in_battle():
            break
        emulator.step(0, RELEASE_FRAMES)
    if not reader.in_battle():
        return None  # flags set but never became a real battle: not ours
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result != "won":
        return "battle_lost" if result == "lost" else "battle_timeout"
    # End-fade: let overworld control return before the caller presses again.
    for _ in range(BATTLE_TRANSITION_SETTLE):
        if not reader.battle_starting() and not reader.in_battle():
            break
        emulator.step(0, RELEASE_FRAMES)
    return None
```

`BATTLE_TRANSITION_SETTLE` is a small bounded constant (e.g. 8) — both loops are
bounded (code-safety #2).

### 3. Anti-poison guard in `navigate_grid`

The top-of-loop `handle_battle_interruption` call already runs every iteration, so
before each press we are in overworld control. The remaining false-wall source is a
battle triggered *by the press itself* (stepping onto grass): `probe_step` returns
`blocked` while a battle is starting. Guard the poison:

```python
if outcome == "blocked":
    if reader.battle_starting() or reader.in_battle():
        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle
        continue                       # consumed a battle; do NOT poison, re-plan
    blocked.add((before.pos, direction))   # genuine wall / NPC
```

Only a press that fails with NO battle context poisons the `blocked` set — the
existing NPC-detour behavior is preserved.

## Known minor race (documented, accepted)

There is a ~20-frame window where new battle flags are already set but the previous
battle's terminal `outcome` has not yet cleared to 0 (probe idle+100:
`flags=0x0004 outcome=1`). During it `battle_starting()` reads False. Because the
top-of-loop handler runs every iteration (~32+ frames/iter) it catches the battle on
the next pass, and the anti-poison guard only poisons when NO battle context is
present. Worst case: one spurious blocked edge that a re-plan routes around; it does
not reproduce the hang. Accepted rather than adding stateful outcome-edge tracking.

## Coupling introduced

`handle_battle_interruption` runs at the top of every `navigate_grid` loop iteration
and now calls `reader.battle_starting()` (before it called `reader.in_battle()`). So
**every** reader that flows into `navigate_grid` — directly or via
`travel_to`/`execute_order`/`map_map`/`survey_world`/`explore_grid` — must expose
`battle_starting()`. In production this is always `WorldReader` (one delegation). In
tests it means adding a one-line `battle_starting(self) -> bool: return False` stub
to the fakes in every file whose reader reaches `navigate_grid`:

- `tests/test_grid_navigator.py`, `tests/test_grid_explorer.py`
- `tests/test_orders.py` (6 fakes), `tests/test_map_traveler.py` (2),
  `tests/test_world_surveyor.py` (2)

No `getattr`/default shim (per code-safety / no backwards-compat hacks) — the fakes
add the method explicitly, mirroring the existing `in_battle()` stub coupling.

## Tests

Pure (crafted bytes / fakes, no ROM):
- `test_game_state.py` (or `test_battle_reader.py`): `battle_starting()` truth table
  — flags set & outcome 0 → True; flags set & outcome 1 → False; flags 0 → False.
- `test_world_reader.py`: `battle_starting()` delegates to the battle reader.
- `test_grid_navigator.py`:
  - **intro race**: a fake reader whose press-time state is "battle_starting True,
    in_battle flips True after a step; play returns won" → `navigate_grid` reaches the
    target and the `blocked` set stays empty (no false wall). This is the direct
    regression for the bug.
  - **fade**: fake where after a won battle `battle_starting`/`in_battle` stay True for
    a couple steps then clear → no poison, nav completes.
  - **genuine wall preserved**: a press that fails with NO battle context still
    poisons and reroutes / returns `unreachable` (existing behavior unchanged).
  - **handle_battle_interruption units**: no battle → None; intro-then-win → None with
    overworld control; lost/timeout → terminal string; no Fighter → "battle_interrupted".

ROM smoke (load-bearing, gated on ROM + Fighter ckpt + `states/post_starter.state`):
- `test_nav_battle_gap_rom.py`: from `post_starter.state`, `navigate_grid` to (11,0)
  (which returns `unreachable` before the fix). Primary assertion (the exact bug
  symptom): `result != "unreachable"`. Corroboration: the player advances strictly
  north of the trap band (final `pos.y < 10`), proving the (11,10) freeze no longer
  poisons the blocked set. Reaching (11,0) exactly is NOT asserted — the separate
  Oldale north-exit geometry gap (west pocket behind a one-way ledge) is out of scope,
  so `arrived`/`left_map`/`timeout` are all acceptable as long as it is not
  `unreachable` and it crossed north. Wraps the real Fighter; SB3/torch imported
  inside the test body.

## Rollout

Worktree `fix/nav-battle-transition-gap`. subagent-driven-development after
writing-plans. Full suite + ruff must be green; the ROM smoke is the load-bearing
proof. Delete the throwaway probe tools (`probe_stuck_cell.py`,
`probe_frozen_at_waypoint.py`, and the earlier reach probes) as part of cleanup, or
keep only if a reviewer wants the evidence.
