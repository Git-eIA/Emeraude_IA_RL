# Brique 3 (part 1) — Trainer-battle combat primitive (`play_trainer_battle`)

**Date:** 2026-08-03
**Status:** approved, ready for planning
**Worktree/branch:** `Emu-p4-trainer-battle` / `feat/p4-trainer-battle`

## Context

The overarching goal is to beat the rival on route_103 with full autonomy from
route_101, decomposed into three independent briques (each its own
spec→plan→impl):

- **Brique 1** (merged, `b365f8c`) — interruptible navigation: a wild battle that
  starts mid-move is handed to the Fighter and travel resumes.
- **Brique 2** (merged, `d6437ff`) — battle-proof survey: `map_map`/`survey_world`
  cross grassy routes, threading the Fighter and aborting on a battle outcome.
- **Brique 3** (this work) — beat the rival trainer on route_103 + campaign
  milestone. This spec covers **part 1 only**: the trainer-capable combat
  primitive. Navigation to route_103, the rival trigger, and the campaign
  milestone wiring are deferred to a follow-up.

### The technical lock

`env/battle_player.py::play_battle` drives an ongoing battle to a terminal
outcome using a caller-injected `predict`. Its loop stops on
`state.outcome != 0 or not state.in_battle`. This is correct for **wild**
battles (single opponent Pokémon; the battle can only end by a terminal
outcome).

Trainer battles (the rival) are **multi-Pokémon**: when the opponent's active
Pokémon faints, the trainer sends out the next one; the battle `outcome`
(`0x0202433A`) becomes non-zero only once the **whole** opponent team is beaten.

The reader computes `in_battle = flags != 0 and opp["max_hp"] > 0`
(`env/game_state.py:209`). During the opponent's send-out animation,
`opp max_hp` momentarily reads 0 → `in_battle` is briefly `False` while the
battle is still going. `play_battle` would therefore **falsely end** the battle
mid-fight (returning `_result(0)` → `"lost"`). We need a sibling that ends only
on a real terminal `outcome`.

## Scope

**In scope (part 1):**

- A new sibling function `play_trainer_battle` in `env/battle_player.py` that
  drives a multi-Pokémon trainer battle to a terminal outcome.
- Pure unit tests (scripted multi-mon fake, no ROM/SB3).
- A disposable capture tool that produces `states/trainer_battle.state`.
- A gated (double-skip) deterministic ROM smoke that loads the artifact and
  drives the real Fighter to `"won"`.

**Out of scope (deferred to Brique 3 part 2 and beyond):**

- Navigation to route_103 and triggering the rival.
- Campaign-milestone wiring (`campaign.py` `advance` staying navigation-only).
- Any strategy for choosing our replacement Pokémon beyond the A-press that
  `advance_to_menu` already performs (see Known risk).
- `capture` / `min_loss` combat directives (Fighter only knows "win").
- Modifying `play_battle` (the wild path stays byte-for-byte intact) or the
  Fighter policy / its training.

## Design

Two changes: a trainer-aware variant of the shared `advance_to_menu` helper, and
the new `play_trainer_battle` that uses it.

### 1. `advance_to_menu` gains a `wait_through_faint` flag (env/battle_turn.py)

This is the crux the gap-check surfaced. The shared helper currently returns on
`state.outcome != 0 or not state.in_battle` (battle_turn.py:57). During a
trainer's send-out, `opp max_hp` momentarily reads 0 → `in_battle` is briefly
`False` → the helper would **return mid-send-out**, handing control back to the
caller with a zero-HP opponent still on screen. A `play_trainer_battle` loop that
only re-checked `outcome` would then feed `predict` a garbage (dead-opponent)
observation.

Fix: add a defaulted keyword so the wild path is **byte-for-byte unchanged** at
its default, and the trainer path waits through the faint/send-out:

```python
def advance_to_menu(emulator: Any, reader: Any, wait_through_faint: bool = False) -> None:
    """Press A to clear dialogue until back at the action menu or battle end.

    Wild battles (default) also return when in_battle drops. Trainer battles pass
    wait_through_faint=True: a faint/send-out momentarily reads opp max_hp==0
    (in_battle False) while the battle continues, so those stop ONLY on a terminal
    outcome or the next live action menu.
    """
    for _ in range(MAX_ADVANCE_PRESSES):
        state = reader.battle_state()
        if state.outcome != 0:
            return
        if not wait_through_faint and not state.in_battle:
            return
        if reader.at_action_menu():
            emulator.step(0, SETTLE_FRAMES)
            return
        press(emulator, buttons.KEY_A)
```

With `wait_through_faint=True`, the loop keeps pressing A through the send-out
(and through our own fainted-mon replacement screen, selecting the first healthy
mon by default) until either the whole opponent team is beaten (`outcome != 0`)
or the next live opponent action menu appears. Still bounded by
`MAX_ADVANCE_PRESSES` (code-safety #2). Default `False` → every existing call
site (the wild `play_battle`, `BattleEmeraldEnv`) is unaffected, guarded by the
unchanged `test_battle_turn.py` / `test_battle_env.py`.

### 2. `play_trainer_battle` (env/battle_player.py)

A sibling of `play_battle`, reusing `observation`, `select_move`,
`BattleReader`, and `_result` **unchanged**, and calling `advance_to_menu` with
`wait_through_faint=True`. The terminal condition is `outcome != 0` alone.

```python
def play_trainer_battle(
    emulator: Any,
    move_type_fn: MoveTypeFn,
    predict: PredictFn,
    max_turns: int = 128,
) -> str:
    """Play an ongoing multi-Pokémon trainer battle to the end.

    Unlike play_battle (wild, single opponent), a trainer sends out the next
    Pokémon when its active one faints; opp max_hp reads 0 for a tick during
    send-out, so this stops ONLY on a terminal outcome, never on not in_battle.

    Returns "won" (outcome bit 0x1 set), "lost" (any other terminal outcome),
    or "battle_timeout" (max_turns reached without a terminal outcome).
    """
    reader = BattleReader(emulator.read_bytes)
    advance_to_menu(emulator, reader, wait_through_faint=True)
    for _ in range(max_turns):
        state = reader.battle_state()
        if state.outcome != 0:
            return _result(state.outcome)
        action = predict(observation(state, move_type_fn))
        select_move(emulator, reader, int(action))
        advance_to_menu(emulator, reader, wait_through_faint=True)
    state = reader.battle_state()
    if state.outcome != 0:
        return _result(state.outcome)
    return "battle_timeout"
```

**Why this is correct.** With `wait_through_faint=True`, `advance_to_menu` only
returns on a terminal `outcome` or a live action menu, so the loop's
`battle_state()` is only sampled either at battle-end or with a live opponent —
`predict` is never fed a zero-HP opponent, and the transient `opp max_hp == 0`
during send-out is never mistaken for a battle end.

**Bounds & style.** Loop bounded by `max_turns` (code-safety #2). `max_turns`
defaults to **128** (multi-mon trainer battles take more turns than the wild
default of 64). Function stays well under 40 lines (#4). `_result` is reused
verbatim — `BATTLE_OUTCOME_WON == 1` in pokeemerald, so `outcome & 0x1` == won
holds for trainer battles too.

**Contract.** Returns `"won"` | `"lost"` | `"battle_timeout"`.

### Known risk (flagged, not solved in part 1)

When **our** active Pokémon faints in a trainer battle, Emerald shows a
replacement-choice screen. Under `wait_through_faint=True`, `advance_to_menu`
keeps pressing A there, which selects the first healthy party mon (Emerald
highlights it by default). This is plausible but **not verified live**; the ROM
smoke exercises it. Related in-distribution risk: the Fighter was trained on wild
single-mon battles and now faces a multi-mon trainer. If the smoke reveals either
problem, it becomes a follow-up (an explicit replacement policy and/or Fighter
retraining) — the primitive's contract is unchanged.

## Testing

### Pure unit tests (no ROM/SB3)

**`tests/test_battle_turn.py` (+1) — the helper flag.** A tiny fake reader whose
`battle_state()` returns a scripted sequence and an `at_action_menu()` toggle:
with `wait_through_faint=False` (default), a state with `in_battle == False`
(outcome 0) returns immediately (wild behaviour, unchanged); with
`wait_through_faint=True`, the same `in_battle == False` state does **not**
return — the helper keeps pressing A until `at_action_menu()` becomes True (or a
non-zero outcome). Asserts the A-press count differs between the two modes.

**`tests/test_battle_player.py` (+4) — `play_trainer_battle`.** A scripted
multi-mon fake emulator/reader (mirroring the existing `_ScriptedBattle` used by
`play_battle`'s tests), where each A-press advances an internal script cursor so
send-out ticks (`opp max_hp == 0`, `at_action_menu` False) are consumed exactly
as the live loop would:

1. **Send-out does not falsely end** — the script yields `opp max_hp == 0` for
   one tick during send-out **without** setting `outcome`, then a live opponent
   menu, then `outcome == 1`. Asserts `play_trainer_battle` returns `"won"`. A
   control assertion runs `play_battle` (wild) on the same fake and asserts it
   returns `"lost"` — because the wild `advance_to_menu` (default
   `wait_through_faint=False`) returns on the transient `not in_battle`, and
   `play_battle`'s loop then reads `not in_battle` and returns `_result(0)`.
   This makes the regression the sibling fixes explicit.
2. **Two opponent mons then win** — beat opponent mon #1 (send-out), beat mon #2,
   `outcome == 1` → `"won"`.
3. **Loss** — `outcome == 2` (BATTLE_OUTCOME_LOST) → `"lost"`.
4. **Timeout** — never terminal within `max_turns` → `"battle_timeout"`.

### Capture tool (`tools/capture_trainer_battle.py`, disposable)

Scripts reaching a trainer battle from a known savestate and dumps
`states/trainer_battle.state`. Exact reachability path (which state to load,
which trainer to trigger) is a planning/implementation detail resolved during
planning; the tool is disposable and mirrors the Brique 2 capture tool
(`tools/capture_route101_in_battle.py`): bounded stepping, `raise SystemExit(1)`
if no trainer battle is reached within the step budget, prints the landing
battle state for the record. **If no trainer battle turns out to be reachable
from an existing savestate by a scripted walk, the tool is descoped and the ROM
smoke stays a documented skip** (honest, exactly as Brique 2's smoke did until
its artifact existed) — the pure tests still fully cover the primitive's logic.

### Gated ROM smoke (`tests/test_battle_player_rom.py`, +1)

Double-skip gate (`POKEMON_EMERALD_ROM` unset **or** artifact missing), mirroring
the Brique 2 smoke. Loads `states/trainer_battle.state`, asserts the precondition
`reader.in_battle()` (really mid-trainer-battle), loads the real Fighter
checkpoint (`checkpoints/fighter/ppo_fighter_final.zip`, `device="cpu"`), wraps
`predict`, runs `play_trainer_battle`, and asserts the result is `"won"` and
`not reader.in_battle()` afterwards. SB3/torch imported inside the test body.
Skips until the artifact is produced locally once → then load-bearing.

## Assumptions

- `BATTLE_OUTCOME_WON == 1` (outcome bit 0x1) applies to trainer battles — same
  `GBATTLE_OUTCOME_ADDR` and enum as wild battles in pokeemerald. **Validated by
  the ROM smoke.**
- `advance_to_menu` with `wait_through_faint=True` absorbs the opponent send-out
  dialogue within `MAX_ADVANCE_PRESSES` (=120) A-presses and reaches the next
  live action menu — it already clears the wild path's own-move result dialogue
  the same way. **Validated by the ROM smoke.**
- The default replacement selection (A-press on our fainted-mon screen) does not
  soft-lock the primitive. **Probed by the ROM smoke; flagged as a follow-up if
  it fails.**
- A trainer battle is reachable from an existing savestate by a scripted walk so
  the capture tool can produce the artifact. **If false, the ROM smoke stays a
  documented skip (Brique 2 precedent); the pure tests remain load-bearing.**

## Non-goals recap

Navigation to route_103, rival trigger, campaign milestone, replacement policy,
capture/min_loss, `play_battle` changes, Fighter retraining.
