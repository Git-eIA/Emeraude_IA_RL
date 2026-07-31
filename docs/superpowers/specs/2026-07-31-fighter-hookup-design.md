# P4 — Fighter hookup on grind (win the triggered battle)

**Date:** 2026-07-31

## Goal

Make the Explorer's `grind` mode actually win the wild battle it triggers, using
the already-trained Fighter policy. This is the first time all three brains chain
end to end: Explorer navigates to grass -> grind treads until a battle starts ->
Fighter plays the battle to a win.

## Scope (one small step)

Win a **single** triggered battle. NOT in scope (later paliers):
- the level-up/auto-heal farm loop (grind -> fight -> grind, heal when low),
- the `capture` / `min_loss` combat directives (Fighter only knows how to win),
- active search for unknown grass, a real Strategist emitting Orders.

## Current state

- `env/orders.py::_execute_grind` treads via `_walk_until_encounter` and returns
  `"encounter_started"` the moment the battle flag rises — then stops (Fighter
  was deliberately not branched).
- The Fighter is trained: PPO `MlpPolicy` over `BattleEmeraldEnv` (17-dim RAM
  observation, `Discrete(4)` action), checkpoint
  `checkpoints/fighter/ppo_fighter_final.zip` (10/10 on the 5 wild savestates).
- One battle turn = `select_move(action)` + `advance_to_menu()` driven by the
  `BattleReader.at_action_menu()` RAM flag (`env/battle_env.py`).
- `make_move_type_fn(emu)` reads the ROM `gBattleMoves` table for the obs
  (`agent/train_fighter.py`).

## Design

### 1. Extract the shared battle-turn choreography — `env/battle_turn.py` (new)

Pure, stateless functions + constants, currently private methods on
`BattleEmeraldEnv`, so both the training env and the live player share ONE copy
of the turn choreography and the 17-dim observation layout:

- constants: `PRESS_HOLD_FRAMES`, `PRESS_RELEASE_FRAMES`, `SETTLE_FRAMES`,
  `OPEN_MENU_TRIES`, `MAX_ADVANCE_PRESSES`, `NUM_TYPES`, `MAX_PP`, `OBS_SIZE`
- `press(emulator, key)` — hold then release (GBA debounce)
- `select_move(emulator, reader, action)` — open move list, navigate, commit
- `advance_to_menu(emulator, reader)` — A-spam dialogue to the next menu or end
- `observation(state, move_type_fn) -> np.ndarray` — the 17-dim vector

`BattleEmeraldEnv` is refactored to delegate to these (behavior-preserving,
guarded by the existing `tests/test_battle_env.py`).

### 2. Live battle player — `env/battle_player.py` (new)

```
play_battle(emulator, move_type_fn, predict, max_turns=64) -> "won"|"lost"|"battle_timeout"
```

Builds its own `BattleReader(emulator.read_bytes)`, reaches the first action menu,
then loops `predict(observation) -> select_move -> advance_to_menu` until
`outcome != 0` or `not in_battle`. `won = outcome & 0x1` (same test as
`BattleEmeraldEnv._info`). No reward, no reset — the battle is already ongoing.

`predict: Callable[[np.ndarray], int]` is injected so `battle_player` stays free
of SB3/torch; production wraps a loaded PPO model.

### 3. Wire into grind — `env/orders.py`

`execute_order` and `_execute_grind` gain two optional injected deps
`move_type_fn=None, predict=None`. After `_walk_until_encounter` returns
`"encounter_started"`: if BOTH deps are provided -> `return play_battle(...)`;
otherwise -> return `"encounter_started"` (unchanged backward-compatible default,
keeps `orders.py` SB3-free and testable without a policy). The `combat` directive
is not branched on yet (only "win" behavior exists).

New grind outcomes: `no_grass_spot_known` | travel pass-through | `no_encounter`
| `encounter_started` (no Fighter) | `won` | `lost` | `battle_timeout`.

## Testing

- `tests/test_battle_turn.py` (pure): `observation()` layout/bounds; `select_move`
  press sequence via a scripted fake.
- `tests/test_battle_player.py` (pure): win, loss, timeout via a scripted battle
  emulator + a fixed `predict`.
- `tests/test_orders.py`: grind + Fighter deps -> `"won"`; grind without deps ->
  `"encounter_started"` (backward compat); `no_encounter` pass-through unchanged.
- `tests/test_battle_player_rom.py` (gated on ROM + checkpoint + states): load a
  real battle savestate + the real Fighter checkpoint, `play_battle` -> `"won"`
  (load-bearing: the Fighter is 10/10).

## Non-goals

Farm loop, capture/min_loss, Strategist emission, nearest-spot selection, active
grass search. Each is its own later step.
