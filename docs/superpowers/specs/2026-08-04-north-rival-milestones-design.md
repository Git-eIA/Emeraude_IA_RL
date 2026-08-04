# North-rival milestones (Phase 1) — design spec (2026-08-04)

## Goal

Give the Explorer a reward chain that pushes it **north out of route_101 to the
route_103 rival and wins that battle**, autonomously. This is Phase 1 of the
user's larger milestone vision (the "gros bloc"): the make-or-break passability
test of the route_101 → Oldale → route_103 corridor plus the first trainer win.

Concretely this delivers **machinery + a smoke test + a manual runbook** — NOT
the multi-hour training run itself. The run is a documented, human-launched
follow-up whose outcome is uncertain (the north corridor has never been crossed
by any policy). Phase 1's job is to make that run *launchable and observable*.

## Scope

**Phase 1 (this spec):**
- `reach_oldale` → `reach_route_103` → `beat_rival` milestone table.
- Reset the training env from `states/post_starter.state` (free-roam route_101,
  lv5 starter) instead of the truck, with the north-only table.
- Hook the trained **Fighter** into the env so wild battles (crossing grass) and
  the route_103 **trainer** battle are actually resolved mid-episode.
- Detect `beat_rival` from a *won trainer battle on route_103* (no unvalidated
  flag guess).
- A gated ROM smoke that proves the env dispatches the Fighter and fires a
  milestone, plus a runbook for the manual run.

**Deferred to Phase 2 (NOT this spec):**
- Return to Littleroot, the two Flora talks on the route, the Pokédex, the
  running shoes. These are detectable (flag mechanism is sound) but only *after*
  Phase 1 produces the savestates that let us confirm those flags live.
- Retraining the Fighter for the trainer matchup if the smoke reveals an
  in-distribution problem (mono-mon rival vs wild-trained Fighter).
- Nearest-spot navigation, capture/min_loss directives, Strategist emitting
  real Orders.

## Confirmed facts (verification done)

- **Story order (canonical, user-confirmed):** starter → save Birch → brief lab
  warp → go NORTH pre-Pokédex: route_101 → Oldale (Rosayère) → route_103 → beat
  rival (Flora, single lv5, ~wild difficulty) → return to lab → THEN Pokédex +
  Balls. So `post_starter` (no Pokédex, TOWN_STATE=2, in_battle False) is the
  NORMAL, intended pre-north state. North is the ungated intended direction.
- **Map IDs (pret map_groups.json, group 0):** LITTLEROOT=(0,9) ✓,
  ROUTE_101=(0,16) ✓, ROUTE_103=(0,18) — corroborated by `orders.py`
  DESTINATIONS, still to pin live. OLDALE=(0,10) is the pret ordinal; **pin it
  live during the run** (dump map_id on first north transition) rather than
  trusting the ordinal.
- **Flag reading is sound:** `game_state.py` flag mechanism (offset 0x1270)
  is load-bearing via clock_set + 8 badge flags; flag IDs are universal
  pokeemerald constants (FR ROM is a translation, identical RAM layout). Not
  needed in Phase 1 (all three milestones are position/battle detected) but
  relevant to Phase 2.
- **Trainer vs wild:** `BATTLE_TYPE_TRAINER = 0x0008` bit in `gBattleTypeFlags`
  distinguishes the two. `BattleReader.battle_state()` currently reads the flags
  word but only uses `flags != 0`; **no trainer accessor exists yet** (G2).
- **Good checkpoints:** Explorer `ppo_emerald_final_pre_palier0.zip` (10.5M,
  10/10 intro), `ppo_emerald_control05.zip` (9/10). Fighter
  `checkpoints/fighter/ppo_fighter_final.zip`.

## Architecture

Three additive changes, each behind a default that preserves the intro path.

### 1. `env/milestones.py` — north-only table + additive env condition

Add an optional `env_condition` to `Milestone` and a `route103_milestones()`
table. `env_condition` closes over an `EnvContext` (env-side signals the
`PlayerState` can't carry, i.e. "a trainer battle was won on route_103").

```python
@dataclass(frozen=True)
class EnvContext:
    """Env-side signals not present in PlayerState RAM."""
    rival_beaten: bool = False


@dataclass(frozen=True)
class Milestone:
    name: str
    condition: Callable[[PlayerState], bool]
    points: float
    terminal: bool = False
    # Additive: when set, BOTH condition(state) AND env_condition(ctx) must hold.
    # Default None keeps the 10 intro lambdas untouched (pure PlayerState).
    env_condition: Callable[[EnvContext], bool] | None = None
```

`OLDALE = (0, 10)` and `ROUTE_103 = (0, 18)` module constants (marked
"pin live"). New table:

```python
def route103_milestones() -> tuple[Milestone, ...]:
    """Phase 1 north push: leave route_101, reach route_103, beat the rival.

    Reset from states/post_starter.state (free-roam route_101, lv5). Does NOT
    reuse starter_milestones(): starter_obtained (party_count>=1, terminal)
    would fire at t=0 and end the episode instantly.
    """
    return (
        Milestone(
            "reach_oldale",
            lambda s: (s.map_group, s.map_num) == OLDALE,
            30.0,
        ),
        Milestone(
            "reach_route_103",
            lambda s: (s.map_group, s.map_num) == ROUTE_103,
            40.0,
        ),
        Milestone(
            "beat_rival",
            # Position guard (on route_103) AND the env says a trainer battle
            # was won there. The rival is the only trainer in Phase 1's range.
            lambda s: (s.map_group, s.map_num) == ROUTE_103,
            100.0,
            terminal=True,
            env_condition=lambda ctx: ctx.rival_beaten,
        ),
    )
```

`MilestoneTracker.update` gains an `EnvContext` argument (defaulted so existing
callers are unaffected):

```python
def update(
    self, state: PlayerState | None, ctx: EnvContext | None = None,
) -> tuple[float, bool]:
    if state is None:
        return 0.0, False
    ctx = ctx or EnvContext()
    reward = 0.0
    terminated = False
    for m in self._milestones:
        if m.name in self._fired or not m.condition(state):
            continue
        if m.env_condition is not None and not m.env_condition(ctx):
            continue
        self._fired.add(m.name)
        reward += m.points
        terminated = terminated or m.terminal
    return reward, terminated
```

Existing `starter_milestones()` callers pass no `ctx` → byte-identical behavior
(all 10 have `env_condition is None`).

### 2. `env/game_state.py` — expose the trainer bit (G2)

Add `BATTLE_TYPE_TRAINER = 0x0008` and surface it. Minimal change: a public
reader method (keeps `BattleState` churn-free for the many existing consumers).

```python
BATTLE_TYPE_TRAINER = 0x0008  # gBattleTypeFlags bit distinguishing trainer battles

class BattleReader:
    ...
    def is_trainer_battle(self) -> bool:
        """True when the current battle is against a trainer (not wild)."""
        return bool(self._u16(GBATTLE_TYPE_FLAGS_ADDR) & BATTLE_TYPE_TRAINER)
```

The env uses this to dispatch `play_battle` (wild) vs `play_trainer_battle`
(trainer). `battle_state().in_battle` remains the gate (already outcome-guarded
by the merged in_battle fix).

### 3. `env/pokemon_env.py` — Fighter hook + injectable milestones

Two additive constructor params and a battle hook in `step`.

New imports (env currently imports only `EmeraldReader, PlayerState` from
game_state and `MilestoneTracker, starter_milestones` from milestones):

```python
from env.battle_player import play_battle, play_trainer_battle
from env.game_state import BattleReader, EmeraldReader, PlayerState
from env.milestones import ROUTE_103, EnvContext, MilestoneTracker, starter_milestones
```

- `__init__(..., milestones=None, move_type_fn=None, predict=None)`:
  `self._milestones = MilestoneTracker(milestones or starter_milestones())`.
  Default preserves every current caller. `move_type_fn`/`predict` stored on
  `self`, injected the same way orders.py already threads the Fighter (env stays
  SB3-free; the training script wraps the loaded PPO).
- Add a battle reader: the env only has `self._reader = EmeraldReader(...)`
  today, so add `self._battle_reader = BattleReader(emulator.read_bytes)`
  alongside it (mirrors how WorldReader holds both).
- Track `self._rival_beaten: bool = False` (reset to False in `reset()`).
- In `step`, place the hook **immediately after `emulator.step(...)`, BEFORE the
  frame append** (`self._frames.append(...)`). Ordering matters: resolving the
  battle first means the frame appended for this step, and the `player_state()`
  read that follows, reflect the *post-battle* overworld — not a stale pre-battle
  frame. The battle is out-of-band for the Explorer's perception (the Fighter is
  a separate brain); the Explorer just sees "walked into a fight, came out the
  other side" in one step. The `pre` map read happens *inside* the hook, before
  `play_trainer_battle`, so it captures route_103 while the battle is still up.

```python
# battle hook: resolve any in-progress battle before the frame/state read.
# Placed right after emulator.step(keys, FRAMES_PER_ACTION), before the
# self._frames.append(...) line.
if self._move_type_fn is not None and self._predict is not None:
    bs = self._battle_reader.battle_state()
    if bs.in_battle:
        # G3: read the map at battle START. Safe here because battles never
        # warp the player — the map stays route_103 for the whole fight. (The
        # transient-garbage RAM trap, town_state=40 / clock_set False, only
        # happens during map WARPS, not battles.)
        pre = self._reader.player_state()
        on_route103 = pre is not None and (pre.map_group, pre.map_num) == ROUTE_103
        if self._battle_reader.is_trainer_battle():
            result = play_trainer_battle(self.emulator, self._move_type_fn, self._predict)
            if result == "won" and on_route103:
                self._rival_beaten = True
        else:
            play_battle(self.emulator, self._move_type_fn, self._predict)
```

Then the existing milestone update passes the context:

```python
milestone_reward, terminated = self._milestones.update(
    state, EnvContext(rival_beaten=self._rival_beaten),
)
```

`beat_rival` fires only when the agent is standing on route_103 (position guard)
**and** `_rival_beaten` is latched by a won trainer battle there. No flag guess.

The `_rival_beaten` latch already encodes "won a trainer battle *on route_103*"
(via the `pre` read), so the milestone lambda's `(map)==ROUTE_103` guard is
technically redundant. It is kept **intentionally** as a credit-assignment
guard: it makes the +100 land on a step where the agent is physically on
route_103, and if the post-battle map read is momentarily off, the latch is
sticky so `beat_rival` still fires on the next clean route_103 read (before the
episode ends). Belt-and-suspenders, not an oversight.

## Detection of `beat_rival` (why this is sound)

- The route_103 rival is the *only* trainer reachable in Phase 1's north
  corridor, so "won a trainer battle while on route_103" uniquely identifies it.
- `is_trainer_battle()` (0x0008) rejects the many wild battles crossing grass —
  those go through `play_battle` and never touch `_rival_beaten`.
- G3: the on-route_103 check reads the map *before* `play_trainer_battle` runs.
  This is safe because battles never warp the player (the map stays route_103
  throughout the fight); the documented RAM-incoherence trap — town_state=40 /
  clock_set False fugaces — is a *warp* phenomenon, not a battle one.
- `beat_rival` is terminal → the episode ends on the win, which is exactly the
  Phase-1 success signal for eval.

## Smoke test (gated ROM)

`tests/test_north_rival_milestones_rom.py`, triple-skip
(ROM unset | Fighter ckpt missing | `states/trainer_battle.state` missing):

- Load `states/trainer_battle.state` (produced by the existing
  `tools/capture_rival_battle.py` / `capture_trainer_battle.py`, the same
  artifact Brique 3's smoke needs).
- Build `PokemonEmeraldEnv([state], milestones=route103_milestones(),
  move_type_fn=..., predict=...)` wrapping the real Fighter.
- Assert precondition `is_trainer_battle()` True and `in_battle` True.
- Step once; assert the battle was **dispatched and resolved** — `in_battle` is
  False afterward (proves `play_trainer_battle` ran, not `play_battle`, which
  would mis-handle send-outs). This is the load-bearing assertion regardless of
  where the state was captured.
- **Conditional on the state being on route_103** (read the map at load): assert
  `_rival_beaten` is True and `beat_rival` is in `tracker.fired` with the episode
  terminated. If the captured state is a trainer battle *elsewhere* (the
  route_103 capture is still deferred — route_103 is currently unreachable, see
  the geometry finding), skip this half rather than asserting a latch that
  correctly did not fire.

This proves the trainer-vs-wild dispatch + (on route_103) the env_condition
wiring end-to-end on a real ROM. The dispatch half is load-bearing the moment
any `states/trainer_battle.state` exists; the latch half becomes load-bearing
once that state is captured on route_103.

Pure tests (no ROM) cover the machinery fully:
- `test_milestones.py`: `env_condition` gates a fire (position true + ctx false
  → no fire; both true → fires); `starter_milestones()` unaffected by a passed
  ctx; `route103_milestones()` shape/points/terminal.
- `test_game_state.py`: `is_trainer_battle()` true/false via crafted flags bytes
  (0x0008 set vs a wild flags word).
- `test_env.py`: injected `milestones=` swaps the table; a fake
  move_type_fn/predict + a scripted trainer battle latches `_rival_beaten` and
  fires `beat_rival`; wild battles do NOT latch it; default ctor still uses
  `starter_milestones()`.

## Manual runbook (the multi-hour run — documented, not executed here)

Run from the main repo (`~/Projets/Emu`, where ROM/checkpoints/states live):

1. **Warm-start** from a good Explorer checkpoint (`ppo_emerald_control05.zip`
   or `_pre_palier0`) — reward change ≠ retrain from scratch; learned navigation
   transfers.
2. Train with `initial_states=[post_starter.state]`,
   `milestones=route103_milestones()`, Fighter wired,
   `max_steps` generous (north corridor is long; start ~8192), MPS/4 envs.
3. **Pin the map IDs live:** log map_id on the first transition off route_101
   → confirm OLDALE and ROUTE_103 against (0,10)/(0,18); backfill the constants
   if they differ.
4. **Dump a savestate at the first `beat_rival`** (add a one-shot save hook in
   the run script). Since `beat_rival` is terminal, this is the *only* moment to
   capture the post-rival state → it unblocks Phase 2 (confirm the return /
   Flora-talk / Pokédex / shoes flags live). This is runbook item **G8**.
5. Eval: 10 stochastic episodes from `post_starter.state`; success =
   `reach_route_103` and `beat_rival` fire in a meaningful fraction. Stall at
   route_101 = R1 materialized (see Risks).

## Risks (documented, accepted for Phase 1)

- **R1 — passability unproven.** No policy has ever crossed route_101 → Oldale.
  The run IS the test. Fallback if it stalls at the north edge of route_101:
  investigate the boundary live (the route103-geometry probe found post_starter
  boxed in the *southern* region — the north exit may need a specific
  walkable-tile approach). Do NOT resurrect Go-Explore multi-reset (proven to
  collapse the chain).
- **R2 — sparse guidance.** The Oldale→route_103 and route_103→rival legs have
  no intermediate milestone; route_102 branches *west* off Oldale (a trap). The
  30/40/100 point gradient is the only pull. If eval shows the agent wandering
  into route_102, add an intermediate `reach_oldale_north_exit` shaping
  milestone in a Phase-1.5 tweak.
- **R3 — throughput.** Crossing grass fires frequent wild battles, each resolved
  synchronously by the Fighter → slower wall-clock steps. Acceptable; note it in
  the runbook so the operator sizes the run accordingly.
- **R4 — mono-Pokémon party.** `post_starter` has a single lv5 starter. A faint
  = whiteout with no replacement. `play_trainer_battle`'s send-out handling is
  untested live against a real faint; if the Fighter loses the rival battle the
  episode just doesn't fire `beat_rival` (no crash). Retraining/replacement
  policy is Phase 2 if the smoke/run shows it matters.

## Non-goals

`map_map` survey unchanged. No Strategist wiring. No capture/min_loss. No
Pokédex/shoes/return milestones. No Fighter retraining. `starter_milestones()`
and every existing env caller behave identically (all new params defaulted).

## Files

- Modify: `env/milestones.py` (EnvContext, Milestone.env_condition,
  route103_milestones, OLDALE/ROUTE_103 consts, tracker ctx arg).
- Modify: `env/game_state.py` (BATTLE_TYPE_TRAINER, `is_trainer_battle`).
- Modify: `env/pokemon_env.py` (milestones/move_type_fn/predict params,
  `_rival_beaten`, battle hook, ctx-passing milestone update).
- Test: `tests/test_milestones.py`, `tests/test_game_state.py`,
  `tests/test_env.py` (pure) + `tests/test_north_rival_milestones_rom.py`
  (gated).
