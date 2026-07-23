# M7 — Fighter (second brain) design

**Date:** 2026-07-23
**Milestone:** M7
**Status:** approved, ready for implementation plan

## Goal

Build the **Fighter**, a second RL brain that learns to win battles, and train
it to beat the Route 103 rival battle reliably from a savestate. The live
handoff between the Explorer and the Fighter in a continuous playthrough is
explicitly **out of scope** (deferred to M8). M7 delivers and validates the
Fighter in isolation.

## Non-goals (M7)

- No real-time Explorer <-> Fighter switching in a live game (M8).
- No change to the existing Explorer policy, `pokemon_env`, or intro chain.
- No switching Pokemon or running away (single Pokemon, trainer battle).
- No multi-battle sequences (gym, route crossing) — that is later.

## Why a separate brain

The Explorer sees **pixels** (`CnnPolicy`). Battles are driven by discrete
state (HP, types, PP, moves) that is far easier to learn from **numbers read
out of RAM** with a dense network (`MlpPolicy`). A separate brain also matches
the long-term 3-brain vision (Strategist / Explorer / Fighter) and lets the
Fighter train on a fast, dedicated battle environment (one episode = one
battle) instead of replaying the intro every time.

The Fighter is a **persistent brain**: it keeps learning across milestones
(curriculum). Type mastery is not learnable at level 5 (no type-diverse moves),
but the observation already carries move types, so it becomes learnable later
with no architecture change.

## Architecture

Existing three layers stay intact; a battle branch is added:

```
emulator/               (unchanged — mGBA wrapper)
env/
  game_state.py         + BattleReader / BattleState (reads battle RAM)
  battle_env.py         NEW: Gym env, one episode = one battle
  battle_rewards.py     NEW: battle reward shaping
agent/
  train_fighter.py      NEW: trains the Fighter (PPO + MlpPolicy)
states/battles/         NEW: savestates of battle openings (the curriculum)
tools/
  probe_battle.py       NEW: empirically validate battle RAM addresses
  make_battle_states.py NEW: capture the battle-opening savestates
```

Each unit is independently testable: `BattleReader` via crafted RAM bytes,
`battle_env` via `FakeEmulator`, `battle_rewards` via pure unit tests, training
measured by a win rate.

## Battle RAM (BattleReader)

Exact BPEF addresses are **not yet known**. Following the established project
pattern (the intro-var probe of 2026-07-23), the **first plan task** is a probe
(`tools/probe_battle.py`) that starts a battle and validates the addresses
empirically before any dependent code is written. Target structures from
pret/pokeemerald (to confirm on BPEF):

- **`gBattleMons`** — array of 4 `BattlePokemon`. Per mon: species, current HP,
  max HP, level, two types, four moves + their PP.
- **`gBattleTypeFlags`** — nonzero while in battle; encodes wild vs trainer.
- **`gBattleOutcome`** — how the battle ended (win / loss / run); 0 while ongoing.

**In-battle detection:** primary signal `gBattleTypeFlags != 0`, cross-checked
with a second signal (opponent max HP > 0). This applies the intro RAM-trap
lesson: **always bound and cross-check** to reject garbage reads during
transitions.

`BattleReader` lives beside `EmeraldReader` in `game_state.py` and exposes an
immutable dataclass:

```python
@dataclass(frozen=True)
class BattleState:
    in_battle: bool
    my_hp: int
    my_max_hp: int
    my_level: int
    my_types: tuple[int, int]
    my_moves: tuple[MoveInfo, ...]      # 4 entries: (type, pp, power)
    opp_hp: int
    opp_max_hp: int
    opp_level: int
    opp_types: tuple[int, int]
    opp_species: int
    outcome: int                        # 0 = ongoing
```

Returns `None` while save blocks relocate, same as `player_state()`.

## Battle environment (battle_env.py)

One episode = one battle.

- **reset():** loads a random savestate from `states/battles/`. Battle openings
  include several wild encounters (Routes 101/103) **and** the rival battle.
- **Action space = Discrete(4):** "use move 0/1/2/3". The env translates the
  chosen move into the button sequence (A to open FIGHT, navigate to the slot,
  A to confirm). Between decision points the env **spams A** to advance dialogue
  until the player must choose again or the battle ends.
- **Observation = normalized float vector:**
  - my HP / max HP, my level; opponent HP / max HP, opponent level
  - my two types + opponent two types (encoded)
  - per move (x4): move type (encoded), PP (normalized), power (normalized)
- **Episode ends** when `outcome != 0` (win/loss) or a max-turn cap is reached.

## Battle rewards (battle_rewards.py)

HP is measured in **health bars** (fraction of max HP), so the scale stays
stable across levels. Dealing damage is worth **2x** taking damage.

| Event | Reward |
|---|---|
| Battle won | +100 |
| Enemy Pokemon fainted | +20 each |
| Own Pokemon fainted | -10 each |
| Super-effective move used | +5 |
| Damage dealt this turn | +10 x (enemy bar removed) |
| Damage taken this turn | -5 x (own bar lost) |
| Per turn elapsed | -0.1 |

Starting constants (tunable): `DEAL_BAR = 10.0`, `TAKE_BAR = 5.0` (2:1 ratio),
`WIN = 100.0`, `ENEMY_FAINT = 20.0`, `OWN_FAINT = -10.0`,
`SUPER_EFFECTIVE = 5.0`, `TURN_PENALTY = -0.1`. There is **no** flat defeat
penalty — losing is captured only through fainted Pokemon and slowness.

Note: `SUPER_EFFECTIVE` stays dormant at the level-5 rival (no type-diverse
moves) and becomes active later as the movepool grows. Harmless meanwhile.

## Training & evaluation

- `tools/make_battle_states.py` captures the battle-opening savestates into
  `states/battles/`.
- `agent/train_fighter.py`: PPO with **MlpPolicy** (dense net, no images),
  lighter and faster than the Explorer.
- **Success metric:** eval on the rival savestate — **win rate** (target
  >= 9/10) plus average turns-to-win. Stochastic eval (`deterministic=False`),
  consistent with Explorer eval practice.

## Testing

- `BattleReader`: `FakeEmulator` returns crafted battle bytes (HP, types, PP);
  parsing verified without a ROM.
- `battle_env`: Gym API compliance (`check_env`) + one simulated turn
  (damage -> positive reward, enemy faint -> +20, etc.) via `FakeEmulator`.
- `battle_rewards`: pure unit tests for each row of the table, mirroring
  `test_rewards.py`.
- Tests needing real battles are gated by the `POKEMON_EMERALD_ROM` env var,
  like the rest of the suite.

## Risks / open items

- Battle RAM addresses unverified on BPEF — mitigated by the probe-first task.
- Super-effective detection needs a type-effectiveness table (18x18) or a RAM
  read of the damage multiplier; the probe task should check whether the game
  exposes the last-move effectiveness before we hand-roll the table.
- Menu navigation from RAM state must reliably reach the move-select prompt;
  the env's "spam A until decision point" loop must detect that prompt robustly.
