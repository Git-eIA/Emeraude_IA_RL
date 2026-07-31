# Go-Explore Palier 0 — Reset-State Distribution (design)

Date: 2026-07-31
Status: approved (design), pending implementation plan
Scope: Explorer training only. Disjoint from `feat/p4-heal-mode`.

## Motivation

At ~9-10.5M steps the Explorer PPO reached `route_101` 9/10 and `meet_rival`
10/10, then **regressed to 0/10** while `ep_rew_mean` kept rising. This is
textbook **detachment** (Go-Explore, arxiv 2004.12919): the policy drifts away
from a promising region and forgets how to return, and mid-episode tile-farming
emerges on the long walk from spawn.

Palier 0 tests the cheapest Go-Explore idea — a **reset-state distribution**:
start some training episodes from savestates captured at the frontier
(route_101, meet_rival, …) instead of always from `states/initial.state` (the
truck). Hypothesis: keeping a foothold at the frontier prevents the collapse.

This is the smallest reversible increment. If it pays off we go to Palier 1
(auto-archive during rollouts + cutting hackable bonuses). If not, we abandon at
near-zero cost.

## Scope & isolation

Palier 0 changes **exactly one variable**: the distribution of start states at
training time. It does **NOT** touch the reward function (`NEW_TILE_REWARD`,
`REVISIT_PENALTY`, `TIME_PENALTY`, milestones) — cutting hackable bonuses is
Palier 1. Keeping rewards fixed isolates whether reset-distribution alone fixes
detachment.

Non-goals (deferred to Palier 1+):
- Explicit probability knob `p` for truck-vs-frontier mix.
- Auto-populated archive `{cell_key -> savestate}` during rollouts.
- Reward changes / removal of hackable bonuses.
- Go-Explore phase-2 robustification (DAgger / backward algorithm) — never.

## Chosen approach

Mirror the existing, proven pattern in `env/battle_env.py` (the Fighter env),
which already takes `initial_states: list[bytes]` and picks one uniformly at
random on `reset()`. No new abstraction is invented.

The truck-vs-frontier mix is controlled purely by **which files sit in the
states directory** (uniform draw), not by a separate knob. `states/initial.state`
is always included, so the early game stays in the mix and the change degrades
gracefully to current behaviour when no frontier states exist yet.

## Components

Three files touched, all outside the `feat/p4-heal-mode` file set.

### `env/pokemon_env.py`
- Change the constructor parameter `initial_state: bytes` to
  `initial_states: list[bytes]`; store the list.
- Guard: empty list raises `ValueError` (mirror `battle_env`).
- `reset()`: `idx = int(self.np_random.integers(len(self._initial_states)))`
  then `self.emulator.load_state(self._initial_states[idx])`.
- No backward-compat shim: `agent/train.py` is the only production caller and is
  updated in the same change.

### `agent/train.py`
- Replace the single `STATE_PATH = states/initial.state` load with: read
  `states/initial.state` PLUS every `states/explorer/*.state`, into a list.
- Reuse a `load_states`-style helper (already present in `train_fighter.py`).
- `initial.state` is always in the list; if `states/explorer/` is missing or
  empty, training still runs (uniform over just the truck = current behaviour).
- Pass the list to every `PokemonEmeraldEnv` (all parallel envs share the same
  list; each env's `np_random` picks independently).

### `tools/capture_frontier.py` (new)
- Modeled on `tools/capture_open_map.py`.
- Drives the trained Explorer checkpoint (`checkpoints/ppo_emerald_final`) from
  `states/initial.state`, stepping the emulator.
- Uses the `milestones.py` tracker; on each **newly** reached milestone (e.g.
  `enter_rival_house`, `meet_rival`, `route_101`), saves a savestate to
  `states/explorer/<milestone>.state`.
- One-shot tool, run with cwd = main repo (checkpoints/roms/states live there,
  all gitignored). Bounded by a max-steps budget (no `while True`).

## Data flow

```
capture_frontier.py  --(trained checkpoint)-->  states/explorer/*.state
                                                        |
agent/train.py  reads  states/initial.state + states/explorer/*.state  --> list[bytes]
                                                        |
PokemonEmeraldEnv(initial_states=list)  -->  reset() draws idx uniformly  -->  load_state
```

`states/` is fully gitignored; `states/explorer/` holds local artifacts only.
Nothing new is committed to the repo besides code + this spec + the plan.

## Testing

- **Unit (no ROM)** — mirror `test_battle_env.py`:
  - env built with a `FakeEmulator` + two distinct fake state blobs; `reset()`
    loads one of the two (assert `load_state` was called with a member of the
    list).
  - empty `initial_states` raises `ValueError`.
  - determinism: with a fixed seed, `reset()` selects the expected index.
- **Smoke (ROM-gated)** — build the env with real `[initial.state,
  open_map.state]` bytes; `reset()` returns a valid observation without crash.
- No test for the `train.py` loop (consistent with existing trainers).

Run tests (from this worktree, venv/roms live in the main repo):
`PYTHONPATH=/Users/_eloi/Projets/Emu-go-explore POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`

## Experiment protocol (runbook, not code)

1. Run `capture_frontier.py` to populate `states/explorer/`.
2. Resume a checkpoint; train +~1.5M steps with the multi-reset env.
3. Jalon eval (10 stochastic episodes, `max_steps=4096`, dedup milestones by
   set) from `states/initial.state`.
4. **Go/no-go (A first):** success = `route_101` / `meet_rival` hold or improve
   and do NOT collapse as before, compared against the documented baseline
   (route_101 9/10, meet_rival 10/10 at ~9-10.5M, then 0/10). If the result is
   ambiguous, Palier 1 runs the rigorous A/B (control mono-reset vs treatment
   multi-reset from the same checkpoint, same budget).

## Risks & notes

- If `capture_frontier.py`'s checkpoint cannot reach a given milestone
  reliably, that frontier state is simply absent — training degrades to the
  states it did capture (+ truck). Acceptable for Palier 0.
- The uniform draw means truck frequency = `1 / (1 + n_frontier)`. With the
  early game already well learned, a low truck frequency is fine; if the agent
  starts forgetting the intro, add truck copies to the directory (still no code
  change). Revisit with knob `p` only in Palier 1 if needed.
