# Go-Explore Palier 1 — cut the hackable tile bonus

**Date:** 2026-07-31

## Goal

Test whether removing the per-tile exploration bonus cures the tile-farming
detachment regression (route_101 collapsing from 9/10 to 0/10 around 10.5M
steps). Palier 0 (reset-state distribution) held this bonus fixed and did not
cure the collapse; Palier 1 changes exactly this one variable.

## Hypothesis

`NEW_TILE_REWARD = 0.5` is farmable: a large town has hundreds of distinct
tiles, so wandering to collect fresh `+0.5`s is a steady, low-risk reward stream
that substitutes for the hard sequential milestone chain. Removing it should
make the milestone chain the only positive reward worth chasing, so the policy
stops loitering (`ep_len_mean` should drop) and stops forgetting the chain.

## Change (one variable)

`env/rewards.py`: `NEW_TILE_REWARD` `0.5 -> 0.0`.

Everything else is unchanged:
- `REVISIT_PENALTY = -0.01` kept. With a new tile now paying `0.0`, there is
  nothing to farm; the revisit penalty only discourages pacing back and forth.
  Net effect: new tile pays `-0.02` (time only), revisit pays `-0.03`, so new
  ground is still mildly preferred without being a positive income stream.
- `TIME_PENALTY`, milestone rewards, and level rewards unchanged.

## Test updates (reflect the new design)

- `tests/test_rewards.py::test_new_tile_reward_stays_below_smallest_milestone`
  -> assert `NEW_TILE_REWARD == 0.0` (no exploration bonus).
- `tests/test_env.py::test_moving_to_new_tile_gives_positive_reward`
  -> rename to reflect Palier 1: a new tile pays only `TIME_PENALTY` and still
  pays strictly more than a revisit.

## Experiment (single-variable, cleaner than Palier 0)

Only the reward changes; the reset distribution is held at the baseline recipe.

- Resume from the same checkpoint as the Palier 0 treatment:
  `checkpoints/ppo_emerald_9905920_steps.zip` (9.9M).
- Budget `+1_500_000` steps, `--max-steps 4096`.
- Reset pool = **truck only** (move `states/explorer/` aside so
  `load_initial_states` degrades to truck-only). This isolates the reward change
  as the single variable versus the Palier 0 multi-reset run.
- Final model saved by the trainer under `captures/<run_id>/checkpoints/`, so
  the restored good baseline `checkpoints/ppo_emerald_final.zip` is untouched.

## Go / No-Go

Jalon eval: 10 stochastic episodes from `states/initial.state`, `max_steps=4096`,
dedup milestones by final set.

- **Go**: `meet_rival` / `reach_route_101` hold or improve versus the Palier 0
  treatment collapse (0/10), ideally back toward the 9.9M pre-treatment level
  (meet_rival 8/10, route_101 7/10). Watch `ep_len_mean` drop (less farming).
- **No-Go / ambiguous**: chain still collapses -> the tile bonus was not the
  (only) culprit; run the paired mono-reset control or revisit reward shaping.

## Non-goals

No potential-based shaping, no per-episode cap, no decay schedule, no reward
change beyond the single tile-bonus constant. Those are fallbacks only if this
clean removal is inconclusive.

## Result — NO-GO, and the whole Go-Explore premise was misdiagnosed

Jalon eval (10 stochastic episodes, `max_steps=4096`, from the truck):

| milestone | resume 9.9M | P0 multi-reset (0.5) | P1 cut bonus (0.0) | control (0.5, truck) |
|---|---|---|---|---|
| enter_rival_house | 8/10 | 4/10 | 0/10 | 10/10 |
| meet_rival | 8/10 | 1/10 | 0/10 | 9/10 |
| reach_route_101 | 7/10 | 0/10 | 0/10 | 9/10 |
| starter_obtained | 7/10 | 0/10 | 0/10 | 9/10 |

- **Palier 1 (cut bonus to 0.0): NO-GO, worse than Palier 0.** Every episode
  stalled at `back_outside`, 0/10 on the whole upper chain.
- **The control decides it.** Resuming the *same* 9.9M with the *original*
  recipe (0.5 tile bonus, truck-only) did not collapse — it improved 7/10 -> 9/10
  and finished the chain fast (~2000 steps, not farming to 4096).
- Therefore: (1) "continued training degrades" is false — the original recipe
  improves; (2) both interventions were net-harmful (multi-reset hurt, cutting
  the bonus destroyed exploration); (3) the tile bonus is load-bearing, not
  hackable in practice, and the feared tile-farming does not actually dominate.

**Action:** the Palier 1 commit (`9643842`) is reverted — `NEW_TILE_REWARD`
restored to `0.5` and the tests restored. The control produced a fresh strong
Explorer at `captures/control05/checkpoints/ppo_emerald_final` (9/10). The good
`checkpoints/ppo_emerald_final.zip` (10/10 baseline) remains restored. Reset pool
stays truck-only (`states/explorer/` kept aside as `states/explorer_palier0_bak/`).
