# M5 — Story Milestones & Starter Objective (Design)

**Date:** 2026-07-22
**Builds on:** `2026-07-22-pokemon-emerald-rl-design.md` (M1-M4 pipeline, merged)

## Long-term target architecture (north star, NOT this iteration)

Hierarchical RL with three brains, all eventually learned:

```
STRATEGIST  — global strategy: current goal, which Pokémon to level,
              train-vs-progress decisions. Rule-based first, learned later.
    │ assigns goals / shapes worker rewards
    ├── EXPLORER — overworld: navigation, story progression, dialogues.
    │             (= current PPO agent + milestone rewards)
    └── FIGHTER  — battles: move choice, switching, items.
                  (scripted first, learned later)
Switch signal: "in battle" flag read from RAM.
Shared state: RAM readers (party, levels, HP, position, story flags).
```

Build order: Explorer (this iteration) → battle detection + full party
readers → Fighter (scripted, then RL) → Strategist (rules, then RL).
Never make two stages learn at once before the lower one is stable.

## This iteration: the agent must obtain the starter

Starting point: `states/initial.state` (Littleroot Town outdoors, clock/mom
intro already done, party empty).

Success path the agent must discover: exit town north → Route 101 → Birch
cutscene → navigate bag menu → pick starter (→ first battle, unlosable
in practice by pressing A).

### Reward design (reward shaping)

Hierarchy of scale — tile ≪ minor milestone ≪ major milestone:

| Signal | Amount | Condition | Notes |
|---|---|---|---|
| Exploration | +1 | new (map_group, map_num, x, y) | existing |
| Reach Route 101 | +20 | map becomes Route 101 | first waypoint |
| **Starter obtained** | **+100** | party_count 0 → 1 | main objective |
| Party level sum | +5 / level | sum of party levels increases | combat-learning base, one-time per level gained |

Rules learned from RL practice:
- **One-time per episode**: each milestone pays once, then never again
  (prevents farming loops). Story flags/party growth are monotonic, which
  makes this safe.
- **No step penalty** for now (risk of discouraging exploration early).
- Milestone conditions are read from RAM state only (no pixel heuristics).

### Episode termination

- `terminated=True` when the starter is obtained (milestone reached).
  One objective = one episode = one clear success metric.
- `truncated=True` at max_steps (existing behavior).
- Success metric: % of episodes reaching the starter (visible in
  tensorboard via episode reward: successful episodes end near +100+).

### Components

1. **`env/game_state.py`** — extend:
   - `read_flag(flag_id)` generalized from the badge popcount logic
     (flags array at SaveBlock1+0x1270; badge code refactored on top).
   - `party_levels()` — read levels of party Pokémon from `gPlayerParty`
     (EWRAM 0x020244EC, 100-byte structs, level field unencrypted).
     Exact offsets pinned from pret/pokeemerald during implementation and
     verified against the real save state (BPEF checked like previous
     addresses via pokebot-gen3 symbol tables).
2. **`env/milestones.py`** (new) — `MilestoneTracker`: ordered table of
   `(name, condition, points, terminal)` entries evaluated per step against
   PlayerState; each fires once per episode; `reset()` on episode start.
   Extensible: next iterations only append rows (rival, Pokédex, badge 1…).
3. **`env/pokemon_env.py`** — plug MilestoneTracker into `step()`:
   total reward = exploration + milestones + level delta;
   `terminated` from the terminal milestone; milestones listed in `info`.
4. **`agent/train.py`** — unchanged (hyperparameters revisited only if
   validation run fails).

### Validation

- Unit tests with FakeEmulator: milestone fires once, terminal ends episode,
  level-sum delta, no-regression on existing env tests.
- Real training run (~1-2h, several hundred thousand steps): success if the
  agent obtains the starter in a growing fraction of episodes. If it never
  succeeds, first lever: increase Route-101 waypoint reward or add one
  intermediate waypoint (Birch bag cutscene flag) — not hyperparameter
  tuning first.

## Out of scope (next iterations)

- Battle detection flag + full party/HP/moves readers (Fighter's eyes)
- Scripted then learned Fighter; battle reward shaping (damage, victory)
- Rule-based Strategist; goal-conditioned worker rewards
- Milestones beyond the starter (rival 103, Pokédex, badge 1 chain)
- `tools/watch.py` (watch trained agent live)
