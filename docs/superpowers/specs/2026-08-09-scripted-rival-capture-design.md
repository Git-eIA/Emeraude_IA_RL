# Scripted Rival Capture — Design

**Date:** 2026-08-09
**Status:** Approved (brainstorming)
**Scope:** Rewrite the throwaway tool `tools/capture_trainer_battle.py` so it produces the
`states/trainer_battle.state` artifact by spawning + engaging the route_103 rival.

## Goal

Produce `states/trainer_battle.state`: a savestate the game is sitting inside the
route_103 rival trainer battle (`is_trainer_battle()` True). This single artifact makes two
currently-skipped ROM smokes **load-bearing**:

- `tests/test_battle_player_rom.py::test_fighter_wins_a_real_trainer_battle`
- `tests/test_north_rival_milestones_rom.py` (route_103 latch / beat_rival half)

## Background — probe findings (this is fact, read live)

The multi-session "rival is unreachable on route_103" saga was a red herring. Ground-truth
probes settled it:

- **Reach is solved.** `states/route_103_reached.state` sits on route_103 (map `(0,18)`) at
  `(10,21)`, reached autonomously by ledge-aware A* (`explore_grid`/`navigate_grid`), zero
  attrition. Physical reach of route_103 is NOT the blocker.
- **The rival is HIDDEN by a spawn flag, not absent.** Reading the ROM map-header chain
  (`gMapHeader @0x02037318` → `.events +0x04` → `MapEvents{objectEventCount u8@0x00,
  objectEvents ptr@0x04}` → `ObjectEventTemplate[]` stride `0x18`) lists all 20 objects the
  map declares. Object 10 is the rival:

  ```
  obj[10] gfx=0x40 tile=(7,3) trainerType=0 flagId=0x0382 flag=True -> HIDDEN
  ```

  Its hide-flag `0x0382` is SET in SaveBlock1, so the object never spawns. The legitimate
  clear is Birch's post-lab "go find my kid" dialog — which `route_103_reached.state` has not
  performed. This tool clears the flag directly in RAM (cheat-spawn) purely to mint the
  artifact; the legit scripted campaign (Option B) is a separate, deferred project.

- **Heal spot** (for the attrition fallback, read from `SaveBlock1.lastHealLocation @0x1C`):
  map `(1,1)`, cell `(4,2)`.

- **Grass misclassification hazard.** Some route_103 tall-grass metatiles decode to
  `behavior=0xff` (undecoded) → `classify_at` falls through to `FREE` → the grass-avoidance
  blocked-set (which only blocks edges adjacent to `TileKind.GRASS`) leaks → wilds can fire on
  tiles the planner believes are safe sand. The mono-mon L5 party can whiteout on the trip.
  This is why the tool needs an attrition safety net.

## Approach (approved)

Rewrite `tools/capture_trainer_battle.py` as a self-contained throwaway driver. It is a tool,
not library code — it runs in the main repo (`/Users/_eloi/Projets/Emu`) where the ROM,
states, and Fighter checkpoint live.

Flow:

1. **Load** `states/route_103_reached.state`; build `WorldReader` + `BattleReader`; load the
   Fighter (`PPO.load("checkpoints/fighter/ppo_fighter_final.zip", device="cpu")`) and
   `make_move_type_fn(emu)`.
2. **Clear** hide-flag `0x0382` in RAM via a `rawWrite8` helper
   (`emu._core._core.rawWrite8(emu._core._core, addr, -1, value)`;
   `addr = SaveBlock1Ptr + 0x1270 + flagId//8`).
3. **Force a map reload** so object events respawn with the flag now clear: leave route_103
   south into Oldale `(0,10)`, then re-enter north back onto route_103 `(0,18)`. The rival now
   spawns at `(7,3)`. The south/north crossing is done by **A\* navigation to the connection
   edge** (`plan_path_grid` toward the edge cell), not by holding a single direction — holding
   DOWN from `(10,21)` can stall against a wall/ledge depending on the column. Both crossings
   are bounded (give up after a fixed step budget and report).
4. **Navigate** to a standable cell 4-adjacent to `(7,3)` using shortest A* (`plan_path_grid`),
   grass allowed. Grass is allowed (not fully blocked) because the northward traverse to the
   `(7,3)` area was observed once at **0 wilds** (probe reached `(4,2)`, 3 tiles away, in 17
   presses) and the attrition net (step 6) covers the RNG risk — not because the trip is short
   (it is the full north traverse from the reload edge). `handle_battle_interruption` lets the
   Fighter clear any wild triggered en route.
5. **Talk:** face `(7,3)`, then **spam A through any pre-battle dialogue** until either
   `battle_starting()` is True or a bounded A-press budget is exhausted. The rival is a
   scripted event battle (`trainerType=0`), so talking likely runs dialogue before the battle
   engages; a single A-press would miss it. On `battle_starting()` AND `is_trainer_battle()` →
   write `states/trainer_battle.state`, exit 0. Else exit 1 with diagnostics
   (`starting`, `trainer`, `in_battle`, `pos`).
6. **Attrition safety net (approved: nav-first + teleport-fallback).** Run real navigation
   first. If navigation ends in a whiteout (`handle_battle_interruption` returns a losing
   outcome), fall back to a RAM-teleport of the player's coordinates to a cell adjacent to
   `(7,3)`, then talk. The teleport is a logged, explicit last resort — never the default
   path — so the artifact is honest about how it was produced. **The teleport mechanism is
   not yet proven:** writing only `gObjectEvents[0].currentCoords` may not relocate the player
   cleanly (camera / collision / `SaveBlock1.pos` / `gPlayerAvatar` can desync). The
   implementation plan must either pin down the exact set of addresses to write, or replace
   the fallback with **retry the reload+nav with fresh RNG** if a clean teleport proves
   infeasible.

## Risks and open questions

- **Does talking to obj[10] actually start a battle?** No probe has confirmed it. Spawning the
  sprite (flag clear + reload) is not the same as the talk-triggers-battle script being active
  — the rival battle can be gated by additional story flags. If the sprite spawns but talking
  never yields `battle_starting()`, **approach A is dead** and the legit story-flag path
  (Option B: Birch's post-lab dialog) becomes mandatory. In that case the tool exits 1 with
  the `starting=False` diagnostic and we escalate to Option B rather than patching this tool.
  Success of this tool == `is_trainer_battle()` True, which simultaneously confirms this open
  question.

## Non-goals

- No legit story flag-clear (Birch dialog) — that is the deferred scripted-campaign Option B.
- No grind / no reader grass-misclassification fix — deferred projects.
- No changes to library code, milestones, or the two consuming smokes themselves (they only
  gain a real artifact to load).

## Files

- **Rewrite:** `tools/capture_trainer_battle.py` (throwaway driver, runs in main repo).
- **Produces:** `states/trainer_battle.state` (gitignored artifact).
- **Unblocks (no edit):** `tests/test_battle_player_rom.py`,
  `tests/test_north_rival_milestones_rom.py`.

## Verification

- Run the tool in the main repo with `POKEMON_EMERALD_ROM` set; assert it exits 0 and writes
  `states/trainer_battle.state`.
- With the artifact present, both ROM smokes stop skipping and pass:
  `test_fighter_wins_a_real_trainer_battle` (Fighter beats the real trainer, `in_battle` False
  after) and the route_103 half of `test_north_rival_milestones_rom.py`.
