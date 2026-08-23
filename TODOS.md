# TODOS

Deferred informational findings from the 2026-08-21 pre-push review (commit `1ed7052`).
Each was intentionally skipped (logged in `~/.gstack/projects/Emu/main-reviews.jsonl`);
none blocks the current drivers. Re-check the fingerprint before fixing — code may
have moved since.

## Review findings (informational, 2026-08-21)

- [ ] **I1 — Stale snapshot replan in `_precision_walk_to`** (`env/map_traveler.py`)
      The walk replans from a snapshot taken before the last move; a mid-walk
      interruption (battle, scripted event) can leave it steering from stale
      coordinates. Re-snapshot after each interruption before replanning.

- [ ] **I2 — `reach_map` burns budget on None snapshots** (`env/map_traveler.py`)
      When `_snapshot_settled` returns None (save-block relocation window), the
      loop consumes a hop attempt without acting. Retry the snapshot a few frames
      later instead of spending budget.

- [ ] **I3 — `_pocket_has` non-atomic SB1/SB2 reads** (`env/game_state.py`)
      Bag pointer (SaveBlock1) and securityKey (SaveBlock2) are read in separate
      calls; a relocation between the two decodes quantities with a mismatched
      key. Read both under one settled check, or verify pointers unchanged after.

- [ ] **I4 — `_Reader` adapter attribute-resolution order** (`tests/test_shoes_leg_rom.py`, also `test_pokedex_return_rom.py`)
      The duck-typed wrapper resolves `world` before `reader`; a name collision
      between WorldReader and EmeraldReader would silently pick the wrong source.
      Assert disjoint attribute sets, or promote a real composite reader class.

- [ ] **I5 — `_verify_control` wastes cycles on None reads** (`env/campaign.py`)
      Each None `player_state()` still consumes a control-check cycle. Same
      pattern as I2: retry the read instead of burning the bounded budget.

- [ ] **I6 — Magic numbers `town_state == 4` / `lab_state == 5`** (`env/campaign.py`)
      Name them (`_TOWN_STATE_SHOES_DONE = 4`, `_LAB_STATE_CUTSCENE_DONE = 5`)
      and add a unit test for the None case (var read during relocation).

- [x] **I7 — Map constants duplicated** (`env/campaign.py` vs `env/milestones.py`)
      LITTLEROOT / ROUTE_101 / OLDALE / ROUTE_103 / LAB tuples live in two
      modules. Fixed: single source `env/maps.py` imported by both (the original
      finding said `agent/milestones.py`; the real duplicate was `env/milestones.py`).

- [ ] **I8 — `control_returns` bare literals** (`tests/conftest.py`)
      Press/settle frame counts and cycle bound are inline literals; name them so
      the anti-false-lock pin stays legible and tunable.

- [ ] **I9 — Untested branches in map_traveler** (`env/map_traveler.py`)
      `_precision_walk_to` failure paths, `_cross_up_warp` no-doorstep case, and
      `cross_scripted_npc` timeout branch have no unit coverage. Add fakes-based
      tests mirroring the hop_via_explore suite.

- [ ] **I10 — Opaque `no_crossing` status** (`env/map_traveler.py`, `cross_scripted_npc`)
      The status does not say WHICH sub-step failed (walk, face, dialogue,
      push-through). Split into distinct statuses or log the failing step.

- [ ] **I11 — A-spam has no early exit** (`env/orders.py`, `_advance_story_dialogue`)
      The loop always completes its press/settle pair even when the predicate
      flipped mid-cycle. Check the predicate after the press too; halves worst-case
      cutscene time.
