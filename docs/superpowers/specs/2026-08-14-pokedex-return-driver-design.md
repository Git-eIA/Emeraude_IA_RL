# Pokédex Return Driver (A2) — Design

**Date:** 2026-08-14
**Status:** Approved (brainstorming), pending user spec review
**Depends on:** A1 (retail flag constants) merged `e182556` — `has_pokedex()` now reads the
correct bit (`FLAG_SYS_POKEDEX_GET = 0x861`). Without A1 this driver's success assertion is
unobservable.

## Goal

A durable, tested driver that starts from the genuine post-rival save
(`states/post_rival.state`, on route_103), returns to Birch's lab, and delivers the Pokédex —
the Phase 2 objective that B2 had to defer. Success = `has_pokedex()` is `True` after the lab
`GivePokedex` cutscene completes.

## Why this is now achievable (grounded)

The whole "Pokédex NO-GO" saga was two independent problems, both now resolved:

1. **The Oldale → route_101 crux was a scripted NPC (Flora), not a sub-RAM mystery.** Flora
   (gfx `0x69`) stands on the south connection tile at logical `(11,19)`. Standing at `(10,19)`
   facing EAST + A-spam opens her dialogue; then a single DOWN crosses into route_101. Proven
   end-to-end by throwaway probes (`tools/probe_oldale_flora_cross.py`,
   `tools/probe_lab_arming.py`): post_rival → Flora gate → continuous descent → lab arrival
   `(6,12)`, no reload, no RAM hack.

2. **`has_pokedex()` read a clear bit (A1, already merged).** `post_rival.state` is *already
   armed*: `VAR_BIRCH_LAB_STATE (0x4084) == 4`. The lab OnFrame table fires
   `map_script_2 VAR_BIRCH_LAB_STATE, 4, GivePokedexEvent` — a pure VAR gate, no flag guard.
   The probe confirmed: on lab entry the cutscene arms, the player auto-walks
   (`lockall applymovement (6,12)→(6,5)` — fires on IDLE, no input), A-spam advances the
   dialogue, and `VAR_BIRCH_LAB_STATE` steps `4 → 5` at ~A#185 (`GivePokedex` complete). The
   Pokédex *is* delivered; A1 makes the detector see it.

This driver hardens that proven probe sequence into durable, reviewed code with a load-bearing
gated ROM test. No new emulator/env capability is introduced — it wires primitives that already
exist plus one new crossing primitive.

## Architecture

An orchestrator flow in `env/campaign.py` chains existing traversal primitives with **one new
primitive** in `env/map_traveler.py`. Two `reach_map` greedy descents bracket an explicit,
scripted-NPC crossing:

```
post_rival (route_103)
  │  _heal_party
  ▼
reach_map(target=OLDALE)        # existing directional descent (_cross_border down)
  │
  ▼
_cross_scripted_npc(OLDALE→…)   # NEW primitive: Flora gate
  │
  ▼
reach_map(target=LAB)           # existing descent (route_101 down, littleroot up)
  │
  ▼
lab GivePokedex cutscene        # idle (OnFrame auto-walk) + bounded A-spam until VAR 4→5
  ▼
has_pokedex() == True
```

**Orchestration choice (locked):** two separate `reach_map` calls with the Flora crossing
explicit *between* them, rather than teaching `reach_map` about the NPC gate. This keeps
`reach_map` generic (border + door crossings only) and isolates the single known scripted-gate
case in the driver. Rejected alternative: coupling `reach_map`'s crossing dispatch to Flora
specifics (more coupling, no reuse payoff for one gate).

## Components

### C1 — `map_traveler._cross_scripted_npc(...)` (new, durable primitive)

Lives beside `_cross_border` / `_cross_up_warp`. Proposed signature:

```python
def _cross_scripted_npc(
    emu, reader, memory, from_map, *,
    stand_tile: tuple[int, int],   # e.g. (10, 19)
    face_dir: str,                 # e.g. "right" — toward the NPC
    cross_dir: str,                # e.g. "down"  — toward the next map
    max_presses: int,
) -> bool:
    ...
```

Behaviour:
1. Precision-walk to `stand_tile` (reuse `_precision_walk_to`). Fail → `False`.
2. Turn to face the NPC: tap `DIRECTION_KEYS[face_dir]` a few frames (no step — walls the
   player, only rotates the sprite).
3. Bounded A-spam to play/clear the NPC dialogue.
4. Push `cross_dir` (stepping) until `snapshot_settled().map_id != from_map`, bounded.
5. Return `True` iff `map_id` changed, else `False`.

**Why `face_dir` and `cross_dir` are distinct params:** Flora is looked at facing EAST but the
crossing is DOWN. Keeping them separate makes the primitive reusable for any NPC-gated
connection, not Flora-only.

### C2 — `campaign.run_pokedex_return(...)` (new orchestrator)

```python
def run_pokedex_return(emu, reader, memory, *, move_type_fn, predict) -> str:
    ...
```

Sequence and short-circuit statuses (no blind retries; explicit bounds everywhere):

| Step | Action | Failure status |
|------|--------|----------------|
| 1 | `_heal_party(emu)` | `heal_failed` |
| 2 | `reach_map(..., OLDALE, ...)` → arrived at Oldale | `hop_failed` |
| 3 | `_cross_scripted_npc(... Oldale→route_101, Flora)` | `flora_no_cross` |
| 4 | `reach_map(..., LAB, ...)` → arrived at lab | `descent_stall` |
| 5 | lab cutscene: idle then bounded A-spam until `VAR_BIRCH_LAB_STATE == 5` and `has_pokedex()` | `pokedex_not_delivered` |
| — | all steps pass | `pokedex_delivered` |

The lab cutscene handler (step 5) may be a small module-level helper (e.g.
`_run_lab_pokedex_cutscene(emu, reader, *, max_presses)`) or inline in `run_pokedex_return`;
the plan decides based on readability. It reads `VAR_BIRCH_LAB_STATE` via
`EmeraldReader._var(sb1, 0x4084)`, idles first (the OnFrame auto-walk needs no input), then
A-spams up to `max_presses` (≥ 400; ~185 observed) until `VAR == 5`.

### C3 — Constants (re-introduce / add)

- `OLDALE = (0, 10)`, `ROUTE_103 = (0, 18)` — re-introduced (B2 deleted them).
- `VAR_BIRCH_LAB_STATE = 0x4084` (pret vars.h; lab OnFrame gate).
- Flora gate constants: `stand_tile`, `face_dir`, `cross_dir` (module-level named constants in
  `campaign.py`, passed into `_cross_scripted_npc`).
- `_RETURN_DIRECTIONS` gains `route_103: "down"` (existing entries `route_101: "down"`,
  `littleroot: "up"` unchanged).

### C4 — Reader adapter

Reuse the exact `_Reader` adapter pattern from `test_phase2_rom.py`: forward attribute lookups
to `WorldReader` then `EmeraldReader`, so `run_pokedex_return` gets `snapshot`/`grid_reader`
navigation *and* `has_pokedex`/`_var`/`party_*` state off one object. The driver takes
`predict` / `move_type_fn` from the Fighter (required by `reach_map` and by wild-battle
handling on the hops).

## Data flow / dependencies

`emu` (GbaEmulator) + `_Reader` adapter + `MapMemory` (portals discovered live) + Fighter
`predict`/`move_type_fn`. No serialized portal graph — `reach_map` discovers borders/doors as
it descends.

## Error handling

Every leg returns a status; the orchestrator returns the first failure's descriptive string.
All loops are explicitly bounded (A-spam presses, cross-direction pushes, reach_map's own
timeout). No `while True`, no blind retry of a failed leg.

## Testing

### T1 — load-bearing gated ROM smoke: `tests/test_pokedex_return_rom.py`

Triple-skip (`POKEMON_EMERALD_ROM` unset / Fighter checkpoint missing /
`states/post_rival.state` missing). Loads post_rival, builds the `_Reader` adapter + Fighter
`predict`/`move_type_fn`, runs `run_pokedex_return`, then asserts:

- `result == "pokedex_delivered"`,
- `reader.has_pokedex() is True`,
- `world.snapshot().map_id == LAB`.

Dumps `states/post_pokedex.state` for downstream phases.

This is the single load-bearing proof of the whole chain. It is slow (~minute, Fighter-driven)
and gated, matching `test_phase2_rom.py` / `test_north_rival_milestones_rom.py`.

### Pure unit coverage

`_cross_scripted_npc` gets pure unit tests over a fake snapshot/emu (no ROM), mirroring the
`_cross_up_warp` / `_cross_border` unit tests already in `tests/test_*`: assert it walks to
`stand_tile`, faces `face_dir`, A-spams, pushes `cross_dir`, and reports `True`/`False` on
map-change / no-change. This keeps the crossing logic TDD-covered independently of the slow ROM
smoke.

## Open questions the plan must close (against the code)

**Q1 — heal sourcing.** `_heal_party(emu)` currently exists only in a throwaway probe
(`tools/probe_return_portals.py`), not in durable `env/` code. The plan must decide: port a
minimal heal helper into `env/` (its own small function, unit-testable), or drop step 1 if the
Fighter reliably survives the hops from a full-HP post_rival party. Do NOT import from
`tools/` in durable code.

**Q2 — route_103 → Oldale leg mechanism.** The "reach route_103" saga crossed this border with
`_hop_via_explore_then_scan` (an `explore_grid` sweep in a throwaway probe), NOT with
`reach_map`/`_cross_border`. The plan must verify, by reading `reach_map` + `_cross_border`,
that a directional descent can cross route_103 → Oldale at all — and that it **stops at Oldale**
without greedily plunging back north into the route_103 connection. If `reach_map` cannot do
this leg cleanly, replace step 2 with a direct `_cross_border(down)` (or port the proven
directional hop) instead of a full `reach_map`. This is the single highest-risk unknown; the
plan's first task should be a gating de-risk probe (STOP if the leg can't be crossed durably),
mirroring B2's Task 1 gate.

## Scope / non-goals

- **Pokédex only.** Running shoes / Poké Balls are out of scope (shoes come from a different
  event; the lab cutscene bundles Pokédex + Balls but only the Pokédex is asserted).
- No changes to the Order `story` mode, its detectors, or `PHASE2_CAMPAIGN` (B2's reach-lab
  campaign stays as-is). This is a standalone driver.
- No new emulator/env capability. No RAM pokes, no savestate splicing (both proven dead).

## Files

- Modify: `env/map_traveler.py` (add `_cross_scripted_npc` + its unit tests' target).
- Modify: `env/campaign.py` (add `run_pokedex_return`, re-introduce `OLDALE`/`ROUTE_103`,
  extend `_RETURN_DIRECTIONS`, `VAR_BIRCH_LAB_STATE`, Flora constants).
- Create: `tests/test_pokedex_return_rom.py` (gated end-to-end smoke).
- Modify: `tests/test_map_traveler*.py` (pure unit tests for `_cross_scripted_npc`).
