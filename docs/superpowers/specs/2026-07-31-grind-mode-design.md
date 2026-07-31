# Grind mode — design (P4 étape 3)

**Date:** 2026-07-31
**Feature:** `execute_order(mode="grind")` — the Explorer's 2nd "savoir-reconnaître" skill: recognize grass by effect, travel to a known grass cell, and walk in it until a wild battle starts.

---

## Purpose

This is the mirror of heal mode (P4 étape 2), applied to the second recognition case: **grass**.

- **Heal** recognized a place *by its effect*: "my party HP refilled to full here" → this is a healing spot.
- **Grind** recognizes a place *by its effect*: "a wild battle started here" → this is grass.

Both share the same honest chicken-and-egg: a place is only learned **after** experiencing the effect there. Reading the tile identity directly (via `MetatileBehavior` RAM) is deferred (Façon B) — and blocked anyway, because the `tile_behavior` RAM address on BPEF is not probed yet (`world_reader` returns `None`).

The Strategist gives a **pure intention** ("grind"), never a place name. It is the Explorer's job to know WHERE grass is.

### Scope (deliberately small — one bite)

This bite delivers ONLY:
1. Recognizing grass by effect (a wild battle started here → remember the cell under label `"has_grass"`).
2. Generalizing the MapMemory recognition store to be label-keyed (fulfils the earlier stated intent: "on part du soin, avec l'intention de généraliser au 2e").
3. Executing `mode="grind"`: travel to a known grass cell, walk until an encounter starts, then stop.

This bite does NOT deliver (later bites):
- Wiring the Fighter to win the battle that grind triggers.
- The grind level-goal loop (grind → win → repeat until team is strong enough).
- Active search for unknown grass (or healing) spots.
- Wiring the Strategist to emit real `Order` objects.
- Nearest-spot selection (still `spots[0]`).
- The `tile_behavior` RAM probe (Façon B, ledges).

---

## Architecture

Six touch points, mirroring heal mode. Two of them (MapMemory generalization, WorldReader passthrough) are shared infrastructure; the rest are grind-specific.

### 1. `env/world_reader.py` — `in_battle()` passthrough

`WorldReader` is the single reader object passed everywhere (`execute_order`/`travel_to`/`navigate_to`/`map_map`). It already passes `party_hp()` through to its `EmeraldReader`. To also detect a battle start, `WorldReader` needs a `BattleReader` — which needs the same raw read callable that `EmeraldReader` gets.

**Constructor change (small refactor):** today `WorldReader(reader: EmeraldReader)` and every call site does `WorldReader(EmeraldReader(emu.read_bytes))`. Change `WorldReader` to take the raw read callable and own both sub-readers:

```python
from env.game_state import BattleReader, EmeraldReader, ReadFn

class WorldReader:
    def __init__(self, read: ReadFn) -> None:
        self._reader = EmeraldReader(read)
        self._battle = BattleReader(read)
    # ... snapshot(), party_hp() unchanged (still delegate to self._reader)

    def in_battle(self) -> bool:
        """True while a wild/trainer battle is active."""
        return self._battle.battle_state().in_battle
```

`BattleReader.in_battle` is NOT a standalone attribute — it lives on `BattleReader.battle_state().in_battle` (`flags != 0 AND opp.max_hp > 0`). The 6 call sites (`tools/capture_open_map.py`, `tests/test_live_navigator_rom.py`, `tests/test_world_reader.py`, `tests/test_world_surveyor_rom.py`, `tests/test_map_traveler_rom.py`, `tests/test_map_explorer_rom.py`) change from `WorldReader(EmeraldReader(x))` to `WorldReader(x)`. This makes `WorldReader` own its perception sub-readers instead of receiving a half-built one — cleaner, no private-field access.

**Coupling introduced (documented):** once `memory` is passed to `navigate_to`/`map_map`, the `reader` must expose BOTH `party_hp()` (heal) and `in_battle()` (grass). In production the reader is always `WorldReader` (has both). In tests, the fakes gain an always-`False` `in_battle()` stub (keeps the `EncounterWatcher` silent), exactly like the always-full `party_hp()` stub.

### 2. `env/encounter_detector.py` (NEW) — edge detector

Pure module, no ROM, exact structural copy of `heal_detector.HealWatcher`:

```python
from __future__ import annotations


class EncounterWatcher:
    """Fires once on the step where a wild battle transitions from absent to present."""

    def __init__(self) -> None:
        self._was_in_battle = True  # optimistic so an already-in-battle first read isn't a spurious start

    def observe(self, in_battle: bool) -> bool:
        started = in_battle and not self._was_in_battle
        self._was_in_battle = in_battle
        return started
```

Reused twice: to LEARN grass (during movement) and to KNOW when grind's walk-loop has triggered a battle.

> **Design note — why a separate module/watcher rather than reusing HealWatcher.**
> The two watchers observe *different signals* (HP for heal, battle flag for grass) and hold *different state* (`_was_full` vs `_was_in_battle`). They are structurally identical but semantically distinct. Keeping them separate is honest and avoids a premature abstraction. (If a third recognition case appears with the same shape, a generic `EdgeWatcher` could be extracted then — YAGNI for now.)

### 3. `env/map_memory.py` — generalize the recognition store to label-keyed cells

Today heal cells live in `self._healing_cells: dict[map_id, cell]` with `healing_spots()`. Generalize to a label-keyed store so grass reuses the same machinery:

```python
# in __init__
self._labeled_cells: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}

# in observe()
if event.healed:
    node.labels.add("healing_spot")
    self._labeled_cells.setdefault("healing_spot", {})[snapshot.map_id] = snapshot.pos
if event.encounter_started:
    node.labels.add("has_grass")
    self._labeled_cells.setdefault("has_grass", {})[snapshot.map_id] = snapshot.pos

def cells_labeled(self, label: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All (map_id, cell) pairs remembered under the given recognition label."""
    return [(map_id, cell) for map_id, cell in self._labeled_cells.get(label, {}).items()]

def healing_spots(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Shortcut for cells_labeled('healing_spot')."""
    return self.cells_labeled("healing_spot")
```

`WorldEvent` already has `encounter_started: bool` (it was seeded in P1 and already sets the `"has_grass"` label). This change adds the **cell** storage under `"has_grass"`, last-write-wins per map (same semantics as healing cells). `healing_spots()` keeps working unchanged (now a shortcut) — existing heal tests stay green.

### 4. `env/live_navigator.py` — learn grass during movement

`navigate_to` already instantiates a `HealWatcher` and, when `memory` is set, calls `memory.observe(before, WorldEvent(healed=True))` on the not-full→full edge. Add a parallel `EncounterWatcher`:

```python
# before the loop
heal_watcher = HealWatcher()
enc_watcher = EncounterWatcher()

# inside the loop, after the `before is None` guard, before the arrived check
if memory is not None:
    if heal_watcher.observe(reader.party_hp()):
        memory.observe(before, WorldEvent(healed=True))
    if enc_watcher.observe(reader.in_battle()):
        memory.observe(before, WorldEvent(encounter_started=True))
```

Order note: like heal, an encounter that starts on the arrival tick is still learned before the `"arrived"` return.

### 5. `env/map_explorer.py` — learn grass while wandering

This is where wild battles actually happen (the Explorer wanders a whole map in `map_map`). Wire the same `EncounterWatcher` into the `map_map` loop. The loop already fetches `here = snapshot_settled(reader)` and `continue`s when it is `None`, so the watcher check sits right after the valid-snapshot point (`reached.add(here.pos)`):

```python
# before the loop
enc_watcher = EncounterWatcher()

# inside the loop, right after `reached.add(here.pos)` (here is a valid snapshot, on target_map)
if enc_watcher.observe(reader.in_battle()):
    memory.observe(here, WorldEvent(encounter_started=True))
```

`map_map` already receives `memory` (it records portals). This wiring means grass gets recognized as a natural side-effect of cartography. (Heal was NOT wired into `map_map` — you don't heal while mapping — but grass is exactly the thing mapping stumbles into.)

**Why this is enough:** a wild battle is an overlay, not a map change — the player's overworld position (SaveBlock1) stays readable and unchanged during the fight, so `snapshot_settled` still returns a valid `here` on `target_map`. The probe step that walks into grass triggers the battle; the watcher fires on the *next* iteration's `in_battle()` edge and records against `here.pos` (the grass cell). No None-skip special-casing is needed — the existing `continue` on `None` already handles the rare relocating tick, and the next battle on that cell would catch it anyway.

### 6. `env/orders.py` — fill the `grind` branch

`execute_order` currently returns `"not_implemented"` for `mode == "grind"`. Replace with a `_execute_grind`, mirror of `_execute_heal`:

```python
def _execute_grind(emulator, reader, memory, wallmap, max_hops=20) -> str:
    """Travel to a known grass cell, then walk in it until a wild battle starts."""
    spots = memory.cells_labeled("has_grass")
    if not spots:
        return "no_grass_spot_known"
    goal_map, goal_cell = spots[0]  # v1: first known spot (nearest-choice is later)
    outcome = travel_to(emulator, reader, memory, wallmap, goal_map, goal_cell, max_hops=max_hops)
    if outcome != "arrived":
        return outcome  # pass-through: unknown_route / unreachable / lost / timeout
    return _walk_until_encounter(emulator, reader)
```

Signature mirrors `_execute_heal` exactly (no `order` param — grind ignores the destination name, same as heal).

`_walk_until_encounter` = the analogue of `_heal_here`'s "press A until full", but "step around until a battle starts":

```python
GRIND_MAX_STEPS = 60
GRIND_STEP_FRAMES = 24
GRIND_RELEASE_FRAMES = 8

def _walk_until_encounter(emulator, reader) -> str:
    watcher = EncounterWatcher()
    directions = (buttons.KEY_UP, buttons.KEY_DOWN, buttons.KEY_LEFT, buttons.KEY_RIGHT)
    for i in range(GRIND_MAX_STEPS):
        if watcher.observe(reader.in_battle()):
            return "encounter_started"
        emulator.step(directions[i % len(directions)], GRIND_STEP_FRAMES)
        emulator.step(0, GRIND_RELEASE_FRAMES)  # release (GBA debounce)
    return "encounter_started" if reader.in_battle() else "no_encounter"
```

Cycling directions keeps the player treading within the grass patch rather than walking off it (a single held direction would leave the patch). The step/release framing reuses the same GBA debounce timing as the rest of the Explorer.

**Check ordering in `execute_order`:** grind, like heal, ignores the order's destination (pure intention). But heal was moved *before* destination resolution specifically because heal ignores destination. Grind does the same — put the `mode == "grind"` branch alongside heal, before `DESTINATIONS.get`. So the head of `execute_order` becomes:

```python
if order.mode == "heal":
    return _execute_heal(emulator, reader, memory, wallmap, max_hops=max_hops)
if order.mode == "grind":
    return _execute_grind(emulator, reader, memory, wallmap, max_hops=max_hops)
dest = DESTINATIONS.get(order.destination)
if dest is None:
    return "unknown_destination"
# ... advance path
```

Since grind now acts, the `if order.mode != "advance": return "not_implemented"` line disappears — all three modes (advance/heal/grind) are handled. The `execute_order` docstring drops `"not_implemented"` and adds grind's outcomes.

---

## Outcome contract — `execute_order(mode="grind")`

| Outcome | Meaning |
|---|---|
| `no_grass_spot_known` | No cell has ever been labeled `"has_grass"` — nothing to grind on |
| `unknown_route` / `unreachable` / `lost` / `timeout` | Travel to the grass cell failed (pass-through from `travel_to`) |
| `encounter_started` | Reached the grass and a wild battle began — grind's job for this bite is done |
| `no_encounter` | Reached the grass but no battle started within the step budget |

---

## Data flow

```
Strategist ── Order(mode="grind") ──▶ execute_order
                                          │
                                          ├─ memory.cells_labeled("has_grass") empty? ─▶ "no_grass_spot_known"
                                          │
                                          ├─ travel_to(grass cell) ── fail ─▶ pass-through
                                          │
                                          └─ _walk_until_encounter
                                                 │ step around, EncounterWatcher on reader.in_battle()
                                                 ├─ battle starts ─▶ "encounter_started"
                                                 └─ budget out ────▶ "no_encounter"

LEARNING (separate, passive):
navigate_to / map_map ── each tick ── EncounterWatcher(reader.in_battle())
                                          └─ edge fires ─▶ memory.observe(pos, WorldEvent(encounter_started=True))
                                                              └─ stores cell under "has_grass"
```

---

## Testing

All unit tests are pure (no ROM), mirroring heal mode's test structure.

- **`tests/test_encounter_detector.py`** (NEW): `EncounterWatcher` fires on absent→present edge; silent when already in battle on first read; silent on present→absent; fires again after a battle ends and a new one starts.
- **`tests/test_map_memory.py`** (+): `cells_labeled("has_grass")` remembers the cell on `encounter_started`; empty without any encounter; last-write-wins per map; `healing_spots()` still equals `cells_labeled("healing_spot")` (shortcut regression); the two labels don't cross-contaminate.
- **`tests/test_world_reader.py`** (+): `in_battle()` passthrough returns the battle reader's flag (via a fake reader).
- **`tests/test_orders.py`** (+): grind with no grass spot → `no_grass_spot_known`; grind with a known grass cell → `encounter_started` (via a fake `GrassWorld` that flips `in_battle` True after N steps); grind where the grass cell is unreachable → travel pass-through; grind never battling → `no_encounter`; grind ignores the order destination (unknown destination name still grinds).
- **`tests/test_live_navigator.py`** (+): learns a grass cell on the in-battle edge during movement (via an `EncounterFakeWorld`); regression: `memory=None` unchanged.
- **`tests/test_map_explorer.py`** (+): learns a grass cell when a battle fires during `map_map` wandering.
- **Fakes updated:** every fake reader passed with `memory` gains an always-`False` `in_battle()` stub (keeps the watcher silent), alongside the existing always-full `party_hp()` stub — `MultiMapWorld`, `NamedWorld`, `WorldGrid`, `FakeWorld`, `ExploreWorld`.

No ROM smoke this bite (deferred): would need a savestate "standing on/near grass with a party" and a deterministic encounter, which we don't have. Consistent with heal mode deferring its ROM smoke.

---

## Error handling / edge cases

- **Empty party during `_walk_until_encounter`:** no encounters can start with an empty party in practice; if `in_battle()` never flips, the bounded loop returns `no_encounter`. No special-casing.
- **Battle already active when grind starts:** the `EncounterWatcher` is fresh (`_was_in_battle=True`), so an already-active battle is NOT counted as a new start — the loop would step and eventually return `no_encounter`. Acceptable: grind is meant to *trigger* a battle from the overworld, not to react to one already running.
- **`snapshot is None` while learning in `map_map`:** skip recording that tick (unknown cell), log-and-continue. Next battle on the cell catches it.
- **Bounded loops only** (code-safety #2): `_walk_until_encounter` bounded by `GRIND_MAX_STEPS`; watchers are O(1) per tick; no `while True`.

---

## Files

- `env/world_reader.py` — MODIFY: `in_battle()` passthrough + `BattleReader` wiring.
- `env/encounter_detector.py` — CREATE: `EncounterWatcher`.
- `env/map_memory.py` — MODIFY: label-keyed `_labeled_cells`, `cells_labeled()`, `healing_spots()` shortcut, grass cell storage.
- `env/live_navigator.py` — MODIFY: `EncounterWatcher` learning branch.
- `env/map_explorer.py` — MODIFY: `EncounterWatcher` learning during `map_map`.
- `env/orders.py` — MODIFY: `_execute_grind` + `_walk_until_encounter`, grind branch ordering.
- Tests as listed above.
