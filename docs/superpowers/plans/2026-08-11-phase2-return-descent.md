# Phase 2 Return Descent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `_RETURN_PORTALS` return-navigation with a durable, battle-proof `reach_map` greedy descent so `run_campaign(PHASE2_CAMPAIGN)` walks route_103 -> Littleroot -> lab and delivers the Pokédex + 5 Poké Balls.

**Architecture:** A throwaway probe (Task 1) settles the crux — which crossing primitive actually descends south — and records the real portal chain. `reach_map` (Tasks 2-3) is a per-hop loop that looks up a caller-supplied `direction_by_map` and delegates each crossing to `_cross_toward`, the primitive the probe proved. `run_campaign` (Task 4) dispatches a single reach-home milestone; the gated ROM smoke (Task 5) runs the whole campaign end to end.

**Tech Stack:** Python 3.12, pytest, ruff (line-length 100). Emerald (BPEF) ROM via `emulator.gba.GbaEmulator`. Existing helpers: `env/grid_navigator` (`navigate_grid`, `handle_battle_interruption`, `snapshot_settled`, `probe_step`, `DELTAS`), `env/grid_explorer.explore_grid`, `env/map_memory.MapMemory`.

**Working directory:** git worktree `/Users/_eloi/Projets/Emu-phase2-return-descent`, branch `feat/phase2-return-descent`. Run tests with `.venv/bin/python -m pytest` and lint with `.venv/bin/ruff check` from the MAIN repo `/Users/_eloi/Projets/Emu` (the venv/ROM/checkpoints/states live there); the worktree shares them via the same tree — coordinate paths with the controller.

---

## File Structure

- **Modify `env/map_traveler.py`** — add `reach_map` (per-hop loop) and `_cross_toward` (the crossing primitive). Sits beside `travel_to`; reuses `_snapshot_settled`, `BATTLE_OUTCOMES`, `DELTAS`.
- **Modify `env/campaign.py`** — add `Milestone.reach` field, add `_RETURN_DIRECTIONS`, add the reach dispatch to `run_campaign`, edit `PHASE2_CAMPAIGN` (two advance milestones -> one reach-home milestone), delete `_PortalSeed` / `_RETURN_PORTALS` / `seed_return_portals`.
- **Create `tests/test_reach_map.py`** — pure unit tests of the `reach_map` loop with a scripted fake reader/emulator and a monkeypatched `_cross_toward`.
- **Modify `tests/test_campaign.py`** — add reach-milestone dispatch coverage; remove any `seed_return_portals` test.
- **Modify `tests/test_map_memory.py`** — remove the `seed_return_portals` regression test if it lives here (implementer greps for `seed_return_portals`).
- **Modify `tests/test_phase2_rom.py`** — drop the `seed_return_portals` import/call (portals are discovered live).
- **Rewrite `tools/probe_return_portals.py`** — the Task 1 discovery probe (throwaway, untracked or deleted after facts are frozen).

---

## Task 1: Discovery probe — settle the crossing primitive and record the real chain

**Files:**
- Rewrite: `tools/probe_return_portals.py`

This is a throwaway, ROM-dependent diagnostic — NOT test-driven. Its job is to prove a southbound descent reaches the lab and to print the real portal chain + the crossing mechanism that worked, so Task 3 codes the right primitive.

- [ ] **Step 1: Rewrite the probe to try a directed descent per hop**

Rewrite `tools/probe_return_portals.py` so that, from `states/post_rival.state` with the Fighter and a healed party, it drives a per-hop southbound descent and records what works. It must try, in order, and log which one fires per hop:

1. `explore_grid(emu, reader, memory, cur_map, move_type_fn, predict)` — the proven northbound sweep. If the map changes, that hop crossed via the sweep; record the portal `explore_grid` stored.
2. If the sweep did not leave the map, a **hold-DOWN** fallback: from the clean entry position, press the hop direction (`DOWN` for ledge legs, `UP` for the Littleroot->lab door) repeatedly via `probe_step`, bounded, letting murets carry the player across the border; on transition, `memory.record_portal(...)` the real `(from_cell, direction, to_map, to_cell)`.

Reuse the existing `_heal_party`, `_dump`, and the `RETURN_CHAIN`/`CHAIN_ORDER` constants already in the file. Direction per map: `DOWN` for route_103/Oldale/route_101, `UP` for Littleroot.

- [ ] **Step 2: Run the probe and record the findings**

Run (from the main repo):
```
POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
  .venv/bin/python tools/probe_return_portals.py states/post_rival.state
```
Expected: `REACHED lab = True` and a printed chain of real portals in `_PortalSeed` order, each line noting which primitive (sweep vs hold-DOWN) crossed that hop.

**GATING:** If `REACHED lab = False`, STOP and report to the controller which hop stalled and the grid dump there. Do NOT start Task 2 until the probe reaches the lab. Record in the task output: (a) the per-hop crossing primitive that worked, (b) the real portal chain. Tasks 3 and the ROM smoke depend on these facts.

- [ ] **Step 3: Commit the probe**

```bash
git add tools/probe_return_portals.py
git commit -m "probe: directed southbound descent — record real return portal chain"
```

---

## Task 2: `reach_map` loop (map-hop skeleton, crossing injected)

**Files:**
- Modify: `env/map_traveler.py`
- Test: `tests/test_reach_map.py`

The loop is pure control flow: snapshot, stop on goal, look up direction, delegate the crossing to `_cross_toward`, propagate battle outcomes. Delegating keeps the loop testable without a ROM (tests monkeypatch `_cross_toward`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reach_map.py`:

```python
from __future__ import annotations

import env.map_traveler as mt
from env.map_traveler import reach_map


class _Snap:
    def __init__(self, map_id):
        self.map_id = map_id
        self.pos = (0, 0)


class _Reader:
    """Fake reader whose snapshot() returns the current map in a scripted chain."""

    def __init__(self, maps):
        self._maps = list(maps)
        self._i = 0

    def snapshot(self):
        return _Snap(self._maps[min(self._i, len(self._maps) - 1)])


class _Emu:
    def step(self, *_):
        pass


def test_arrives_when_already_on_goal():
    reader = _Reader([(0, 9)])
    assert reach_map(_Emu(), reader, object(), (0, 9), {}) == "arrived"


def test_descends_chain_to_goal(monkeypatch):
    reader = _Reader([(0, 18), (0, 10), (0, 16), (0, 9)])
    directions = {(0, 18): "down", (0, 10): "down", (0, 16): "down"}

    def fake_cross(emulator, reader_, memory, from_map, direction, **kw):
        reader_._i += 1   # advance to the next map in the chain
        return "crossed"

    monkeypatch.setattr(mt, "_cross_toward", fake_cross)
    assert reach_map(_Emu(), reader, object(), (0, 9), directions) == "arrived"


def test_stalls_on_unknown_map():
    reader = _Reader([(0, 18)])
    assert reach_map(_Emu(), reader, object(), (0, 9), {}) == "stall"


def test_stalls_when_crossing_blocked(monkeypatch):
    reader = _Reader([(0, 18)])
    monkeypatch.setattr(mt, "_cross_toward", lambda *a, **k: "blocked")
    assert reach_map(_Emu(), reader, object(), (0, 9), {(0, 18): "down"}) == "stall"


def test_propagates_battle_outcome(monkeypatch):
    reader = _Reader([(0, 18)])
    monkeypatch.setattr(mt, "_cross_toward", lambda *a, **k: "battle_lost")
    assert reach_map(_Emu(), reader, object(), (0, 9), {(0, 18): "down"}) == "battle_lost"


def test_times_out_when_stuck_on_a_map(monkeypatch):
    reader = _Reader([(0, 18)])   # cross returns "crossed" but map never changes
    monkeypatch.setattr(mt, "_cross_toward", lambda *a, **k: "crossed")
    assert reach_map(_Emu(), reader, object(), (0, 9), {(0, 18): "down"}, max_hops=3) == "timeout"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reach_map.py -v`
Expected: FAIL with `AttributeError`/`ImportError` (`reach_map` / `_cross_toward` not defined).

- [ ] **Step 3: Implement `reach_map` and a placeholder `_cross_toward`**

Add to `env/map_traveler.py` (after `travel_to`):

```python
def reach_map(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    goal_map: tuple[int, int],
    direction_by_map: dict[tuple[int, int], str],
    *,
    move_type_fn: Any = None,
    predict: Any = None,
    max_hops: int = 20,
) -> str:
    """Greedy multi-map descent to goal_map, crossing each map in its
    direction_by_map direction. goal_map is the stop condition only.

    Returns 'arrived' | 'stall' | 'timeout' | one of BATTLE_OUTCOMES.
    'stall' = current map absent from direction_by_map, or the crossing did not
    fire. 'timeout' = hop budget exhausted.
    """
    for _ in range(max_hops):
        here = _snapshot_settled(reader)
        if here is None:
            emulator.step(0, 1)   # relocating; idle a beat and retry
            continue
        if here.map_id == goal_map:
            return "arrived"
        direction = direction_by_map.get(here.map_id)
        if direction is None:
            return "stall"
        crossed = _cross_toward(
            emulator, reader, memory, here.map_id, direction,
            move_type_fn=move_type_fn, predict=predict,
        )
        if crossed in BATTLE_OUTCOMES:
            return crossed
        if crossed != "crossed":
            return "stall"
    return "timeout"


def _cross_toward(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    from_map: tuple[int, int],
    direction: str,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Cross the from_map border in `direction`. Implemented in Task 3.

    Returns 'crossed' | 'blocked' | one of BATTLE_OUTCOMES.
    """
    raise NotImplementedError   # Task 3 fills this per the Task 1 probe
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reach_map.py -v`
Expected: PASS (all 6 — the tests monkeypatch `_cross_toward`, so the `NotImplementedError` body is never hit).

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check env/map_traveler.py tests/test_reach_map.py`
Expected: clean.

```bash
git add env/map_traveler.py tests/test_reach_map.py
git commit -m "feat: reach_map greedy multi-map descent loop (crossing injected via _cross_toward)"
```

---

## Task 3: `_cross_toward` — the real crossing primitive

**Files:**
- Modify: `env/map_traveler.py`
- Test: `tests/test_reach_map.py`

Implement `_cross_toward` per the primitive Task 1's probe proved. The body below is the proven-northbound mechanism (explore_grid sweep, else a directed hold-DOWN fallback); **reconcile it against Task 1's recorded chain** — if the probe showed hold-DOWN is required for a hop, that fallback is the primary path there. The unit test uses a fake that transitions the map after N presses so the loop body is exercised without a ROM; the real ROM behavior is validated by Task 1 and the Task 5 smoke.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reach_map.py`:

```python
class _CrossSnap:
    def __init__(self, map_id, pos):
        self.map_id = map_id
        self.pos = pos


class _CrossReader:
    """Fake: after `presses` DOWN steps the border transitions to `to_map`."""

    def __init__(self, from_map, to_map, presses):
        self._from, self._to, self._presses = from_map, to_map, presses
        self._n = 0
        self.grid_reader = object()

    def snapshot(self):
        crossed = self._n >= self._presses
        return _CrossSnap(self._to if crossed else self._from, (8, 21))


class _CrossEmu:
    def __init__(self, reader):
        self._reader = reader

    def step(self, *_):
        self._reader._n += 1


class _RecordingMemory:
    def __init__(self):
        self.portals = []

    def record_portal(self, *args):
        self.portals.append(args)

    def remember_grid(self, *_):
        pass


def test_cross_toward_holds_down_until_border_transitions(monkeypatch):
    reader = _CrossReader((0, 18), (0, 10), presses=3)
    emu = _CrossEmu(reader)
    memory = _RecordingMemory()

    # Force the hold-DOWN fallback path: make the sweep a no-op that never leaves.
    monkeypatch.setattr(mt, "explore_grid", lambda *a, **k: "complete")
    # probe_step advances the fake by stepping the emu and reports transition on cross.
    monkeypatch.setattr(mt, "probe_step",
                        lambda e, r, before, d: (e.step(0, 1), "transition" if r._n >= r._presses else "blocked")[1])
    monkeypatch.setattr(mt, "snapshot_settled", lambda r: r.snapshot())

    out = mt._cross_toward(emu, reader, memory, (0, 18), "down")
    assert out == "crossed"
    assert memory.portals   # the real crossing was recorded
```

(The exact monkeypatch surface may shift once Task 1 fixes the primitive; keep the assertion `out == "crossed"` and `memory.portals` non-empty as the contract.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reach_map.py::test_cross_toward_holds_down_until_border_transitions -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `_cross_toward`**

Replace the `_cross_toward` body in `env/map_traveler.py`. Add the imports it needs at the top of the module:

```python
from env.grid_explorer import explore_grid
from env.grid_navigator import (
    handle_battle_interruption,
    probe_step,
    snapshot_settled,
)
from env.grid_snapshot import GridSnapshot
from env.map_grid_reader import TileKind

_HOLD_PRESSES = 24   # bounded murets-carry budget per crossing (probe-confirmed)
_STANDABLE = (TileKind.FREE, TileKind.GRASS)
```

```python
def _cross_toward(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    from_map: tuple[int, int],
    direction: str,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Cross the from_map border in `direction`.

    First try the proven border sweep (explore_grid); if it leaves the map, that
    hop crossed. Otherwise fall back to a bounded hold-`direction` that lets murets
    carry the player across (the primitive the Task 1 probe proved for ledge legs).
    Battle-proof: clears a wild encounter via the Fighter before each press.

    Returns 'crossed' | 'blocked' | one of BATTLE_OUTCOMES.
    """
    swept = explore_grid(emulator, reader, memory, from_map,
                         move_type_fn=move_type_fn, predict=predict)
    if swept in BATTLE_OUTCOMES:
        return swept
    after = snapshot_settled(reader)
    if after is not None and after.map_id != from_map:
        return "crossed"   # the sweep auto-crossed a border

    # Hold-`direction` fallback: press the border direction, letting murets carry us.
    dx, dy = DELTAS[direction]
    for _ in range(_HOLD_PRESSES):
        battle = handle_battle_interruption(emulator, reader, move_type_fn, predict)
        if battle is not None:
            return battle
        before = snapshot_settled(reader)
        if before is None:
            continue
        outcome = probe_step(emulator, reader, before, direction)
        landed = snapshot_settled(reader)
        if landed is not None and landed.map_id != from_map:
            memory.record_portal(from_map, before.pos, direction,
                                 landed.map_id, True, landed.pos)
            return "crossed"
    return "blocked"
```

**Reconcile with Task 1:** if the probe showed a hop needs a specific descent column before holding, seed that by navigating to the column first (use the from_cell the probe recorded). Keep the return contract identical.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reach_map.py -v`
Expected: PASS (all tests, including the new crossing test).

- [ ] **Step 5: Lint and commit**

Run: `.venv/bin/ruff check env/map_traveler.py tests/test_reach_map.py`
Expected: clean.

```bash
git add env/map_traveler.py tests/test_reach_map.py
git commit -m "feat: _cross_toward — sweep-then-hold border crossing (battle-proof)"
```

---

## Task 4: Campaign wiring — reach milestone, drop placeholders

**Files:**
- Modify: `env/campaign.py`
- Test: `tests/test_campaign.py`
- (Search/remove) `tests/test_map_memory.py`, `tests/test_phase2_rom.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_campaign.py` (mirror the existing fake `order_fn` / reader pattern already in that file):

```python
def test_reach_milestone_dispatches_reach_map(monkeypatch):
    import env.campaign as camp

    calls = {}

    def fake_reach(emulator, reader, memory, goal_map, direction_by_map, **kw):
        calls["goal"] = goal_map
        calls["dirs"] = direction_by_map
        return "arrived"

    monkeypatch.setattr(camp, "reach_map", fake_reach)
    curriculum = (camp.Milestone("lab", 0, reach=camp.LAB),)
    out = camp.run_campaign(object(), _StubReader(), MapMemory(), curriculum=curriculum)
    assert out == "campaign_complete"
    assert calls["goal"] == camp.LAB
    assert calls["dirs"] == camp._RETURN_DIRECTIONS


def test_reach_milestone_aborts_on_non_arrived(monkeypatch):
    import env.campaign as camp

    monkeypatch.setattr(camp, "reach_map", lambda *a, **k: "stall")
    curriculum = (camp.Milestone("lab", 0, reach=camp.LAB),)
    out = camp.run_campaign(object(), _StubReader(), MapMemory(), curriculum=curriculum)
    assert out == "stall"
```

`_StubReader` must expose whatever `run_campaign` reads before the reach branch (e.g. a `party_levels()` returning a high list so no level_up path is taken); reuse the existing stub in `tests/test_campaign.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_campaign.py -k reach -v`
Expected: FAIL (`Milestone` has no `reach`; `_RETURN_DIRECTIONS` undefined).

- [ ] **Step 3: Edit `env/campaign.py`**

1. Delete the `_PortalSeed` class, `_RETURN_PORTALS`, and `seed_return_portals`.
2. Add the return-direction constant near the map ids:

```python
# Southbound return crossing directions per map (ledge descent, then lab door up).
_RETURN_DIRECTIONS: dict[tuple[int, int], str] = {
    ROUTE_103: "down", OLDALE: "down", ROUTE_101: "down", LITTLEROOT: "up",
}
```

3. Import `reach_map`:

```python
from env.map_traveler import reach_map
```

4. Add the `reach` field to `Milestone` (last, keeps positional compat):

```python
    reach: tuple[int, int] | None = None   # reach mode: reach_map(goal=reach) home
```

5. Replace the first two `PHASE2_CAMPAIGN` entries with one reach-home milestone:

```python
PHASE2_CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("lab", 0, reach=LAB),
    Milestone("lab", 0, story_target=lambda r: r.has_pokedex()),
    Milestone("lab", 0, story_target=lambda r: r.has_item(0x4, 5)),
    Milestone("route_101_shoes", 0, story_target=lambda r: r.has_running_shoes()),
)
```

6. Add the reach dispatch at the head of the `run_campaign` loop, before the `story_target` branch:

```python
        if milestone.reach is not None:
            arrived = reach_map(
                emulator, reader, memory, milestone.reach, _RETURN_DIRECTIONS,
                move_type_fn=move_type_fn, predict=predict, max_hops=max_hops,
            )
            if arrived != "arrived":
                return arrived
            continue
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: PASS. Fix any existing test that referenced `seed_return_portals`.

- [ ] **Step 5: Remove the orphaned `seed_return_portals` test + smoke import**

Grep for `seed_return_portals` across `tests/`:
Run: `.venv/bin/python -m pytest tests/test_map_memory.py -v` after deleting any test that called `seed_return_portals`.
In `tests/test_phase2_rom.py`, drop `seed_return_portals` from the import and delete the `seed_return_portals(memory)` line (portals are discovered live now).

- [ ] **Step 6: Lint and commit**

Run: `.venv/bin/ruff check env/campaign.py tests/test_campaign.py tests/test_phase2_rom.py`
Expected: clean.

```bash
git add env/campaign.py tests/test_campaign.py tests/test_map_memory.py tests/test_phase2_rom.py
git commit -m "feat: reach-home milestone via reach_map; drop placeholder _RETURN_PORTALS"
```

---

## Task 5: Full-campaign ROM smoke (load-bearing)

**Files:**
- Modify: `tests/test_phase2_rom.py`

The smoke already runs `run_campaign(PHASE2_CAMPAIGN)` from `post_rival.state`; Task 4 removed the `seed_return_portals` call. Confirm it now drives the `reach_map` descent + story A-spam end to end and asserts the deliverables.

- [ ] **Step 1: Confirm the smoke body**

Ensure `tests/test_phase2_rom.py` (a) no longer imports/calls `seed_return_portals`, (b) still asserts `result == "campaign_complete"`, `reader.has_pokedex()`, `reader.has_running_shoes()`, and additionally asserts `reader.has_item(0x4, 5)` (the Balls milestone), (c) dumps `states/post_phase2.state`.

- [ ] **Step 2: Run the smoke (load-bearing)**

Run (from the main repo):
```
POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
  .venv/bin/python -m pytest tests/test_phase2_rom.py -v
```
Expected: PASS (not skipped) — the campaign descends to the lab via `reach_map`, delivers the Pokédex + Balls, and `states/post_phase2.state` is written.

**GATING:** if it fails at the descent, report the reach_map outcome; the fix belongs in Task 3's primitive (reconcile with the Task 1 probe), not in weakening the assertion.

- [ ] **Step 3: Run the full suite + lint**

Run: `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check env/ tests/`
Expected: all green, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase2_rom.py
git commit -m "test: Phase 2 ROM smoke drives reach_map descent end to end (load-bearing)"
```

---

## Self-Review Notes

- **Spec coverage:** Unit 1 -> Task 1; Unit 2 -> Tasks 2-3; Unit 3 -> Task 4; Unit 4 -> Task 5. All covered.
- **Type consistency:** `reach_map(..., goal_map, direction_by_map, *, move_type_fn, predict, max_hops)` and `_cross_toward(..., from_map, direction, *, move_type_fn, predict)` are used identically in Tasks 2-4. `Milestone.reach: tuple[int, int] | None`. Return contracts: `reach_map` -> `arrived|stall|timeout|BATTLE_OUTCOMES`; `_cross_toward` -> `crossed|blocked|BATTLE_OUTCOMES`.
- **De-risk gate:** Task 1 is explicitly gating; Tasks 3/5 point back to it on ROM failure rather than weakening assertions.
