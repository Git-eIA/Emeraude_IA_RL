# Pokédex Return Driver (A2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A durable, tested driver `run_pokedex_return` that starts from `states/post_rival.state` (route_103), returns to Birch's lab through the Flora-gated Oldale→route_101 crossing, and delivers the Pokédex (`has_pokedex()` True after the lab GivePokedex cutscene).

**Architecture:** Two new durable primitives in `env/map_traveler.py` — an explore-based hop (`hop_via_explore` + private `_cross_portal`) for the route_103→Oldale leg, and a scripted-NPC crossing (`cross_scripted_npc`, the Flora gate). A new `env/campaign.py` orchestrator `run_pokedex_return` sequences: explore-hop → settle → Flora gate → `reach_map` descent → lab cutscene via the existing `_advance_story_dialogue`. No new emulator/env capability, no RAM pokes, no savestate splicing.

**Tech Stack:** Python 3.12, pytest, mGBA-via-`emulator.gba.GbaEmulator`, Stable-Baselines3 (Fighter checkpoint), existing `env/` navigation primitives (`explore_grid`, `navigate_grid`, `reach_map`, `_precision_walk_to`, `probe_step`, `_snapshot_settled`, `GridSnapshot`, `MapMemory`, `_advance_story_dialogue`).

**Spec:** `docs/superpowers/specs/2026-08-14-pokedex-return-driver-design.md`

**Resolved design decisions (baked into this plan):**
- **leg1 is an explore-hop, NOT `reach_map`.** Proven this session: `reach_map`'s `_cross_border` (DOWN) lands the player at Oldale **(8,1)**, from which the Flora precision-walk to `(10,19)` FAILS. The proven GO path lands Oldale **(11,1)** via `explore_grid` sweep + `_cross_portal` of the discovered south portal — from there the Flora walk succeeds. So the durable route_103→Oldale leg ports that proven explore path (`hop_via_explore` + `_cross_portal`), not `reach_map`. leg3 (route_101→lab) STAYS `reach_map` (B2 proved that descent).
- **Port the PROVEN path only.** `hop_via_explore` ports the `explore_grid` sweep + `_cross_portal` of the probe's `_hop_via_explore_then_scan` (`tools/probe_return_portals.py`). It does NOT port the `_straight_shot`/`_column_scan` fallbacks — they did not run on the GO path (unproven) and `_column_scan` would duplicate the durable `_cross_border` (DRY). `_cross_portal` also drops the probe's redundant `if r == "left_map"` branch (`left_map != "arrived"` already returns via the generic `r != "arrived"` guard).
- **Heal — DROPPED.** The probe's `_heal_party` is a RAM poke (`emu._core.memory.u8.raw_write`); a durable version would need a new `GbaEmulator` write method — a NEW emulator capability the spec forbids (non-goal line 200: "No new emulator/env capability. No RAM pokes"). Wild battles on the hops are handled by the Fighter inside `explore_grid`/`navigate_grid`/`reach_map` via `handle_battle_interruption`; a whiteout surfaces as a battle-outcome string that short-circuits the driver. No heal step, no `heal_failed` status. If the ROM smoke proves heal is genuinely required, that is a separate spec revision — not this driver.
- **Lab cutscene reuses `_advance_story_dialogue`.** The GO probe delivered the Pokédex with `_advance_story_dialogue(emu, reader, lambda r: r.has_pokedex())` (env/orders.py:344) — the same bounded A-spam primitive the Phase 2 story Order uses (`STORY_MAX_PRESSES = 2000`, ≫ the ~185 presses observed). The driver uses it directly. This DROPS the earlier plan's `_run_lab_pokedex_cutscene`, `VAR_BIRCH_LAB_STATE`, and the `buttons` import from `campaign.py`.
- **`_RETURN_DIRECTIONS` is UNCHANGED** (`{ROUTE_101: "down", LITTLEROOT: "up"}`). leg1 is an explore-hop with an explicit `"down"` argument, so route_103 is NOT added to `_RETURN_DIRECTIONS`; the existing pin `tests/test_campaign.py::test_return_directions_are_route101_down_littleroot_up` stays green.
- **The full chain is already proven end-to-end this session** (GO probe: explore-hop → Oldale (11,1) → Flora crosses → `reach_map` reaches lab (6,12) → `_advance_story_dialogue` delivers, `has_pokedex()` True). The single load-bearing durable proof is Task 4's gated ROM smoke; no throwaway re-run gate probe is written (it would only duplicate Task 4).

---

## File Structure

- **`env/map_traveler.py`** — add `from env.grid_explorer import explore_grid`, plus `_cross_portal` (private) and `hop_via_explore` (public — imported by `campaign.py`), and the `cross_scripted_npc` primitive + its module-level constants. Responsibilities: cross a discovered border portal (`_cross_portal`), hop to a neighbouring map by exploring for its portal (`hop_via_explore`), cross a scripted-NPC-gated connection (`cross_scripted_npc`).
- **`env/campaign.py`** — extend the `env.map_traveler` import with `cross_scripted_npc, hop_via_explore`; extend the `env.orders` import with `_advance_story_dialogue`; re-introduce `OLDALE`/`ROUTE_103`; add the Flora + settle constants; add `run_pokedex_return`. The driver only sequences existing primitives.
- **`tests/test_map_traveler.py`** — pure unit tests for `_cross_portal`, `hop_via_explore`, and `cross_scripted_npc` (fake world, no ROM), modelled on the existing crossing-helper tests.
- **`tests/test_campaign.py`** — pure orchestration unit tests for `run_pokedex_return` (monkeypatched legs).
- **`tests/test_pokedex_return_rom.py`** — NEW gated end-to-end ROM smoke (the single load-bearing proof), modelled on `tests/test_phase2_rom.py`.

---

## Task 1: `hop_via_explore` + `_cross_portal` (durable leg1) + unit tests

**Files:**
- Modify: `env/map_traveler.py` (add `explore_grid` import + `_cross_portal` + `hop_via_explore`)
- Test: `tests/test_map_traveler.py` (add unit tests)

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/test_map_traveler.py` (`MapMemory`, `map_traveler` are already imported at the top of the file; `_Snap` is defined here so the tests are self-contained):

```python
# ---------------------------------------------------------------------------
# A2: explore-based hop (route_103 -> Oldale) + portal crossing
# ---------------------------------------------------------------------------

from collections import namedtuple  # noqa: E402

_Snap = namedtuple("_Snap", "map_id pos")


def test_cross_portal_walks_to_from_cell_then_steps_into_the_neighbor(monkeypatch):
    memory = MapMemory()
    memory.record_portal((0, 18), (11, 9), "down", (0, 10), True, (11, 1))
    portal = memory.portal((0, 18), (0, 10))
    calls = []
    monkeypatch.setattr(
        map_traveler, "navigate_grid",
        lambda emu, rdr, target, **kw: (calls.append(target), "arrived")[1],
    )
    assert map_traveler._cross_portal(None, None, memory, portal, None, None) == "arrived"
    assert calls == [(11, 9), (11, 10)]   # from_cell, then one step DOWN (DELTAS["down"])


def test_cross_portal_short_circuits_when_it_cannot_reach_from_cell(monkeypatch):
    memory = MapMemory()
    memory.record_portal((0, 18), (11, 9), "down", (0, 10), True, (11, 1))
    portal = memory.portal((0, 18), (0, 10))
    monkeypatch.setattr(map_traveler, "navigate_grid", lambda *a, **k: "unreachable")
    assert map_traveler._cross_portal(None, None, memory, portal, None, None) == "unreachable"


def test_hop_via_explore_arrives_when_explore_auto_lands_on_the_target(monkeypatch):
    state = {"map": (0, 18), "pos": (11, 0)}
    memory = MapMemory()
    monkeypatch.setattr(map_traveler, "_snapshot_settled",
                        lambda rdr: _Snap(state["map"], state["pos"]))

    def fake_explore(emu, rdr, mem, tmap, **kw):
        state["map"], state["pos"] = (0, 10), (11, 1)   # swept off route_103 onto Oldale
        return "complete"

    monkeypatch.setattr(map_traveler, "explore_grid", fake_explore)
    assert map_traveler.hop_via_explore(
        None, None, memory, (0, 18), (0, 10), "down",
        move_type_fn=None, predict=None,
    ) == "arrived"
    assert memory.portal((0, 18), (0, 10)) is not None


def test_hop_via_explore_crosses_a_discovered_portal_when_explore_stays_put(monkeypatch):
    state = {"map": (0, 18), "pos": (11, 0)}
    memory = MapMemory()
    monkeypatch.setattr(map_traveler, "_snapshot_settled",
                        lambda rdr: _Snap(state["map"], state["pos"]))

    def fake_explore(emu, rdr, mem, tmap, **kw):
        mem.record_portal((0, 18), (11, 9), "down", (0, 10), True, (11, 1))  # found, still on route_103
        return "complete"

    def fake_cross(emu, rdr, mem, portal, mtf, predict):
        state["map"], state["pos"] = (0, 10), (11, 1)
        return "arrived"

    monkeypatch.setattr(map_traveler, "explore_grid", fake_explore)
    monkeypatch.setattr(map_traveler, "_cross_portal", fake_cross)
    assert map_traveler.hop_via_explore(
        None, None, memory, (0, 18), (0, 10), "down",
        move_type_fn=None, predict=None,
    ) == "arrived"


def test_hop_via_explore_reports_no_portal_when_none_is_discovered(monkeypatch):
    state = {"map": (0, 18), "pos": (11, 0)}
    monkeypatch.setattr(map_traveler, "_snapshot_settled",
                        lambda rdr: _Snap(state["map"], state["pos"]))
    monkeypatch.setattr(map_traveler, "explore_grid", lambda *a, **k: "complete")
    assert map_traveler.hop_via_explore(
        None, None, MapMemory(), (0, 18), (0, 10), "down",
        move_type_fn=None, predict=None,
    ) == "no_portal"


def test_hop_via_explore_stalls_when_not_starting_on_from_map(monkeypatch):
    monkeypatch.setattr(map_traveler, "_snapshot_settled", lambda rdr: _Snap((0, 16), (0, 0)))
    assert map_traveler.hop_via_explore(
        None, None, MapMemory(), (0, 18), (0, 10), "down",
        move_type_fn=None, predict=None,
    ) == "stall"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k "cross_portal or hop_via_explore" -v`
Expected: FAIL with `AttributeError: module 'env.map_traveler' has no attribute '_cross_portal'` / `hop_via_explore`.

- [ ] **Step 3: Add the import and the two primitives**

In `env/map_traveler.py`, add after the existing imports (below line 25, `from env.route_planner import plan_route`):

```python
from env.grid_explorer import explore_grid
```

Then add these two functions after `_cross_up_warp` (after line 293), before `reach_map`:

```python
def _cross_portal(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    portal: Any,
    move_type_fn: Any,
    predict: Any,
) -> str:
    """Navigate to a discovered portal's from_cell, then step once into the
    neighbour cell across the border. Returns navigate_grid's outcome
    ('arrived' on success, else the failing nav status)."""
    r = navigate_grid(
        emulator, reader, portal.from_cell,
        memory=memory, move_type_fn=move_type_fn, predict=predict,
    )
    if r != "arrived":
        return r
    dx, dy = DELTAS[portal.direction]
    nb = (portal.from_cell[0] + dx, portal.from_cell[1] + dy)
    return navigate_grid(
        emulator, reader, nb,
        memory=memory, move_type_fn=move_type_fn, predict=predict,
    )


def hop_via_explore(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    from_map: tuple[int, int],
    to_map: tuple[int, int],
    direction: str,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Hop from_map -> to_map by exploring from_map's borders for the portal.

    explore_grid sweeps from_map's reachable border tiles and records portals.
    If the sweep itself lands on to_map, that IS the hop. Otherwise the discovered
    to_map portal is crossed with _cross_portal. Used for route_103 -> Oldale, where
    _cross_border lands at an Oldale tile the Flora walk cannot reach; the explore
    sweep lands at the reachable (11,1) instead.

    Returns 'arrived' on success, else 'stall' (not on from_map / explore left to a
    third map) or 'no_portal' (no to_map portal discovered).
    """
    here = _snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return "stall"
    entry = here.pos
    explore_grid(emulator, reader, memory, from_map,
                 move_type_fn=move_type_fn, predict=predict)
    now = _snapshot_settled(reader)
    if now is not None and now.map_id == to_map:
        memory.record_portal(from_map, entry, direction, to_map, True, now.pos)
        return "arrived"
    portal = memory.portal(from_map, to_map)
    if portal is None:
        return "no_portal"
    if now is None or now.map_id != from_map:
        return "stall"
    _cross_portal(emulator, reader, memory, portal, move_type_fn, predict)
    after = _snapshot_settled(reader)
    if after is not None and after.map_id == to_map:
        return "arrived"
    return "stall"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k "cross_portal or hop_via_explore" -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full map_traveler suite + ruff**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/map_traveler.py tests/test_map_traveler.py`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add env/map_traveler.py tests/test_map_traveler.py
git commit -m "$(cat <<'EOF'
feat: hop_via_explore + _cross_portal (explore-based map hop)

Ports the proven GO path for route_103->Oldale: explore_grid sweeps the border,
then _cross_portal crosses the discovered portal. reach_map's _cross_border lands
Oldale at (8,1) where the Flora walk fails; the explore sweep lands the reachable
(11,1). Only the proven path is ported (no straight_shot/column_scan fallbacks;
column_scan would duplicate _cross_border). Pure unit tests over a fake world.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `cross_scripted_npc` primitive (Flora gate) + unit tests

**Files:**
- Modify: `env/map_traveler.py` (add `from emulator import buttons` + constants + `cross_scripted_npc`)
- Test: `tests/test_map_traveler.py` (add three unit tests)

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/test_map_traveler.py` (`_OneMapWorld`, `_FakeSnap`, `buttons`, `MapMemory`, `map_traveler` are already available):

```python
# ---------------------------------------------------------------------------
# A2: scripted-NPC crossing (Flora gate)
# ---------------------------------------------------------------------------


def test_cross_scripted_npc_crosses_after_dialogue(monkeypatch):
    world = _OneMapWorld(map_id=(0, 10), pos=(10, 15))
    memory = MapMemory()
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19)})))

    def fake_walk(emu, rdr, snap, cell, from_map):
        world.pos = cell
        return True

    monkeypatch.setattr(map_traveler, "_precision_walk_to", fake_walk)

    def fake_probe(emu, rdr, before, direction):
        world.map_id = (0, 16)   # pushing DOWN past Flora flips Oldale -> route_101

    monkeypatch.setattr(map_traveler, "probe_step", fake_probe)

    assert map_traveler.cross_scripted_npc(
        world, world, memory, (0, 10),
        stand_tile=(10, 19), face_dir="right", cross_dir="down", max_presses=10,
    ) is True
    assert memory.portal((0, 10), (0, 16)) is not None


def test_cross_scripted_npc_false_if_it_cannot_reach_the_stand_tile(monkeypatch):
    world = _OneMapWorld(map_id=(0, 10), pos=(10, 15))
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19)})))
    monkeypatch.setattr(map_traveler, "_precision_walk_to", lambda *a: False)
    assert map_traveler.cross_scripted_npc(
        world, world, MapMemory(), (0, 10),
        stand_tile=(10, 19), face_dir="right", cross_dir="down", max_presses=10,
    ) is False


def test_cross_scripted_npc_false_when_the_gate_never_opens(monkeypatch):
    world = _OneMapWorld(map_id=(0, 10), pos=(10, 15))
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19)})))
    monkeypatch.setattr(map_traveler, "_precision_walk_to",
                        lambda emu, rdr, snap, cell, fm: True)
    monkeypatch.setattr(map_traveler, "probe_step", lambda *a: None)   # never flips
    assert map_traveler.cross_scripted_npc(
        world, world, MapMemory(), (0, 10),
        stand_tile=(10, 19), face_dir="right", cross_dir="down", max_presses=10,
    ) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k cross_scripted_npc -v`
Expected: FAIL with `AttributeError: module 'env.map_traveler' has no attribute 'cross_scripted_npc'`.

- [ ] **Step 3: Add the import, constants, and the primitive**

In `env/map_traveler.py`, add after the `from env.grid_explorer import explore_grid` line from Task 1:

```python
from emulator import buttons
```

Then add these constants beside the other crossing constants (after `_BORDER_SORT`, around line 122):

```python
_FACE_FRAMES = 4        # tap toward the NPC to rotate the sprite without stepping
_NPC_ASPAM_FRAMES = 8   # A-press dwell to play/clear one dialogue box
_CROSS_PUSH_MAX = 12    # bounded cross-direction steps after the dialogue clears
```

Then add the primitive after `hop_via_explore` (from Task 1), before `reach_map`:

```python
def cross_scripted_npc(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    from_map: tuple[int, int],
    *,
    stand_tile: tuple[int, int],
    face_dir: str,
    cross_dir: str,
    max_presses: int,
) -> bool:
    """Cross an NPC-gated connection: walk to stand_tile, face the NPC, A-spam its
    dialogue, then push cross_dir until the map changes. Returns True iff the map flipped.

    Flora stands on Oldale's south connection tile; the crossing only opens once her
    dialogue has played. face_dir (toward the NPC) and cross_dir (toward the next map)
    are distinct because Flora is faced EAST but the crossing is DOWN.
    """
    here = _snapshot_settled(reader)
    if here is None or here.map_id != from_map:
        return False
    snap = GridSnapshot.from_reader(reader.grid_reader, from_map)
    if snap is None:
        return False
    if not _precision_walk_to(emulator, reader, snap, stand_tile, from_map):
        return False
    # Tap toward the NPC: the NPC tile walls the player, so this only rotates the sprite.
    emulator.step(DIRECTION_KEYS[face_dir], _FACE_FRAMES)
    emulator.step(0, _PRECISION_RELEASE_FRAMES)
    # Play/clear the dialogue (bounded A-spam; A never steps).
    for _ in range(max_presses):
        emulator.step(buttons.KEY_A, _NPC_ASPAM_FRAMES)
        emulator.step(0, _NPC_ASPAM_FRAMES)
    # Push toward the next map until it flips.
    for _ in range(_CROSS_PUSH_MAX):
        before = _snapshot_settled(reader)
        if before is None:
            return False
        if before.map_id != from_map:
            memory.record_portal(from_map, stand_tile, cross_dir, before.map_id, True, before.pos)
            return True
        probe_step(emulator, reader, before, cross_dir)
    after = _snapshot_settled(reader)
    if after is not None and after.map_id != from_map:
        memory.record_portal(from_map, stand_tile, cross_dir, after.map_id, True, after.pos)
        return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -k cross_scripted_npc -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full map_traveler suite + ruff**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_map_traveler.py -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/map_traveler.py tests/test_map_traveler.py`
Expected: all pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add env/map_traveler.py tests/test_map_traveler.py
git commit -m "$(cat <<'EOF'
feat: cross_scripted_npc primitive (walk, face NPC, A-spam dialogue, push across)

A durable crossing beside _cross_border/_cross_up_warp for NPC-gated connections
(the Oldale->route_101 Flora gate). face_dir and cross_dir are distinct params so
one primitive serves any faced-one-way / cross-another connection. Pure unit tests
over a fake world mirror the _cross_up_warp tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `run_pokedex_return` orchestrator + constants + unit tests

**Files:**
- Modify: `env/campaign.py` (imports, re-introduce OLDALE/ROUTE_103, Flora + settle constants, `run_pokedex_return`)
- Test: `tests/test_campaign.py` (add orchestration unit tests)

- [ ] **Step 1: Write the failing unit tests**

Add `OLDALE, ROUTE_103, run_pokedex_return` to the imports at the top of `tests/test_campaign.py` (extend the existing `from env.campaign import (...)` block). Then append the orchestration tests at the end of the file:

```python
# ---------------------------------------------------------------------------
# A2: run_pokedex_return orchestration
# ---------------------------------------------------------------------------


class _Emu:
    """Fake emulator: run_pokedex_return calls .step for the settle frames."""

    def step(self, keys, frames):
        pass


def _wire_return(monkeypatch, *, hop, flora, descent, cutscene):
    """Stub the four legs of run_pokedex_return; record their calls in order."""
    calls = []
    monkeypatch.setattr(
        campaign, "hop_via_explore",
        lambda *a, **k: (calls.append(("hop", a[3], a[4])), hop)[1],
    )
    monkeypatch.setattr(
        campaign, "cross_scripted_npc",
        lambda *a, **k: (calls.append(("flora", a[3])), flora)[1],
    )
    monkeypatch.setattr(
        campaign, "reach_map",
        lambda *a, **k: (calls.append(("reach", a[3])), descent)[1],
    )
    monkeypatch.setattr(
        campaign, "_advance_story_dialogue",
        lambda *a, **k: (calls.append(("cutscene",)), cutscene)[1],
    )
    return calls


def test_run_pokedex_return_delivers_on_the_happy_path(monkeypatch):
    calls = _wire_return(monkeypatch, hop="arrived", flora=True,
                         descent="arrived", cutscene="story_done")
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "pokedex_delivered"
    assert calls == [
        ("hop", ROUTE_103, OLDALE), ("flora", OLDALE),
        ("reach", LAB), ("cutscene",),
    ]


def test_run_pokedex_return_propagates_the_oldale_hop_outcome(monkeypatch):
    # hop_via_explore's own status (stall/no_portal/battle_lost/...) surfaces verbatim.
    _wire_return(monkeypatch, hop="stall", flora=True,
                 descent="arrived", cutscene="story_done")
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "stall"


def test_run_pokedex_return_stops_if_flora_never_crosses(monkeypatch):
    _wire_return(monkeypatch, hop="arrived", flora=False,
                 descent="arrived", cutscene="story_done")
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "flora_no_cross"


def test_run_pokedex_return_propagates_the_descent_outcome(monkeypatch):
    _wire_return(monkeypatch, hop="arrived", flora=True,
                 descent="stall", cutscene="story_done")
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "stall"


def test_run_pokedex_return_stops_if_the_pokedex_is_not_delivered(monkeypatch):
    _wire_return(monkeypatch, hop="arrived", flora=True,
                 descent="arrived", cutscene="story_timeout")
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "pokedex_not_delivered"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -k "pokedex_return" -v`
Expected: FAIL — `ImportError: cannot import name 'ROUTE_103'` / `run_pokedex_return` not defined.

- [ ] **Step 3: Implement the imports, constants, and orchestrator**

In `env/campaign.py`, change the map_traveler import (line 18) from `from env.map_traveler import reach_map` to:

```python
from env.map_traveler import cross_scripted_npc, hop_via_explore, reach_map
```

And extend the orders import (line 19) from `from env.orders import Order, execute_order, reached` to:

```python
from env.orders import Order, _advance_story_dialogue, execute_order, reached
```

Re-introduce the two dropped map ids in the constants block (add beside `ROUTE_101`/`LITTLEROOT`/`LAB`, leaving `_RETURN_DIRECTIONS` unchanged):

```python
ROUTE_103 = (0, 18)
OLDALE = (0, 10)
```

Add the Flora + settle constants after `_RETURN_DIRECTIONS`:

```python
# Settle frames after the route_103 -> Oldale hop, before the Flora walk (let the
# map transition finish so _precision_walk_to reads a stable grid).
_SETTLE_FRAMES = 60

# Flora gate: she stands on Oldale's south connection tile (11,19). Stand just north
# at (10,19) facing EAST, A-spam her dialogue, then DOWN crosses into route_101.
_FLORA_STAND = (10, 19)
_FLORA_FACE = "right"
_FLORA_CROSS = "down"
_FLORA_MAX_PRESSES = 60
```

Add `run_pokedex_return` after `run_campaign` (end of file):

```python
def run_pokedex_return(
    emulator: Any,
    reader: Any,
    memory: Any,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Drive post_rival.state back to Birch's lab and deliver the Pokédex.

    hop_via_explore hops route_103 -> Oldale (explore sweep + portal cross, landing the
    Flora-reachable Oldale tile); cross_scripted_npc plays Flora's gate into route_101;
    reach_map descends route_101 -> Littleroot -> lab; then _advance_story_dialogue runs
    the lab OnFrame GivePokedex cutscene until has_pokedex(). Every leg is bounded and
    surfaces the first failure.

    Returns 'pokedex_delivered' on success, or on failure the first failing leg's status:
    hop_via_explore / reach_map outcomes ('stall' | 'no_portal' | 'timeout' | 'battle_lost'
    | 'battle_timeout' | 'battle_interrupted' | 'unreachable') propagate verbatim; the Flora
    and cutscene legs surface 'flora_no_cross' | 'pokedex_not_delivered'.
    """
    hopped = hop_via_explore(
        emulator, reader, memory, ROUTE_103, OLDALE, "down",
        move_type_fn=move_type_fn, predict=predict,
    )
    if hopped != "arrived":
        return hopped

    emulator.step(0, _SETTLE_FRAMES)
    if not cross_scripted_npc(
        emulator, reader, memory, OLDALE,
        stand_tile=_FLORA_STAND, face_dir=_FLORA_FACE,
        cross_dir=_FLORA_CROSS, max_presses=_FLORA_MAX_PRESSES,
    ):
        return "flora_no_cross"

    descended = reach_map(
        emulator, reader, memory, LAB, _RETURN_DIRECTIONS,
        move_type_fn=move_type_fn, predict=predict,
    )
    if descended != "arrived":
        return descended

    if _advance_story_dialogue(emulator, reader, lambda r: r.has_pokedex()) != "story_done":
        return "pokedex_not_delivered"
    return "pokedex_delivered"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -q`
Expected: all pass (the new orchestration tests + the unchanged `_RETURN_DIRECTIONS` pin still green).

- [ ] **Step 5: Run ruff**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/campaign.py tests/test_campaign.py`
Expected: ruff clean.

- [ ] **Step 6: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "$(cat <<'EOF'
feat: run_pokedex_return driver (explore-hop, Flora gate, descend, GivePokedex)

New campaign orchestrator: hop_via_explore hops route_103 -> Oldale (landing the
Flora-reachable tile that reach_map's _cross_border misses), cross_scripted_npc plays
Flora's gate into route_101, reach_map descends to the lab, then _advance_story_dialogue
runs the lab GivePokedex cutscene until has_pokedex(). Re-introduce OLDALE/ROUTE_103 +
Flora/settle constants; _RETURN_DIRECTIONS unchanged (leg1 is an explore-hop, not a
reach_map). No heal step, no RAM poke, no new emulator capability. Pure orchestration
unit tests stub each leg.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Gated end-to-end ROM smoke (load-bearing)

**Files:**
- Create: `tests/test_pokedex_return_rom.py`

- [ ] **Step 1: Write the gated smoke test**

```python
"""Gated ROM smoke: run run_pokedex_return from post_rival.state end to end.

The single load-bearing proof of the A2 chain: hop_via_explore hops route_103 -> Oldale,
cross_scripted_npc plays Flora's gate into route_101, reach_map descends to the lab, and
the OnFrame GivePokedex cutscene delivers the Pokédex. Asserts has_pokedex() True + lab
arrival and dumps states/post_pokedex.state. Slow (~minute, Fighter-driven). Triple-skips
without ROM / Fighter checkpoint / post_rival.state.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
FIGHTER_CKPT = "checkpoints/fighter/ppo_fighter_final.zip"
START_STATE = "states/post_rival.state"

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(FIGHTER_CKPT).exists(), reason="Fighter checkpoint missing"),
    pytest.mark.skipif(not Path(START_STATE).exists(), reason="post_rival.state missing"),
]


def test_run_pokedex_return_delivers_the_pokedex() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.campaign import LAB, run_pokedex_return
    from env.game_state import EmeraldReader
    from env.map_memory import MapMemory
    from env.world_reader import WorldReader

    emu = GbaEmulator(ROM)
    with open(START_STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)
    memory = MapMemory()

    model = PPO.load(FIGHTER_CKPT, device="cpu")

    def predict(obs):
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)

    # run_pokedex_return needs snapshot/grid_reader (world) AND has_pokedex (reader);
    # a thin adapter forwards both, matching test_phase2_rom.py.
    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_pokedex_return(
        emu, _Reader(), memory,
        move_type_fn=move_type_fn, predict=predict,
    )

    assert result == "pokedex_delivered", result
    assert reader.has_pokedex() is True
    settled = world.snapshot()
    assert settled is not None and settled.map_id == LAB, settled

    Path("states/post_pokedex.state").write_bytes(emu.save_state())
```

- [ ] **Step 2: Run the gated smoke (with ROM)**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && POKEMON_EMERALD_ROM="/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba" /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_pokedex_return_rom.py -v`
Expected: 1 passed (`has_pokedex()` True, `map_id == LAB`, `states/post_pokedex.state` written). If it fails, do NOT patch the test to pass — return to the GO probe output (`tools/probe_return_portals.py`) and superpowers:systematic-debugging to find which leg regressed.

- [ ] **Step 3: Run ruff**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/ruff check tests/test_pokedex_return_rom.py`
Expected: ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pokedex_return_rom.py
git commit -m "$(cat <<'EOF'
test: gated ROM smoke — run_pokedex_return delivers the Pokedex end to end

Load-bearing proof of the A2 chain from post_rival.state: explore-hop to Oldale, Flora
gate, descend to the lab, GivePokedex cutscene. Asserts has_pokedex() True + lab arrival
and dumps states/post_pokedex.state. Triple-skips without ROM / Fighter / post_rival.state.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Step 1: Run the full suite (with ROM)**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && POKEMON_EMERALD_ROM="/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba" /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: all pass (prior baseline 355 passed + the new unit + gated smoke), no regressions.

- [ ] **Step 2: Run ruff on the whole change**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/ tests/`
Expected: ruff clean.

Then use superpowers:finishing-a-development-branch to complete the branch.

---

## GSTACK REVIEW REPORT

| Runs | Status | Findings |
|------|--------|----------|
| gap-check (symbols/scope/proof) | PASS | 0 undefined symbols, 0 placeholders; 2 src + 3 test files, 4 fns / 0 classes (under threshold); chain proven end-to-end this session |
| eng-review (arch/quality/tests/perf) | PASS with 2 minors | M1, M2 below — both covered by the load-bearing ROM smoke |

**Verified present (no undefined symbols):** `Any` (campaign.py:16, map_traveler.py:11), `DELTAS` (map_traveler.py:14), `buttons`/`_OneMapWorld` (grid_reader/step/snapshot/in_battle/battle_starting)/`_FakeSnap`/`MapMemory` (test_map_traveler.py). Locked decisions correctly encoded: leg1 explore-hop (not reach_map), heal DROPPED (no RAM poke), cutscene via `_advance_story_dialogue`, `_RETURN_DIRECTIONS` unchanged.

**Minors (non-blocking, both covered by Task 4 ROM smoke):**
- M1 — Fighter threading (`move_type_fn`/`predict`) not asserted in the orchestration unit tests (legs stubbed with `lambda *a, **k`). B2 had `test_advance_threads_the_fighter`; optional to port.
- M2 — `emulator.step(0, _SETTLE_FRAMES)` not asserted in unit tests (`_Emu` fake is a no-op).

VERDICT: GO — execute via subagent-driven-development. CODEX/CROSS-MODEL: not run (mature plan, iterated across sessions with F1/F2 absorbed).

NO UNRESOLVED DECISIONS
