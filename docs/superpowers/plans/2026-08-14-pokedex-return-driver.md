# Pokédex Return Driver (A2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A durable, tested driver `run_pokedex_return` that starts from `states/post_rival.state` (route_103), returns to Birch's lab through the Flora-gated Oldale→route_101 crossing, and delivers the Pokédex (`has_pokedex()` True after the lab GivePokedex cutscene).

**Architecture:** One new durable crossing primitive `cross_scripted_npc` in `env/map_traveler.py` (walk to a tile, face an NPC, A-spam its dialogue, push across) sits between two existing `reach_map` greedy descents in a new `env/campaign.py` orchestrator `run_pokedex_return`. The orchestrator ends by running the lab OnFrame GivePokedex cutscene (idle auto-walk + bounded A-spam until `VAR_BIRCH_LAB_STATE` steps 4→5). No new emulator/env capability, no RAM pokes, no savestate splicing.

**Tech Stack:** Python 3.12, pytest, mGBA-via-`emulator.gba.GbaEmulator`, Stable-Baselines3 (Fighter checkpoint), existing `env/` navigation primitives (`reach_map`, `_cross_border`, `_precision_walk_to`, `probe_step`, `GridSnapshot`, `MapMemory`).

**Spec:** `docs/superpowers/specs/2026-08-14-pokedex-return-driver-design.md`

**Resolved design decisions (baked into this plan):**
- **Q1 (heal) — DROPPED.** The probe's `_heal_party` is a RAM poke (`emu._core.memory.u8.raw_write`), unfit for durable code. Wild battles on the hops are handled by the Fighter inside `reach_map`/`_cross_border` via `handle_battle_interruption`; a whiteout surfaces as a battle-outcome string that short-circuits the driver. There is NO heal step and NO `heal_failed` status.
- **Q2 (route_103→Oldale stop) — de-risked by Task 1, mechanism confirmed by reading `reach_map`.** `reach_map` checks `here.map_id == goal_map` BEFORE looking up a direction (`env/map_traveler.py:318`), so `reach_map(goal=OLDALE, {ROUTE_103:"down", ...})` stops at Oldale with no overshoot into route_103. Task 1 is a gating throwaway probe that STOPS the plan if the leg cannot be crossed durably.
- **`_RETURN_DIRECTIONS` is extended** with `ROUTE_103: "down"` and reused for BOTH legs. This breaks the existing pin `tests/test_campaign.py::test_return_directions_are_route101_down_littleroot_up`; Task 3 updates that test.

---

## File Structure

- **`env/map_traveler.py`** — add `from emulator import buttons` import + the `cross_scripted_npc` primitive and its module-level constants, beside the other crossing helpers. One responsibility: cross a scripted-NPC-gated connection.
- **`env/campaign.py`** — add `from emulator import buttons` import, `from env.map_traveler import cross_scripted_npc` (join the existing `reach_map` import), re-introduce `OLDALE`/`ROUTE_103`/`VAR_BIRCH_LAB_STATE`, the Flora + lab-cutscene constants, extend `_RETURN_DIRECTIONS`, and add `run_pokedex_return` + the `_run_lab_pokedex_cutscene` helper. One responsibility per function; the driver only sequences existing primitives.
- **`tests/test_map_traveler.py`** — pure unit tests for `cross_scripted_npc` (fake world, no ROM), modelled on the existing `_cross_up_warp` unit tests.
- **`tests/test_campaign.py`** — pure orchestration unit tests for `run_pokedex_return` (monkeypatched `reach_map`/`cross_scripted_npc`/`_run_lab_pokedex_cutscene`); update the `_RETURN_DIRECTIONS` pin.
- **`tests/test_pokedex_return_rom.py`** — NEW gated end-to-end ROM smoke (the single load-bearing proof), modelled on `tests/test_phase2_rom.py`.
- **`tools/probe_pokedex_return_gate.py`** — Task 1 throwaway de-risk probe (stdout-only, NOT committed).

---

## Task 1: Gating de-risk probe (route_103→Oldale→Flora→lab→cutscene)

**This is a STOP-gate, mirroring B2's Task 1.** It runs the full chain manually from `post_rival.state` and prints each leg's status. If the route_103→Oldale leg cannot be crossed and stopped at Oldale, or Flora's gate never opens, **STOP and report to the user before writing any durable code.** The probe is throwaway: stdout-only, NOT committed (per repo convention for `tools/` probes).

**Files:**
- Create (throwaway, uncommitted): `tools/probe_pokedex_return_gate.py`

- [ ] **Step 1: Write the probe**

```python
"""Throwaway de-risk gate for A2 (Pokédex return). Stdout only, NOT committed.

Runs the full chain from post_rival.state and prints each leg's status:
  1. reach_map(OLDALE, {ROUTE_103: down, ...})   -> must be 'arrived' AND stop at Oldale
  2. Flora gate (stand (10,19), face right, A-spam, push down) -> must flip to route_101
  3. reach_map(LAB, ...)                          -> must be 'arrived' at the lab
  4. lab OnFrame cutscene (idle + A-spam)         -> VAR_BIRCH_LAB_STATE 4->5, has_pokedex True
GO iff every leg passes. Otherwise STOP the plan and report.
"""
from __future__ import annotations

import os

from stable_baselines3 import PPO

from agent.train_fighter import make_move_type_fn
from emulator import buttons
from emulator.gba import GbaEmulator
from env.game_state import EmeraldReader
from env.grid_navigator import DIRECTION_KEYS, snapshot_settled
from env.map_memory import MapMemory
from env.map_traveler import _precision_walk_to, reach_map
from env.grid_snapshot import GridSnapshot
from env.world_reader import WorldReader

ROM = os.environ["POKEMON_EMERALD_ROM"]
OLDALE, ROUTE_103, ROUTE_101, LITTLEROOT, LAB = (0, 10), (0, 18), (0, 16), (0, 9), (1, 4)
VAR_BIRCH_LAB_STATE = 0x4084
DIRECTIONS = {ROUTE_103: "down", ROUTE_101: "down", LITTLEROOT: "up"}


def main() -> None:
    emu = GbaEmulator(ROM)
    with open("states/post_rival.state", "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)
    memory = MapMemory()
    model = PPO.load("checkpoints/fighter/ppo_fighter_final.zip", device="cpu")

    def predict(obs):
        return int(model.predict(obs, deterministic=True)[0])

    move_type_fn = make_move_type_fn(emu)

    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    rdr = _Reader()

    print("start:", world.snapshot())
    hop = reach_map(emu, rdr, memory, OLDALE, DIRECTIONS,
                    move_type_fn=move_type_fn, predict=predict)
    print("leg1 reach OLDALE:", hop, world.snapshot())
    if hop != "arrived" or world.snapshot().map_id != OLDALE:
        print("STOP: route_103->Oldale leg failed"); return

    snap = GridSnapshot.from_reader(rdr.grid_reader, OLDALE)
    walked = _precision_walk_to(emu, rdr, snap, (10, 19), OLDALE)
    emu.step(DIRECTION_KEYS["right"], 4); emu.step(0, 32)
    for _ in range(60):
        emu.step(buttons.KEY_A, 8); emu.step(0, 8)
    for _ in range(12):
        before = snapshot_settled(rdr)
        if before is None or before.map_id != OLDALE:
            break
        emu.step(DIRECTION_KEYS["down"], 8); emu.step(0, 8)
    print("leg2 Flora walk/cross:", walked, world.snapshot())
    if world.snapshot().map_id != ROUTE_101:
        print("STOP: Flora gate never crossed"); return

    desc = reach_map(emu, rdr, memory, LAB, DIRECTIONS,
                     move_type_fn=move_type_fn, predict=predict)
    print("leg3 reach LAB:", desc, world.snapshot())
    if desc != "arrived" or world.snapshot().map_id != LAB:
        print("STOP: route_101->lab descent failed"); return

    emu.step(0, 64)
    delivered = False
    for _ in range(400):
        sb1 = reader._save_block1()
        if sb1 is not None and reader._var(sb1, VAR_BIRCH_LAB_STATE) == 5:
            delivered = reader.has_pokedex(); break
        emu.step(buttons.KEY_A, 8); emu.step(0, 8)
    print("leg4 cutscene: VAR==5?", delivered, "has_pokedex:", reader.has_pokedex())
    print("GATE:", "GO" if delivered and reader.has_pokedex() else "STOP")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && POKEMON_EMERALD_ROM="/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba" /Users/_eloi/Projets/Emu/.venv/bin/python tools/probe_pokedex_return_gate.py`

Expected: each leg prints its status; final line `GATE: GO`. If any leg prints `STOP: ...`, the leg is not durably crossable — **halt the plan and report the failing leg to the user.** Do NOT proceed to Task 2 on a STOP.

- [ ] **Step 3: Record the gate outcome**

On `GATE: GO`, note the observed Flora stand tile / cross behaviour and the A-press count where `VAR` flipped 4→5 (used to size `max_presses` in Task 3). No commit (throwaway probe stays uncommitted).

---

## Task 2: `cross_scripted_npc` primitive + unit tests

**Files:**
- Modify: `env/map_traveler.py` (add import + constants + `cross_scripted_npc`)
- Test: `tests/test_map_traveler.py` (add three unit tests)

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/test_map_traveler.py` (after the Task 2 crossing-helper tests, near line 455; `_OneMapWorld`, `_FakeSnap`, `buttons`, `MapMemory`, `map_traveler` are already imported at the top of the file):

```python
# ---------------------------------------------------------------------------
# A2: scripted-NPC crossing (Flora gate)
# ---------------------------------------------------------------------------


def testcross_scripted_npc_crosses_after_dialogue(monkeypatch):
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


def testcross_scripted_npc_false_if_it_cannot_reach_the_stand_tile(monkeypatch):
    world = _OneMapWorld(map_id=(0, 10), pos=(10, 15))
    monkeypatch.setattr(map_traveler.GridSnapshot, "from_reader",
                        staticmethod(lambda *a: _FakeSnap({(10, 19)})))
    monkeypatch.setattr(map_traveler, "_precision_walk_to", lambda *a: False)
    assert map_traveler.cross_scripted_npc(
        world, world, MapMemory(), (0, 10),
        stand_tile=(10, 19), face_dir="right", cross_dir="down", max_presses=10,
    ) is False


def testcross_scripted_npc_false_when_the_gate_never_opens(monkeypatch):
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

- [ ] **Step 3: Add the import and the primitive**

In `env/map_traveler.py`, add the buttons import after the existing imports (below line 25, `from env.route_planner import plan_route`):

```python
from emulator import buttons
```

Then add these constants beside the other crossing constants (after `_BORDER_SORT`, around line 122):

```python
_FACE_FRAMES = 4        # tap toward the NPC to rotate the sprite without stepping
_NPC_ASPAM_FRAMES = 8   # A-press dwell to play/clear one dialogue box
_CROSS_PUSH_MAX = 12    # bounded cross-direction steps after the dialogue clears
```

Then add the primitive after `_cross_up_warp` (after line 293), before `reach_map`:

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

## Task 3: `run_pokedex_return` orchestrator + constants

**Files:**
- Modify: `env/campaign.py` (imports, constants, extend `_RETURN_DIRECTIONS`, `_run_lab_pokedex_cutscene`, `run_pokedex_return`)
- Test: `tests/test_campaign.py` (update `_RETURN_DIRECTIONS` pin, add orchestration unit tests)

- [ ] **Step 1: Write the failing unit tests**

First, update the existing pin in `tests/test_campaign.py:258-259` to reflect the extended chain:

```python
def test_return_directions_include_route103_down():
    assert _RETURN_DIRECTIONS == {ROUTE_103: "down", ROUTE_101: "down", LITTLEROOT: "up"}
```

Add `ROUTE_103` and `run_pokedex_return` to the imports at the top of `tests/test_campaign.py` (extend the existing `from env.campaign import (...)` block with `ROUTE_103,` and `run_pokedex_return,`). Then append the orchestration tests at the end of the file:

```python
# ---------------------------------------------------------------------------
# A2: run_pokedex_return orchestration
# ---------------------------------------------------------------------------


def _wire_return(monkeypatch, *, hops, flora, descent, cutscene):
    """Stub the four legs of run_pokedex_return; record their calls."""
    calls = []

    def fake_reach(emu, rdr, mem, goal, directions, **kw):
        calls.append(("reach", goal))
        return hops.pop(0) if goal == campaign.OLDALE else descent.pop(0)

    def fake_flora(emu, rdr, mem, from_map, **kw):
        calls.append(("flora", from_map))
        return flora

    def fake_cutscene(emu, rdr, **kw):
        calls.append(("cutscene", None))
        return cutscene

    monkeypatch.setattr(campaign, "reach_map", fake_reach)
    monkeypatch.setattr(campaign, "cross_scripted_npc", fake_flora)
    monkeypatch.setattr(campaign, "_run_lab_pokedex_cutscene", fake_cutscene)
    return calls


def test_run_pokedex_return_delivers_on_the_happy_path(monkeypatch):
    calls = _wire_return(monkeypatch, hops=["arrived"], flora=True,
                         descent=["arrived"], cutscene=True)
    result = run_pokedex_return(None, None, MapMemory())
    assert result == "pokedex_delivered"
    assert calls == [
        ("reach", campaign.OLDALE), ("flora", campaign.OLDALE),
        ("reach", LAB), ("cutscene", None),
    ]


def test_run_pokedex_return_propagates_the_oldale_hop_outcome(monkeypatch):
    # reach_map's own status (stall/timeout/battle_lost/...) surfaces verbatim,
    # not collapsed to an opaque "hop_failed".
    _wire_return(monkeypatch, hops=["stall"], flora=True,
                 descent=["arrived"], cutscene=True)
    assert run_pokedex_return(None, None, MapMemory()) == "stall"


def test_run_pokedex_return_stops_if_flora_never_crosses(monkeypatch):
    _wire_return(monkeypatch, hops=["arrived"], flora=False,
                 descent=["arrived"], cutscene=True)
    assert run_pokedex_return(None, None, MapMemory()) == "flora_no_cross"


def test_run_pokedex_return_propagates_the_descent_outcome(monkeypatch):
    # Same as the hop: the descent reach_map's status surfaces verbatim.
    _wire_return(monkeypatch, hops=["arrived"], flora=True,
                 descent=["stall"], cutscene=True)
    assert run_pokedex_return(None, None, MapMemory()) == "stall"


def test_run_pokedex_return_stops_if_the_pokedex_is_not_delivered(monkeypatch):
    _wire_return(monkeypatch, hops=["arrived"], flora=True,
                 descent=["arrived"], cutscene=False)
    assert run_pokedex_return(None, None, MapMemory()) == "pokedex_not_delivered"


def test_run_lab_pokedex_cutscene_true_when_var_reaches_five():
    class _R:
        def _save_block1(self): return b"sb1"
        def _var(self, sb1, addr): return 5
        def has_pokedex(self): return True

    class _E:
        def step(self, keys, frames): pass

    assert campaign._run_lab_pokedex_cutscene(_E(), _R(), max_presses=10) is True


def test_run_lab_pokedex_cutscene_false_when_var_never_reaches_five():
    class _R:
        def _save_block1(self): return b"sb1"
        def _var(self, sb1, addr): return 4
        def has_pokedex(self): return False

    class _E:
        def step(self, keys, frames): pass

    assert campaign._run_lab_pokedex_cutscene(_E(), _R(), max_presses=5) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -k "pokedex_return or lab_pokedex_cutscene or return_directions" -v`
Expected: FAIL — `ImportError: cannot import name 'ROUTE_103'` / `run_pokedex_return` not defined.

- [ ] **Step 3: Implement the constants, helper, and orchestrator**

In `env/campaign.py`, extend the imports. Change line 18 from `from env.map_traveler import reach_map` to:

```python
from env.map_traveler import cross_scripted_npc, reach_map
```

Add after the `from typing import Any` import (top of file, near line 16):

```python
from emulator import buttons
```

Replace the constants block (lines 21-29) with:

```python
# Map-group ids on the southbound return path.
ROUTE_103 = (0, 18)
ROUTE_101 = (0, 16)
OLDALE = (0, 10)
LITTLEROOT = (0, 9)
LAB = (1, 4)

# Return chain for reach_map, reused for both A2 legs: descend route_103 -> Oldale,
# then (after the Flora gate) route_101 -> Littleroot, then the lab door warp (up).
_RETURN_DIRECTIONS: dict[tuple[int, int], str] = {
    ROUTE_103: "down", ROUTE_101: "down", LITTLEROOT: "up",
}

# Lab OnFrame GivePokedex gate: map_script_2 fires GivePokedexEvent when this VAR == 4;
# the script steps it 4 -> 5 as the Pokédex is delivered (pret/pokeemerald vars.h).
VAR_BIRCH_LAB_STATE = 0x4084

# Flora gate: she stands on Oldale's south connection tile (11,19). Stand just north
# at (10,19) facing EAST, A-spam her dialogue, then DOWN crosses into route_101.
_FLORA_STAND = (10, 19)
_FLORA_FACE = "right"
_FLORA_CROSS = "down"
_FLORA_MAX_PRESSES = 60

# Lab cutscene: idle first so the OnFrame applymovement auto-walk fires (no input),
# then A-spam the dialogue. ~185 presses observed; 400 is a safe bound.
_LAB_CUTSCENE_IDLE_FRAMES = 64
_LAB_CUTSCENE_MAX_PRESSES = 400
_LAB_ASPAM_FRAMES = 8
```

Add these two functions after `run_campaign` (end of file):

```python
def _run_lab_pokedex_cutscene(emulator: Any, reader: Any, *, max_presses: int) -> bool:
    """Run the lab OnFrame GivePokedex cutscene: idle so the auto-walk fires, then
    A-spam the dialogue until VAR_BIRCH_LAB_STATE steps 4 -> 5. Returns True iff the
    Pokédex was delivered (VAR == 5 and has_pokedex())."""
    emulator.step(0, _LAB_CUTSCENE_IDLE_FRAMES)   # OnFrame applymovement needs no input
    for _ in range(max_presses):
        sb1 = reader._save_block1()
        if sb1 is not None and reader._var(sb1, VAR_BIRCH_LAB_STATE) == 5:
            return reader.has_pokedex()
        emulator.step(buttons.KEY_A, _LAB_ASPAM_FRAMES)
        emulator.step(0, _LAB_ASPAM_FRAMES)
    sb1 = reader._save_block1()
    return (
        sb1 is not None
        and reader._var(sb1, VAR_BIRCH_LAB_STATE) == 5
        and reader.has_pokedex()
    )


def run_pokedex_return(
    emulator: Any,
    reader: Any,
    memory: Any,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Drive post_rival.state back to Birch's lab and deliver the Pokédex.

    reach_map descends route_103 -> Oldale; cross_scripted_npc plays Flora's gate into
    route_101; reach_map descends route_101 -> Littleroot -> lab; then the lab OnFrame
    GivePokedex cutscene runs. Every leg is bounded and surfaces the first failure.

    Returns 'pokedex_delivered' on success, or on failure the first failing leg's status:
    the reach_map leg outcomes ('stall' | 'timeout' | 'battle_lost' | 'battle_timeout' |
    'battle_interrupted') are propagated verbatim; the two boolean legs surface
    'flora_no_cross' | 'pokedex_not_delivered'.
    """
    hopped = reach_map(
        emulator, reader, memory, OLDALE, _RETURN_DIRECTIONS,
        move_type_fn=move_type_fn, predict=predict,
    )
    if hopped != "arrived":
        return hopped

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

    if not _run_lab_pokedex_cutscene(emulator, reader, max_presses=_LAB_CUTSCENE_MAX_PRESSES):
        return "pokedex_not_delivered"
    return "pokedex_delivered"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -q`
Expected: all pass (the updated `_RETURN_DIRECTIONS` pin + the new orchestration tests).

- [ ] **Step 5: Run ruff**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/campaign.py tests/test_campaign.py`
Expected: ruff clean.

- [ ] **Step 6: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "$(cat <<'EOF'
feat: run_pokedex_return driver (reach Oldale, Flora gate, descend, GivePokedex cutscene)

New campaign orchestrator: reach_map descends route_103 -> Oldale, cross_scripted_npc
plays Flora's gate into route_101, reach_map descends to the lab, then the lab OnFrame
cutscene runs (idle auto-walk + bounded A-spam until VAR_BIRCH_LAB_STATE 4->5). Re-introduce
OLDALE/ROUTE_103/VAR_BIRCH_LAB_STATE + Flora constants; extend _RETURN_DIRECTIONS with
route_103 (reused for both legs). No heal step (the Fighter handles wild battles in-loop;
the probe's _heal_party was a RAM poke). Pure orchestration unit tests stub each leg.

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

The single load-bearing proof of the A2 chain: reach_map descends route_103 -> Oldale,
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

    # run_pokedex_return needs snapshot/grid_reader (world) AND _var/has_pokedex (reader);
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
Expected: 1 passed (the branch executes; `has_pokedex()` True, `map_id == LAB`, `states/post_pokedex.state` written). If it fails, do NOT patch the test to pass — return to the Task 1 probe output and systematic-debugging to find which leg regressed.

- [ ] **Step 3: Run ruff**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && /Users/_eloi/Projets/Emu/.venv/bin/ruff check tests/test_pokedex_return_rom.py`
Expected: ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pokedex_return_rom.py
git commit -m "$(cat <<'EOF'
test: gated ROM smoke — run_pokedex_return delivers the Pokedex end to end

Load-bearing proof of the A2 chain from post_rival.state: reach Oldale, Flora gate,
descend to the lab, GivePokedex cutscene. Asserts has_pokedex() True + lab arrival and
dumps states/post_pokedex.state. Triple-skips without ROM / Fighter / post_rival.state.

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

- [ ] **Step 3: Confirm the throwaway probe is uncommitted**

Run: `cd /Users/_eloi/Projets/Emu-pokedex-return && git status --short tools/probe_pokedex_return_gate.py`
Expected: shows `??` (untracked) — the probe must NOT be committed.

Then use superpowers:finishing-a-development-branch to complete the branch.
