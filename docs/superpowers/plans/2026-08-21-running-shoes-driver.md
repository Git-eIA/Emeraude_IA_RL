# Running Shoes Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the GivePokedex cutscene properly inside `run_pokedex_return` (control returned), then a new `run_shoes_leg` driver exits the lab, triggers the mom/running-shoes event, and verifies control — producing a healthy `states/post_shoes.state`.

**Architecture:** Pure orchestration over existing primitives (spec: `docs/superpowers/specs/2026-08-21-running-shoes-driver-design.md`). One new public reader method (`birch_lab_state`), one private cutscene finisher (`_finish_lab_cutscene`, DRY-reuses `_advance_story_dialogue`), one new driver (`run_shoes_leg`) built from three private bounded leg helpers plus an intentionally-ignored `hop_via_explore` call. Two gated ROM smokes pin the dump-then-reload control contract.

**Tech Stack:** Python 3.12, pytest (fakes, no ROM for unit tests), mgba via `emulator.gba.GbaEmulator` for gated smokes, ruff (line-length 100).

**Ground truth (probe, 2026-08-21):** cutscene completes at ~150 A from `lab_arrival` (dex + 5 balls + `VAR_BIRCH_LAB_STATE`==5), B x10 drains re-opened Birch boxes, DOWN x11 exits the lab, the mom event intercepts ANY northbound nav in Littleroot (`hop_via_explore` returns `no_portal` — expected), ~14 (A x4 + B) cycles flip FLAG 0x112 then `town_state` 3→4, control returns immediately after.

## File Structure

- Modify: `env/game_state.py` — add `VAR_BIRCH_LAB_STATE` + `EmeraldReader.birch_lab_state()`
- Modify: `env/campaign.py` — constants, `_press`, `_finish_lab_cutscene`, `_exit_lab`, `_drain_mom_event`, `_verify_control`, `run_shoes_leg`; wire `_finish_lab_cutscene` into `run_pokedex_return`
- Modify: `tests/test_game_state.py` — extend `build_memory` with `lab_state`, 3 new tests
- Modify: `tests/test_campaign.py` — cutscene-finisher tests, leg-helper tests, `run_shoes_leg` orchestration tests, update `_wire_return`
- Modify: `tests/conftest.py` — shared `control_returns(emu)` ROM helper
- Modify: `tests/test_pokedex_return_rom.py` — completion asserts + dump-then-reload control pin
- Create: `tests/test_shoes_leg_rom.py` — gated shoes smoke, dumps `states/post_shoes.state`

All commands run from the worktree root. Use `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest` (the venv lives in the main checkout). ROM smokes are NOT run in the worktree (no `states/`); they run from `/Users/_eloi/Projets/Emu` after merge (see Task 5).

---

### Task 1: `EmeraldReader.birch_lab_state()`

**Files:**
- Modify: `env/game_state.py`
- Test: `tests/test_game_state.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_game_state.py`, extend `build_memory` — add the `lab_state` keyword parameter and plant the var. The signature becomes:

```python
def build_memory(
    *,
    x: int,
    y: int,
    map_group: int,
    map_num: int,
    badge_bits: int,
    party_count: int,
    clock_set: bool = False,
    town_state: int = 0,
    lab_state: int = 0,
) -> dict[int, bytes]:
```

and right after the existing town_state line (`save_block1[0x139C + 0xA0 : ...]`), add:

```python
    # VAR_BIRCH_LAB_STATE = 0x4084 -> vars index 0x84, u16 LE
    save_block1[0x139C + 0x108 : 0x139C + 0x10A] = lab_state.to_bytes(2, "little")
```

Then add the three tests (after `test_read_flag_invalid_pointer_is_false`):

```python
def test_birch_lab_state_reads_the_var():
    memory = build_memory(
        x=6, y=5, map_group=1, map_num=4, badge_bits=0, party_count=1, lab_state=5
    )
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.birch_lab_state() == 5


def test_birch_lab_state_zero_by_default():
    memory = build_memory(x=6, y=5, map_group=1, map_num=4, badge_bits=0, party_count=1)
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.birch_lab_state() == 0


def test_birch_lab_state_invalid_pointer_is_none():
    memory = {SAVE_BLOCK1_PTR: (0x00000000).to_bytes(4, "little")}
    reader = EmeraldReader(make_fake_read(memory))
    assert reader.birch_lab_state() is None
```

(No new imports needed: `EmeraldReader`, `SAVE_BLOCK1_PTR`, `build_memory`, `make_fake_read` are already in scope.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_game_state.py -k birch -v`
Expected: 3 FAILED with `AttributeError: 'EmeraldReader' object has no attribute 'birch_lab_state'`

- [ ] **Step 3: Implement `birch_lab_state`**

In `env/game_state.py`, below `VAR_LITTLEROOT_TOWN_STATE = 0x4050` add:

```python
# The lab OnFrame GivePokedex cutscene steps this var 4 -> 5 exactly when the script
# completes (Pokédex + 5 Poke Balls delivered, releaseall run). Public read so drivers
# can detect real completion instead of stopping at the early has_pokedex() flag.
VAR_BIRCH_LAB_STATE = 0x4084
```

In `EmeraldReader`, after `has_running_shoes()` add:

```python
    def birch_lab_state(self) -> int | None:
        """VAR_BIRCH_LAB_STATE value, or None while save blocks are relocating."""
        sb1 = self._save_block1()
        if sb1 is None:
            return None
        return self._var(sb1, VAR_BIRCH_LAB_STATE)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_game_state.py -v`
Expected: all PASS (the 3 new ones plus the existing file).

- [ ] **Step 5: Commit**

```bash
git add env/game_state.py tests/test_game_state.py
git commit -m "feat: EmeraldReader.birch_lab_state reads VAR_BIRCH_LAB_STATE (0x4084)"
```

---

### Task 2: `_finish_lab_cutscene` wired into `run_pokedex_return`

**Files:**
- Modify: `env/campaign.py`
- Test: `tests/test_campaign.py`

Context: today `run_pokedex_return` ends with `_advance_story_dialogue(emulator, reader, lambda r: r.has_pokedex())` — that predicate is satisfied ~55 A-presses BEFORE the balls land and the script's `releaseall`, so the driver returns mid-cutscene and the dumped state looks control-locked (the 2026-08-17 false-lock crux). The fix: A-spam until the COMBINED predicate, then B x10 to drain the Birch dialogue boxes the extra A-presses re-opened.

- [ ] **Step 1: Write the failing tests**

In `tests/test_campaign.py`: add imports at the top —

```python
from emulator import buttons
from env.campaign import _finish_lab_cutscene
```

(add `_finish_lab_cutscene` inside the existing `from env.campaign import (...)` block, alphabetically after `_RETURN_DIRECTIONS`; add `from emulator import buttons` with the other top-level imports).

At the end of the file add:

```python
# ---------------------------------------------------------------------------
# Shoes driver: _finish_lab_cutscene
# ---------------------------------------------------------------------------


class _RecordingEmu:
    """Records every step; scripted readers key off the A-press count."""

    def __init__(self):
        self.steps = []

    @property
    def a_presses(self):
        return sum(1 for key, _ in self.steps if key == buttons.KEY_A)

    def step(self, keys, frames):
        self.steps.append((keys, frames))


class _CutsceneReader:
    """has_pokedex is True from press 0 (the flag flips ~55 presses early on real
    hardware); balls and lab_state==5 only land after presses_to_done A-presses."""

    def __init__(self, emu, presses_to_done):
        self._emu = emu
        self._presses_to_done = presses_to_done

    def has_pokedex(self):
        return True

    def has_poke_balls(self, min_qty):
        return self._emu.a_presses >= self._presses_to_done

    def birch_lab_state(self):
        return 5 if self._emu.a_presses >= self._presses_to_done else 4


def test_finish_lab_cutscene_does_not_stop_at_the_pokedex_flag():
    # Anti-false-lock pin: stopping at has_pokedex() alone is the exact bug that
    # produced the mid-cutscene dump; the A-spam must run until balls + lab_state==5.
    emu = _RecordingEmu()
    assert _finish_lab_cutscene(emu, _CutsceneReader(emu, presses_to_done=7)) is True
    assert emu.a_presses == 7


def test_finish_lab_cutscene_releases_with_b_after_completion():
    emu = _RecordingEmu()
    assert _finish_lab_cutscene(emu, _CutsceneReader(emu, presses_to_done=3)) is True
    b_count = sum(1 for key, _ in emu.steps if key == buttons.KEY_B)
    assert b_count == 10
    last_a = max(i for i, (key, _) in enumerate(emu.steps) if key == buttons.KEY_A)
    first_b = min(i for i, (key, _) in enumerate(emu.steps) if key == buttons.KEY_B)
    assert first_b > last_a  # release strictly follows the drain


def test_finish_lab_cutscene_times_out_without_b_release():
    emu = _RecordingEmu()
    reader = _CutsceneReader(emu, presses_to_done=10_000)  # never completes
    assert _finish_lab_cutscene(emu, reader) is False
    assert not any(key == buttons.KEY_B for key, _ in emu.steps)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -k finish_lab -v`
Expected: FAIL at import time — `ImportError: cannot import name '_finish_lab_cutscene'`

- [ ] **Step 3: Implement `_finish_lab_cutscene`**

In `env/campaign.py`: add `from emulator import buttons` with the imports. Below the `_FLORA_*` constants add:

```python
# GivePokedex cutscene completion (2026-08-17 investigation + 2026-08-21 probe):
# the Pokédex flag flips ~55 A-presses BEFORE the 5 Poke Balls land and the script's
# releaseall; continued A-spam re-talks Birch, so B presses must drain the re-opened
# dialogue boxes before any direction input can move the player.
_RELEASE_B_PRESSES = 10  # 5+ works, 3 insufficient (probe-measured)
_BUTTON_FRAMES = 8       # A/B press and release frames (probe-proven cadence)
```

Below `run_campaign` (before `run_pokedex_return`) add:

```python
def _press(emulator: Any, key: int, hold: int, rest: int) -> None:
    """Press one key for hold frames, then rest with no input."""
    emulator.step(key, hold)
    emulator.step(0, rest)


def _finish_lab_cutscene(emulator: Any, reader: Any) -> bool:
    """Play the GivePokedex cutscene to completion and return WITH control.

    Stopping at has_pokedex() alone dumps a mid-cutscene state that resumes as a
    false control-lock; real completion is dex + 5 balls + VAR_BIRCH_LAB_STATE==5.
    After that, B presses close the Birch boxes the extra A-spam re-opened."""
    done = _advance_story_dialogue(
        emulator, reader,
        lambda r: r.has_pokedex() and r.has_poke_balls(5) and r.birch_lab_state() == 5,
    )
    if done != "story_done":
        return False
    for _ in range(_RELEASE_B_PRESSES):
        _press(emulator, buttons.KEY_B, _BUTTON_FRAMES, _BUTTON_FRAMES)
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -k finish_lab -v`
Expected: 3 PASS

- [ ] **Step 5: Wire into `run_pokedex_return` and update the orchestration stubs**

In `env/campaign.py`, replace the final leg of `run_pokedex_return`:

```python
    if _advance_story_dialogue(emulator, reader, lambda r: r.has_pokedex()) != "story_done":
        return "pokedex_not_delivered"
    return "pokedex_delivered"
```

with:

```python
    if not _finish_lab_cutscene(emulator, reader):
        return "pokedex_not_delivered"
    return "pokedex_delivered"
```

In the `run_pokedex_return` docstring, replace the sentence "then _advance_story_dialogue runs the lab OnFrame GivePokedex cutscene until has_pokedex()." with "then _finish_lab_cutscene plays the lab OnFrame GivePokedex cutscene to completion (dex + 5 balls + VAR_BIRCH_LAB_STATE==5) and drains the re-opened Birch dialogue with B, so the driver returns WITH control." Keep the status contract line unchanged.

In `tests/test_campaign.py`, update `_wire_return`: replace the `_advance_story_dialogue` stub

```python
    monkeypatch.setattr(
        campaign, "_advance_story_dialogue",
        lambda *a, **k: (calls.append(("cutscene",)), cutscene)[1],
    )
```

with

```python
    monkeypatch.setattr(
        campaign, "_finish_lab_cutscene",
        lambda *a, **k: (calls.append(("cutscene",)), cutscene)[1],
    )
```

and update the five existing orchestration tests: every `cutscene="story_done"` becomes `cutscene=True`, and in `test_run_pokedex_return_stops_if_the_pokedex_is_not_delivered` the `cutscene="story_timeout"` becomes `cutscene=False`.

- [ ] **Step 6: Run the campaign suite**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -v`
Expected: all PASS (including the five updated orchestration tests).

- [ ] **Step 7: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "feat: run_pokedex_return finishes the GivePokedex cutscene and returns with control"
```

---

### Task 3: `run_shoes_leg` and its leg helpers

**Files:**
- Modify: `env/campaign.py`
- Test: `tests/test_campaign.py`

- [ ] **Step 1: Write the failing helper tests**

In `tests/test_campaign.py`, add `from types import SimpleNamespace` to the top-level imports, and extend the `from env.campaign import (...)` block with `_drain_mom_event`, `_exit_lab`, `_verify_control`, `run_shoes_leg`. Then append:

```python
# ---------------------------------------------------------------------------
# Shoes driver: leg helpers
# ---------------------------------------------------------------------------


class _WalkEmu(_RecordingEmu):
    @property
    def down_presses(self):
        return sum(1 for key, _ in self.steps if key == buttons.KEY_DOWN)

    @property
    def b_presses(self):
        return sum(1 for key, _ in self.steps if key == buttons.KEY_B)


class _ExitReader:
    """In the lab until downs_needed DOWN presses have landed, then Littleroot."""

    def __init__(self, emu, downs_needed):
        self._emu = emu
        self._downs_needed = downs_needed

    def player_state(self):
        where = LITTLEROOT if self._emu.down_presses >= self._downs_needed else LAB
        return SimpleNamespace(map_group=where[0], map_num=where[1], x=7, y=16, town_state=3)


def test_exit_lab_stops_when_littleroot_is_reached():
    emu = _WalkEmu()
    assert _exit_lab(emu, _ExitReader(emu, downs_needed=11)) is True
    assert emu.down_presses == 11  # probe measured 11; no extra presses after arrival


def test_exit_lab_times_out_when_the_map_never_changes():
    emu = _WalkEmu()
    assert _exit_lab(emu, _ExitReader(emu, downs_needed=10_000)) is False
    assert emu.down_presses == 60  # bounded at _LAB_EXIT_MAX_PRESSES


class _DrainReader:
    """Shoes land after shoes_after A presses; town_state hits 4 after town4_after."""

    def __init__(self, emu, shoes_after, town4_after):
        self._emu = emu
        self._shoes_after = shoes_after
        self._town4_after = town4_after

    def has_running_shoes(self):
        return self._emu.a_presses >= self._shoes_after

    def player_state(self):
        ts = 4 if self._emu.a_presses >= self._town4_after else 3
        return SimpleNamespace(map_group=0, map_num=9, x=10, y=9, town_state=ts)


def test_drain_mom_event_requires_shoes_and_town_state_4():
    # Pin the AND: shoes at 8 presses, town_state 4 only at 16 — the drain must
    # NOT stop at the shoes flag alone (same class of bug as the pokedex false-lock).
    emu = _WalkEmu()
    assert _drain_mom_event(emu, _DrainReader(emu, shoes_after=8, town4_after=16)) is True
    assert emu.a_presses == 16


def test_drain_mom_event_times_out():
    emu = _WalkEmu()
    reader = _DrainReader(emu, shoes_after=10**6, town4_after=10**6)
    assert _drain_mom_event(emu, reader) is False
    assert emu.a_presses == 80 * 4  # bounded at _SHOES_MAX_CYCLES x _SHOES_A_PER_CYCLE


class _ControlReader:
    """Position tracks DOWN presses only once b_needed B presses drained the boxes."""

    def __init__(self, emu, b_needed):
        self._emu = emu
        self._b_needed = b_needed

    def player_state(self):
        moved = self._emu.b_presses >= self._b_needed
        y = 9 + (self._emu.down_presses if moved else 0)
        return SimpleNamespace(map_group=0, map_num=9, x=10, y=y, town_state=4)


def test_verify_control_succeeds_when_a_down_press_moves():
    emu = _WalkEmu()
    assert _verify_control(emu, _ControlReader(emu, b_needed=0)) is True
    assert emu.down_presses == 1


def test_verify_control_drains_a_reopened_box_with_b_then_succeeds():
    # Probe P6 pattern: the drain's last A may have re-opened a box, so a failed
    # DOWN is followed by B x2 before retrying.
    emu = _WalkEmu()
    assert _verify_control(emu, _ControlReader(emu, b_needed=2)) is True
    assert emu.down_presses == 2
    assert emu.b_presses == 2


def test_verify_control_times_out_when_nothing_ever_moves():
    class _Frozen:
        def player_state(self):
            return SimpleNamespace(map_group=1, map_num=4, x=6, y=5, town_state=3)

    emu = _WalkEmu()
    assert _verify_control(emu, _Frozen()) is False
    assert emu.down_presses == 30  # bounded at _CONTROL_MAX_CYCLES
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -v`
Expected: FAIL at import time — `ImportError: cannot import name '_drain_mom_event'`

- [ ] **Step 3: Implement the leg helpers**

In `env/campaign.py`, below `_BUTTON_FRAMES` add:

```python
# run_shoes_leg bounds (probe-measured 2026-08-21, margin >= x2). Direction presses
# use the 12/4 cadence the probes and precision walks share.
_MOVE_PRESS_FRAMES = 12
_MOVE_REST_FRAMES = 4
_LAB_EXIT_MAX_PRESSES = 60  # measured: 11 DOWN presses lab -> Littleroot
_SHOES_MAX_CYCLES = 80      # measured: ~14 cycles to shoes + town_state 4
_SHOES_A_PER_CYCLE = 4
_CONTROL_MAX_CYCLES = 30    # measured: 1 cycle for control to return
_CONTROL_B_PRESSES = 2      # drain a box the drain's last A re-opened (probe P6)
```

Below `_finish_lab_cutscene` add:

```python
def _exit_lab(emulator: Any, reader: Any) -> bool:
    """Walk DOWN out of the lab until the map is Littleroot (bounded)."""
    for _ in range(_LAB_EXIT_MAX_PRESSES):
        _press(emulator, buttons.KEY_DOWN, _MOVE_PRESS_FRAMES, _MOVE_REST_FRAMES)
        ps = reader.player_state()
        if ps is not None and (ps.map_group, ps.map_num) == LITTLEROOT:
            return True
    return False


def _drain_mom_event(emulator: Any, reader: Any) -> bool:
    """A/B cycles until the shoes land AND town_state reaches 4 (bounded).

    The shoes flag flips before the event script finishes; town_state 3 -> 4 marks
    real completion, so both are required (anti-false-lock, same as the cutscene)."""

    def _done() -> bool:
        ps = reader.player_state()
        return reader.has_running_shoes() and ps is not None and ps.town_state == 4

    # Check-first loop with a final re-check: presses are bounded at exactly
    # _SHOES_MAX_CYCLES cycles, and a state completed BY the last cycle's presses
    # is still detected by the trailing _done().
    for _ in range(_SHOES_MAX_CYCLES):
        if _done():
            return True
        for _ in range(_SHOES_A_PER_CYCLE):
            _press(emulator, buttons.KEY_A, _BUTTON_FRAMES, _BUTTON_FRAMES)
        _press(emulator, buttons.KEY_B, _BUTTON_FRAMES, _BUTTON_FRAMES)
    return _done()


def _verify_control(emulator: Any, reader: Any) -> bool:
    """Prove control returned: a DOWN press changes the position or map (bounded).

    A still-open dialogue box swallows direction input, so each failed press is
    followed by B presses to drain it before retrying (probe P6 pattern)."""
    for _ in range(_CONTROL_MAX_CYCLES):
        before = reader.player_state()
        _press(emulator, buttons.KEY_DOWN, _MOVE_PRESS_FRAMES, _MOVE_REST_FRAMES)
        after = reader.player_state()
        if (
            before is not None and after is not None
            and ((before.x, before.y) != (after.x, after.y)
                 or (before.map_group, before.map_num) != (after.map_group, after.map_num))
        ):
            return True
        for _ in range(_CONTROL_B_PRESSES):
            _press(emulator, buttons.KEY_B, _BUTTON_FRAMES, _BUTTON_FRAMES)
    return False
```

- [ ] **Step 4: Run the helper tests**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -k "exit_lab or drain_mom or verify_control" -v`
Expected: 7 PASS

- [ ] **Step 5: Write the failing orchestration tests**

Append to `tests/test_campaign.py`:

```python
# ---------------------------------------------------------------------------
# Shoes driver: run_shoes_leg orchestration
# ---------------------------------------------------------------------------


def _wire_shoes(monkeypatch, *, exit_lab, hop, drain, control):
    """Stub the four legs of run_shoes_leg; record their calls in order."""
    calls = []
    monkeypatch.setattr(
        campaign, "_exit_lab", lambda *a, **k: (calls.append("exit"), exit_lab)[1],
    )
    monkeypatch.setattr(
        campaign, "hop_via_explore",
        lambda *a, **k: (calls.append(("hop", a[3], a[4], a[5])), hop)[1],
    )
    monkeypatch.setattr(
        campaign, "_drain_mom_event", lambda *a, **k: (calls.append("drain"), drain)[1],
    )
    monkeypatch.setattr(
        campaign, "_verify_control", lambda *a, **k: (calls.append("control"), control)[1],
    )
    return calls


def test_run_shoes_leg_delivers_and_ignores_the_hop_result(monkeypatch):
    # 'no_portal' is the EXPECTED hop outcome — the mom event freezes the sweep;
    # the leg must succeed regardless of what hop_via_explore returns.
    calls = _wire_shoes(monkeypatch, exit_lab=True, hop="no_portal", drain=True, control=True)
    assert run_shoes_leg(_Emu(), None, MapMemory()) == "shoes_delivered"
    assert calls == ["exit", ("hop", LITTLEROOT, ROUTE_101, "up"), "drain", "control"]


def test_run_shoes_leg_surfaces_the_lab_exit_timeout(monkeypatch):
    # A mid-cutscene start (stale post_pokedex.state) fails here cleanly.
    calls = _wire_shoes(monkeypatch, exit_lab=False, hop="arrived", drain=True, control=True)
    assert run_shoes_leg(_Emu(), None, MapMemory()) == "lab_exit_timeout"
    assert calls == ["exit"]  # no hop/drain after a failed exit


def test_run_shoes_leg_surfaces_the_shoes_timeout(monkeypatch):
    _wire_shoes(monkeypatch, exit_lab=True, hop="arrived", drain=False, control=True)
    assert run_shoes_leg(_Emu(), None, MapMemory()) == "shoes_timeout"


def test_run_shoes_leg_surfaces_the_control_timeout(monkeypatch):
    _wire_shoes(monkeypatch, exit_lab=True, hop="arrived", drain=True, control=False)
    assert run_shoes_leg(_Emu(), None, MapMemory()) == "control_timeout"
```

- [ ] **Step 6: Run to verify they fail**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -k run_shoes_leg -v`
Expected: FAIL at import time — `ImportError: cannot import name 'run_shoes_leg'`

- [ ] **Step 7: Implement `run_shoes_leg`**

In `env/campaign.py`, after `_verify_control` add:

```python
def run_shoes_leg(
    emulator: Any,
    reader: Any,
    memory: Any,
    *,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Drive a healthy post-Pokédex lab state through the mom/running-shoes event.

    Exits the lab (bounded DOWN walk), then walks north via hop_via_explore whose
    result is DELIBERATELY ignored: the scripted mom event intercepts any northbound
    nav in Littleroot, so 'no_portal' IS the expected success path — the
    shoes/town_state predicate is the arbiter, not the hop status. Bounded A/B
    cycles drain the event, then a DOWN press proves control returned. The reader
    is the same composite contract as run_pokedex_return (WorldReader snapshot/grid
    attributes for the hop, EmeraldReader flags/vars for the predicates).

    Returns 'shoes_delivered' | 'lab_exit_timeout' | 'shoes_timeout' |
    'control_timeout'.
    """
    if not _exit_lab(emulator, reader):
        return "lab_exit_timeout"
    emulator.step(0, _SETTLE_FRAMES)
    # Result intentionally unchecked — see docstring; a timeout in the drain below
    # surfaces honestly if the event unexpectedly never fires.
    hop_via_explore(
        emulator, reader, memory, LITTLEROOT, ROUTE_101, "up",
        move_type_fn=move_type_fn, predict=predict,
    )
    if not _drain_mom_event(emulator, reader):
        return "shoes_timeout"
    if not _verify_control(emulator, reader):
        return "control_timeout"
    return "shoes_delivered"
```

- [ ] **Step 8: Run the full campaign suite**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_campaign.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "feat: run_shoes_leg drives the mom/running-shoes event from the lab"
```

---

### Task 4: ROM smokes (control pins + post_shoes dump)

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_pokedex_return_rom.py`
- Create: `tests/test_shoes_leg_rom.py`

These tests are ROM-gated: in the worktree they must COLLECT cleanly and SKIP (no ROM/state there). The real runs happen from `/Users/_eloi/Projets/Emu` in Task 5. No failing-test-first cycle is possible without the ROM; the discipline here is: write, verify collection+skip, review, commit.

- [ ] **Step 1: Add the shared control helper to `tests/conftest.py`**

Append at the end of `tests/conftest.py`:

```python
def control_returns(emu) -> bool:
    """True when a direction press moves the player within a few tries (ROM smokes).

    Anti-false-lock pin shared by the pokedex and shoes smokes: presses DOWN
    (12/4 frames); if the player did not move, drains a possibly open dialogue
    box with B x2 before retrying (the mom-event probe's P6 pattern)."""
    from emulator import buttons
    from env.game_state import EmeraldReader

    reader = EmeraldReader(emu.read_bytes)
    for _ in range(5):
        before = reader.player_state()
        emu.step(buttons.KEY_DOWN, 12)
        emu.step(0, 4)
        after = reader.player_state()
        if (
            before is not None and after is not None
            and ((before.x, before.y) != (after.x, after.y)
                 or (before.map_group, before.map_num) != (after.map_group, after.map_num))
        ):
            return True
        for _ in range(2):
            emu.step(buttons.KEY_B, 8)
            emu.step(0, 8)
    return False
```

- [ ] **Step 2: Extend `tests/test_pokedex_return_rom.py`**

Update the module docstring's last sentence to: "Asserts has_pokedex() True + 5 balls + lab arrival, dumps states/post_pokedex.state FIRST, then reloads the dump and verifies control (anti-false-lock pin). Slow (~minute, Fighter-driven). Triple-skips without ROM / Fighter checkpoint / post_rival.state."

Replace the tail of `test_run_pokedex_return_delivers_the_pokedex` — the current last line

```python
    Path("states/post_pokedex.state").write_bytes(emu.save_state())
```

with:

```python
    # _finish_lab_cutscene now guarantees full delivery, not just the early flag.
    assert reader.has_poke_balls(5) is True
    assert reader.birch_lab_state() == 5

    # Dump FIRST, then reload the dump and verify control on the RELOADED session:
    # the pin must prove the DUMP is healthy, and verification must not mutate it.
    state_bytes = emu.save_state()
    Path("states/post_pokedex.state").write_bytes(state_bytes)

    from tests.conftest import control_returns

    emu2 = GbaEmulator(ROM)
    emu2.load_state(state_bytes)
    emu2.step(0, 4)
    assert control_returns(emu2), "reloaded post_pokedex.state is control-locked"
```

- [ ] **Step 3: Create `tests/test_shoes_leg_rom.py`**

```python
"""Gated ROM smoke: run_shoes_leg delivers the running shoes end to end.

From a healthy (post-release) post_pokedex.state: exit the lab, walk north into the
scripted mom interception, drain the event until FLAG 0x112 + town_state==4, verify
control, and dump states/post_shoes.state — then reload the dump and re-verify control
(anti-false-lock pin: the dump must be healthy, not just the live session). No Fighter:
Littleroot and the lab have no wild grass, so no battle can interrupt the leg.
Double-skips without ROM / post_pokedex.state. Run AFTER test_pokedex_return_rom.py,
which re-dumps post_pokedex.state healthy (the pre-existing dump was mid-cutscene).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROM = os.environ.get("POKEMON_EMERALD_ROM")
STATE = "states/post_pokedex.state"

pytestmark = [
    pytest.mark.skipif(not ROM, reason="POKEMON_EMERALD_ROM not set"),
    pytest.mark.skipif(not Path(STATE).exists(), reason="post_pokedex.state missing"),
]


def test_run_shoes_leg_delivers_the_running_shoes() -> None:
    from emulator.gba import GbaEmulator
    from env.campaign import run_shoes_leg
    from env.game_state import EmeraldReader
    from env.map_memory import MapMemory
    from env.world_reader import WorldReader
    from tests.conftest import control_returns

    emu = GbaEmulator(ROM)
    with open(STATE, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)

    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)
    memory = MapMemory()

    # run_shoes_leg needs snapshot/grid_reader (world) AND flags/vars (reader);
    # a thin adapter forwards both, matching test_pokedex_return_rom.py.
    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_shoes_leg(emu, _Reader(), memory)

    assert result == "shoes_delivered", result
    assert reader.has_running_shoes() is True
    ps = reader.player_state()
    assert ps is not None and ps.town_state == 4, ps

    # Dump FIRST, then reload and verify control on the reloaded session (the pin
    # must prove the DUMP is healthy; verification never mutates the dumped state).
    state_bytes = emu.save_state()
    Path("states/post_shoes.state").write_bytes(state_bytes)

    emu2 = GbaEmulator(ROM)
    emu2.load_state(state_bytes)
    emu2.step(0, 4)
    assert control_returns(emu2), "reloaded post_shoes.state is control-locked"
```

- [ ] **Step 4: Verify collection and clean skip (no ROM in the worktree)**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest tests/test_shoes_leg_rom.py tests/test_pokedex_return_rom.py -v`
Expected: 2 SKIPPED (no ROM env var in the worktree), 0 errors.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_pokedex_return_rom.py tests/test_shoes_leg_rom.py
git commit -m "test: gated ROM smokes — healthy post_pokedex re-dump + run_shoes_leg with dump-then-reload control pins"
```

---

### Task 5: Full verification

- [ ] **Step 1: Full unit suite + ruff in the worktree**

Run: `PYTHONPATH=. /Users/_eloi/Projets/Emu/.venv/bin/pytest -q` then `/Users/_eloi/Projets/Emu/.venv/bin/ruff check .`
Expected: all pass (baseline was 370 passed + skips; this plan adds 3 game_state + 3 cutscene + 7 helper + 4 orchestration = 17 unit tests, and converts 0 — expect ~387 passed), ruff clean. Fix anything that fails before proceeding.

- [ ] **Step 2: ROM smokes (from the MAIN checkout, after merge — controller runs this, not a subagent)**

From `/Users/_eloi/Projets/Emu`, ORDER MATTERS (A2 re-dumps the healthy state the shoes smoke consumes):

```bash
POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba PYTHONPATH=. .venv/bin/pytest \
  tests/test_pokedex_return_rom.py -v
POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba PYTHONPATH=. .venv/bin/pytest \
  tests/test_shoes_leg_rom.py -v
POKEMON_EMERALD_ROM=roms/pokemon_emerald_fr.gba PYTHONPATH=. .venv/bin/pytest \
  tests/test_balls_pocket_rom.py tests/test_flag_constants_rom.py tests/test_phase2_rom.py -v
```

Expected: pokedex smoke PASSES and re-dumps `states/post_pokedex.state` (now post-release); shoes smoke PASSES and dumps `states/post_shoes.state`; the third command confirms no regression on the other gated smokes (NOTE: `test_balls_pocket_rom.py` A-spams from `has_pokedex` — on the now-HEALTHY post-release state its `has_poke_balls(5)` is True at load, the bounded loop exits at iteration 0, still green).

- [ ] **Step 3: Cleanup (main checkout)**

Delete the throwaway probe: `rm tools/probe_mom_trigger.py` (uncommitted, stdout-only, superseded by the spec's ground-truth section).
