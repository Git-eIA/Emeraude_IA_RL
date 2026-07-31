# Grind Level-Up / Auto-Heal Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `"level_up"` Order mode that loops the single-battle `grind` primitive to raise the party to a target **average** level, auto-healing when HP gets low.

**Architecture:** A new `_execute_level_up` in `env/orders.py` composes the existing `_execute_grind` (one battle) and `_execute_heal` (one heal) primitives without modifying them, driven by two pure decision helpers (a mixed heal trigger, an average-level stop). The single-battle `grind` mode is untouched.

**Tech Stack:** Python 3.12, pytest (pure, no ROM), Stable-Baselines3-free (Fighter injected via `move_type_fn`/`predict`).

---

## File Structure

- `env/heal_detector.py` — add `party_needs_heal(hp, threshold)` pure helper (heal-decision domain; already holds `party_is_full`/`HealWatcher`).
- `env/world_reader.py` — add `party_levels()` passthrough (the shared reader must expose levels, like `party_hp()`).
- `env/orders.py` — add `_reached` helper, `_execute_level_up` loop, and the `mode == "level_up"` dispatch branch; extend `execute_order` with `target_level`/`heal_threshold`/`max_cycles` params; update the two docstrings.
- `tests/test_heal_detector.py` — tests for `party_needs_heal`.
- `tests/test_world_reader.py` — test for the `party_levels()` passthrough.
- `tests/test_orders.py` — a `FarmWorld` fake + the loop's behavior tests.

---

## Task 1: Mixed heal-trigger predicate

**Files:**
- Modify: `env/heal_detector.py`
- Test: `tests/test_heal_detector.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_heal_detector.py` (update the import line at the top too):

Change line 4 from:
```python
from env.heal_detector import HealWatcher, party_is_full
```
to:
```python
from env.heal_detector import HealWatcher, party_is_full, party_needs_heal
```

Append at the end of the file:
```python
def test_needs_heal_true_when_a_member_is_ko_even_if_totals_fine() -> None:
    # 5/10 total is above the 0.4 threshold, but a fainted member forces a heal.
    assert party_needs_heal([(0, 5), (5, 5)], 0.4) is True


def test_needs_heal_true_when_total_fraction_below_threshold() -> None:
    # Nobody KO'd, but 3/10 = 0.3 < 0.4.
    assert party_needs_heal([(1, 5), (2, 5)], 0.4) is True


def test_needs_heal_false_when_full_and_above_threshold() -> None:
    assert party_needs_heal([(5, 5), (4, 5)], 0.4) is False


def test_needs_heal_false_when_empty() -> None:
    assert party_needs_heal([], 0.4) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_heal_detector.py -v`
Expected: FAIL with `ImportError: cannot import name 'party_needs_heal'`.

- [ ] **Step 3: Write the implementation**

In `env/heal_detector.py`, add after `party_is_full` (before the `HealWatcher` class):
```python
def party_needs_heal(hp: list[tuple[int, int]], threshold: float) -> bool:
    """True if any member has fainted (0 HP) OR the party's total HP fraction is
    below `threshold`. False for an empty party. Mixed trigger: a KO forces a
    heal even when the totals look fine; a low total forces a heal even with
    nobody KO'd."""
    if not hp:
        return False
    if any(cur == 0 for cur, _ in hp):
        return True
    total_cur = sum(cur for cur, _ in hp)
    total_max = sum(mx for _, mx in hp)
    return total_max > 0 and total_cur / total_max < threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_heal_detector.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add env/heal_detector.py tests/test_heal_detector.py
git commit -m "feat: party_needs_heal — mixed KO-or-low-total heal trigger"
```

---

## Task 2: Expose party levels on the shared reader

**Files:**
- Modify: `env/world_reader.py`
- Test: `tests/test_world_reader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_world_reader.py`:
```python
def test_party_levels_passthrough_returns_ram_reader_levels() -> None:
    emu = FakeEmulator()
    emu.party_count = 2
    emu.party_levels = [7, 5]
    assert _reader(emu).party_levels() == [7, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_world_reader.py::test_party_levels_passthrough_returns_ram_reader_levels -v`
Expected: FAIL with `AttributeError: 'WorldReader' object has no attribute 'party_levels'`.

- [ ] **Step 3: Write the implementation**

In `env/world_reader.py`, add after `party_hp` (lines 39-41):
```python
    def party_levels(self) -> list[int]:
        """Passthrough to the RAM reader: level of each party member in order."""
        return self._reader.party_levels()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_world_reader.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add env/world_reader.py tests/test_world_reader.py
git commit -m "feat: WorldReader.party_levels passthrough"
```

---

## Task 3: The level-up loop + dispatch

**Files:**
- Modify: `env/orders.py`
- Test: `tests/test_orders.py`

This task builds a `FarmWorld` fake, then TDDs the loop. The fake plays BOTH emulator and reader: it snapshots at a fixed cell that is registered in memory as BOTH the grass spot and the healing spot, so `travel_to` always arrives immediately (no navigation). Treading (d-pad presses out of battle) triggers a scripted battle the injected Fighter wins in one turn; winning bumps the party level and applies post-battle damage; A-presses out of battle (the nurse) refill HP.

- [ ] **Step 1: Add the `FarmWorld` fake and the first two tests**

Append to `tests/test_orders.py` (after the existing grind+Fighter tests). It reuses the module-level `_KEY_TO_DIR`/`_u16b` already defined in this file:
```python
# ---------------------------------------------------------------------------
# Level-up loop tests
# ---------------------------------------------------------------------------


class FarmWorld:
    """Full grind-loop fake: treads -> wins a scripted 1-turn battle -> levels up
    and takes damage; A-presses out of battle (the Center) refill the party.

    Snapshots at a fixed cell that is BOTH the grass spot and the healing spot,
    so travel_to always arrives immediately. `party_levels()` reports the same
    level for every member, so the average equals `self._level`.
    """

    MAP = (0, 16)
    CELL = (5, 12)

    def __init__(
        self,
        start_level: int,
        target_hp_after: list[tuple[int, int]],
        levels_per_win: int = 1,
        steps_to_encounter: int = 3,
        party_size: int = 1,
        can_win: bool = True,
    ) -> None:
        self.map_id = self.MAP
        self.pos = self.CELL
        self._level = start_level
        self._levels_per_win = levels_per_win
        self._to_enc = steps_to_encounter
        self._party_size = party_size
        self._can_win = can_win
        self._hp_after = list(target_hp_after)
        self._hp: list[tuple[int, int]] = [(5, 5)] * party_size  # start full
        self.battles_won = 0
        self.heals = 0
        self._steps = 0
        self._battle = False
        self._phase = "menu"
        self._opp_hp = 6
        self._my_hp = 19
        self._outcome = 0

    def _start_battle(self) -> None:
        self._battle = True
        self._phase = "menu"
        self._opp_hp = 6
        self._outcome = 0
        self._steps = 0

    def _end_battle(self) -> None:
        if self._can_win:
            self._opp_hp = 0
            self._outcome = 1
            self._level += self._levels_per_win
            self.battles_won += 1
        else:
            self._outcome = 2   # terminal, not won -> play_battle returns "lost"
        self._battle = False
        self._hp = list(self._hp_after)

    def step(self, keys: int, frames: int) -> None:
        from emulator import buttons

        if not self._battle:
            if keys & buttons.KEY_A:                 # nurse dialog: refill to full
                self._hp = [(mx, mx) for _, mx in self._hp]
                self.heals += 1
                return
            if _KEY_TO_DIR.get(keys) is not None:    # treading
                self._steps += 1
                if self._steps >= self._to_enc:
                    self._start_battle()
            return
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            self._end_battle()

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_levels(self) -> list[int]:
        return [self._level] * self._party_size

    def party_hp(self) -> list[tuple[int, int]]:
        return list(self._hp)

    def in_battle(self) -> bool:
        return self._battle

    def read_bytes(self, addr: int, size: int) -> bytes:
        from env.game_state import (
            ACTION_MENU_VALUE,
            BATTLE_MON_SIZE,
            GBATTLE_ACTION_MENU_ADDR,
            GBATTLE_MONS_ADDR,
            GBATTLE_OUTCOME_ADDR,
            GBATTLE_TYPE_FLAGS_ADDR,
            GMOVE_RESULT_FLAGS_ADDR,
        )

        if addr == GBATTLE_ACTION_MENU_ADDR:
            return bytes([ACTION_MENU_VALUE if self._phase == "menu" else 0])
        if addr == GBATTLE_TYPE_FLAGS_ADDR:
            return _u16b(0 if self._outcome else 1) + b"\x00\x00"
        if addr == GBATTLE_OUTCOME_ADDR:
            return bytes([self._outcome])
        if addr == GMOVE_RESULT_FLAGS_ADDR:
            return _u16b(0)
        pbase = GBATTLE_MONS_ADDR
        obase = GBATTLE_MONS_ADDR + BATTLE_MON_SIZE
        for base, hp, mx in ((pbase, self._my_hp, 19), (obase, self._opp_hp, 18)):
            if base <= addr < base + BATTLE_MON_SIZE:
                buf = bytearray(BATTLE_MON_SIZE)
                buf[0x00:0x02] = _u16b(1)
                buf[0x0C:0x0E] = _u16b(1)
                buf[0x24] = 10
                buf[0x21], buf[0x22] = 12, 12
                buf[0x28:0x2A] = _u16b(hp)
                buf[0x2A] = 5
                buf[0x2C:0x2E] = _u16b(mx)
                off = addr - base
                return bytes(buf[off : off + size])
        raise AssertionError(f"unexpected read at 0x{addr:08X}")


def _farm_memory(*, with_healing_spot: bool = True) -> MapMemory:
    memory = MapMemory()
    snap = WorldSnapshot(FarmWorld.MAP, FarmWorld.CELL, None)
    memory.observe(snap, WorldEvent(encounter_started=True))   # grass here
    if with_healing_spot:
        memory.observe(snap, WorldEvent(healed=True))          # Center here
    return memory


_FIGHTER = {"move_type_fn": (lambda mid: 12), "predict": (lambda obs: 0)}


def test_level_up_reaches_target_after_several_battles() -> None:
    world = FarmWorld(start_level=7, target_hp_after=[(5, 5)])  # full: never heals
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(), WallMap(),
        target_level=10, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.battles_won == 3   # 7 -> 8 -> 9 -> 10


def test_level_up_already_at_target_fights_nothing() -> None:
    world = FarmWorld(start_level=10, target_hp_after=[(5, 5)])
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(), WallMap(),
        target_level=10, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.battles_won == 0
```

- [ ] **Step 2: Run the two tests to verify they fail**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -k level_up -v`
Expected: FAIL — `execute_order` has no `target_level` keyword (`TypeError: execute_order() got an unexpected keyword argument 'target_level'`).

- [ ] **Step 3: Implement the loop, helper, dispatch, and params**

In `env/orders.py`:

(a) Extend the heal_detector import (line 21) from:
```python
from env.heal_detector import party_is_full
```
to:
```python
from env.heal_detector import party_is_full, party_needs_heal
```

(b) Extend `execute_order`'s signature and add the dispatch branch. Change the signature (lines 51-60) to add three params, and add the branch after the `grind` branch (after line 78, before `dest = DESTINATIONS.get(...)`):
```python
def execute_order(
    order: Order,
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
    target_level: int = 0,
    heal_threshold: float = 0.4,
    max_cycles: int = 50,
) -> str:
```
and the new branch:
```python
    if order.mode == "level_up":
        return _execute_level_up(
            emulator, reader, memory, wallmap,
            target_level=target_level, heal_threshold=heal_threshold,
            max_cycles=max_cycles, max_hops=max_hops,
            move_type_fn=move_type_fn, predict=predict,
        )
```

(c) Add the helper and the loop at the end of the file (after `_walk_until_encounter`):
```python
def _reached(levels: list[int], target: int) -> bool:
    """True when the party's mean level is at or above the target."""
    return bool(levels) and sum(levels) / len(levels) >= target


def _execute_level_up(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    target_level: int,
    heal_threshold: float = 0.4,
    max_cycles: int = 50,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Grind wild battles until the party's mean level reaches target_level,
    healing whenever party_needs_heal fires. Composes _execute_grind (one battle)
    and _execute_heal (one heal); the loop is bounded by max_cycles.

    Returns "leveled_up" | "grind_exhausted" | a grind pass-through
    ("no_grass_spot_known" | "no_encounter"-driven continue | "lost" |
    "battle_timeout" | travel outcomes) | a heal pass-through
    ("no_healing_spot_known" | "heal_failed" | travel outcomes).

    NOTE: expects a Fighter (move_type_fn + predict). Without them the first
    _execute_grind returns "encounter_started", surfaced verbatim.
    """
    for _ in range(max_cycles):
        if _reached(reader.party_levels(), target_level):
            return "leveled_up"
        result = _execute_grind(
            emulator, reader, memory, wallmap,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
        if result == "won":
            if party_needs_heal(reader.party_hp(), heal_threshold):
                healed = _execute_heal(
                    emulator, reader, memory, wallmap, max_hops=max_hops
                )
                if healed != "healed":
                    return healed
        elif result == "no_encounter":
            continue   # no battle this cycle (RNG); retry, budget-bounded
        else:
            return result
    return "leveled_up" if _reached(reader.party_levels(), target_level) else "grind_exhausted"
```

- [ ] **Step 4: Run the two tests to verify they pass**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -k level_up -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "feat: level_up mode — loop grind to a target average level"
```

---

## Task 4: Auto-heal detour behavior

**Files:**
- Test: `tests/test_orders.py`

No production code changes — these exercise the heal detour and abort paths of `_execute_level_up` already written in Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orders.py`:
```python
def test_level_up_detours_to_heal_when_hp_is_low() -> None:
    # After each win the single member drops to 1/5 = 0.2 < 0.4 -> heal, then resume.
    world = FarmWorld(start_level=9, target_hp_after=[(1, 5)])
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(), WallMap(),
        target_level=10, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.battles_won == 1
    assert world.heals >= 1
    assert all(cur == mx for cur, mx in world.party_hp())   # healed to full


def test_level_up_heals_on_ko_even_when_totals_are_fine() -> None:
    # 5/10 total is above threshold, but a KO'd member forces the heal.
    world = FarmWorld(start_level=9, target_hp_after=[(0, 5), (5, 5)], party_size=2)
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(), WallMap(),
        target_level=10, heal_threshold=0.4, **_FIGHTER,
    )
    assert result == "leveled_up"
    assert world.heals >= 1


def test_level_up_aborts_when_heal_needed_but_no_spot_known() -> None:
    world = FarmWorld(start_level=1, target_hp_after=[(1, 5)])
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(with_healing_spot=False), WallMap(),
        target_level=10, **_FIGHTER,
    )
    assert result == "no_healing_spot_known"
```

- [ ] **Step 2: Run to verify they pass** (behavior already implemented in Task 3)

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -k level_up -v`
Expected: PASS (5 level_up tests total). If any fail, the loop's heal branch is wrong — fix `_execute_level_up`, not the tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_orders.py
git commit -m "test: level_up auto-heal detour (low HP, KO, no-spot abort)"
```

---

## Task 5: Budget and lost-battle termination

**Files:**
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orders.py`:
```python
def test_level_up_exhausts_budget_without_reaching_target() -> None:
    world = FarmWorld(start_level=5, target_hp_after=[(5, 5)])  # full: never heals
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(), WallMap(),
        target_level=20, max_cycles=2, **_FIGHTER,
    )
    assert result == "grind_exhausted"
    assert world.battles_won == 2   # bounded by max_cycles


def test_level_up_aborts_on_a_lost_battle() -> None:
    world = FarmWorld(start_level=5, target_hp_after=[(5, 5)], can_win=False)
    order = Order(destination="route_101", mode="level_up", combat="win")
    result = execute_order(
        order, world, world, _farm_memory(), WallMap(),
        target_level=10, **_FIGHTER,
    )
    assert result == "lost"
```

- [ ] **Step 2: Run to verify they pass** (behavior already implemented in Task 3)

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -k level_up -v`
Expected: PASS (7 level_up tests total).

- [ ] **Step 3: Commit**

```bash
git add tests/test_orders.py
git commit -m "test: level_up stops on budget exhaustion and lost battle"
```

---

## Task 6: Documentation

**Files:**
- Modify: `env/orders.py`

- [ ] **Step 1: Update the module docstring**

In `env/orders.py`, change the module docstring's mode summary (lines 3-7) from:
```
The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. "advance" navigates via travel_to;
"heal" travels to a known healing spot and presses A until the party is full;
"grind" travels to a known grass cell and treads until a wild battle starts.
The combat directive is stored for a future Fighter hookup.
```
to:
```
The Strategist (chef) emits an Order naming a destination + a mode + a combat
directive; the Explorer (worker) executes it. "advance" navigates via travel_to;
"heal" travels to a known healing spot and presses A until the party is full;
"grind" travels to a known grass cell and treads until a wild battle starts, then
the Fighter plays it; "level_up" loops grind to a target average level, healing
when the party gets low. The combat directive is stored for a future Fighter use.
```

- [ ] **Step 2: Update `execute_order`'s docstring**

In `execute_order`'s docstring, change the mode line (line 65) from:
```
    advance requires `destination` to be in DESTINATIONS.
```
to:
```
    advance requires `destination` to be in DESTINATIONS. level_up loops grind to
    the given target_level (mean of the party), healing when party_needs_heal.
```
and append to the Returns list (after line 70, inside the docstring):
```
    level_up adds: "leveled_up" | "grind_exhausted".
```

- [ ] **Step 3: Run the full suite + ruff**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/ tests/`
Expected: all pass, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add env/orders.py
git commit -m "docs: document level_up mode in orders.py"
```

---

## Self-Review Notes

- **Spec coverage:** `party_needs_heal` (Task 1) = spec Brick 2; `party_levels()` passthrough (Task 2) = Brick 1; `_execute_level_up` + `_reached` + dispatch + params (Task 3) = Brick 3 + Architecture + Parameters; heal detour / KO / no-spot (Task 4), budget / lost (Task 5) = spec's Outcomes + Testing list; docstrings (Task 6). The "already at target → leveled_up with zero battles" spec point is Task 3 step 1's second test. `no_encounter` continue is covered by the loop code (no dedicated test — it is a `continue`, not a terminal outcome, and forcing it in `FarmWorld` would require an un-triggering tread that then never progresses; the branch is trivial and the RNG path is documented).
- **Type consistency:** `_reached(levels, target)`, `party_needs_heal(hp, threshold)`, and `_execute_level_up(..., target_level, heal_threshold=0.4, max_cycles=50, max_hops=20, move_type_fn=None, predict=None)` are used identically everywhere. `FarmWorld` exposes `party_levels`/`party_hp`/`in_battle`/`snapshot`/`step`/`read_bytes` — every method the loop, `travel_to`, and `play_battle` call.
- **Fake fidelity:** `FarmWorld.read_bytes` mirrors the proven `GrassBattleWorld.read_bytes` byte-for-byte; the battle drives to `outcome=1` (won) or `outcome=2` (lost) exactly as `play_battle` expects.
