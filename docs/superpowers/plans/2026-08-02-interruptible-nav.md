# Interruptible Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `navigate_to` survive a wild battle that starts mid-move — detect it, hand it to the trained Fighter via `play_battle`, win, and resume walking — so `travel_to` and `execute_order`'s `advance` mode inherit grass-crossing for free.

**Architecture:** A single small helper `_handle_battle_interruption` at the top of `navigate_to`'s loop (after the has_grass/heal recording block, before the target/plan/probe logic) hands any in-progress battle to the injected Fighter. New terminal outcomes (`battle_lost`, `battle_timeout`, `battle_interrupted`) propagate verbatim up through `travel_to` (guarded by a `BATTLE_OUTCOMES` tuple) and `execute_order`'s `advance` path. The Fighter is dependency-injected (`move_type_fn`, `predict`) so `live_navigator`/`map_traveler` stay free of Stable-Baselines3/torch.

**Tech Stack:** Python 3.12, pytest (pure, no ROM, no SB3). Reuses `env/battle_player.play_battle`, `env/encounter_detector.EncounterWatcher`, `env/heal_detector.HealWatcher`.

**Spec:** `docs/superpowers/specs/2026-08-02-interruptible-nav-design.md`

---

## File Structure

- `env/live_navigator.py` — new import `from env.battle_player import play_battle`; new helper `_handle_battle_interruption`; `navigate_to` gains `move_type_fn`/`predict` and the interruption check in its loop.
- `env/map_traveler.py` — `travel_to` gains `move_type_fn`/`predict`, threads them to all three `navigate_to` calls, and returns any `BATTLE_OUTCOMES` result immediately.
- `env/orders.py` — `execute_order`'s `advance` path threads `move_type_fn`/`predict`; docstring returns list extended.
- `tests/test_live_navigator.py` — new battle-navigation fake + win-and-resume / lost / no-Fighter tests; two existing no-Fighter grass tests updated to the new `battle_interrupted` contract.
- `tests/test_map_traveler.py` — new battle fake + loss-propagation test.
- `tests/test_orders.py` — advance-through-grass fake + win / no-Fighter tests.

---

### Task 1: `_handle_battle_interruption` + `navigate_to` wiring

**Files:**
- Modify: `env/live_navigator.py`
- Test: `tests/test_live_navigator.py`

- [ ] **Step 1: Update the two existing no-Fighter grass tests to the new contract**

The new behavior: with no Fighter wired, an in-progress battle makes `navigate_to` return `"battle_interrupted"` (regardless of `memory`). The has_grass recording still happens first (recording runs before the interruption handler). Replace the two existing tests at the bottom of `tests/test_live_navigator.py` (`test_learns_grass_cell_on_the_in_battle_edge` and `test_navigate_without_memory_ignores_battles`) with:

```python
def test_no_fighter_learns_grass_then_returns_battle_interrupted() -> None:
    # Walking (0,0)->(2,0); a battle fires on (1,0). With no Fighter wired the
    # cell is still tagged has_grass (recording runs first), then navigate_to
    # aborts with battle_interrupted.
    world = EncounterFakeWorld(grass_at=(1, 0), start=(0, 0))
    memory = MapMemory()
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50, memory=memory)
    assert result == "battle_interrupted"
    assert memory.cells_labeled("has_grass") == [((0, 0), (1, 0))]


def test_no_fighter_returns_battle_interrupted_even_without_memory() -> None:
    # memory=None: the battle is still detected and aborts with battle_interrupted.
    world = EncounterFakeWorld(grass_at=(1, 0), start=(0, 0))
    result = navigate_to(world, world, WallMap(), target=(2, 0), max_steps=50)
    assert result == "battle_interrupted"
```

- [ ] **Step 2: Add the battle-navigation fake and the win-and-resume + lost tests**

Append to `tests/test_live_navigator.py` (the `buttons` import at the top is already present):

```python
def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class BattleNavWorld(FakeWorld):
    """FakeWorld that starts a wild battle on grass_at. The injected Fighter
    plays it out via play_battle; on a win the battle clears (in_battle drops to
    False) so walking resumes. can_win=False makes the Fighter lose.

    Serves the battle-reader bytes exactly like GrassBattleWorld in test_orders.
    """

    _RESOLVE_PRESSES = 2

    def __init__(self, grass_at: tuple[int, int], can_win: bool = True,
                 **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._grass_at = grass_at
        self._can_win = can_win
        self._battle = False
        self._fought = False
        self._opp_hp = 18
        self._my_hp = 19
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        if self._battle:
            self._battle_step(keys)
            return
        super().step(keys, frames)
        if self.pos == self._grass_at and not self._fought:
            self._battle = True
            self._fought = True

    def _battle_step(self, keys: int) -> None:
        if keys == 0:
            return
        if self._phase == "menu" and keys & buttons.KEY_A:
            self._phase = "moves"
        elif self._phase == "moves" and keys & buttons.KEY_A:
            if not self._can_win:
                self._outcome = 2   # terminal loss -> play_battle returns "lost"
                self._battle = False
                return
            self._opp_hp = max(0, self._opp_hp - 6)
            if self._opp_hp == 0:
                self._outcome = 1
                self._battle = False   # won: resume walking
            self._phase = "resolving"
            self._resolve_left = self._RESOLVE_PRESSES
        elif self._phase == "resolving" and keys & buttons.KEY_A:
            self._resolve_left -= 1
            if self._resolve_left <= 0 and self._outcome == 0:
                self._phase = "menu"

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


def test_fighter_wins_the_interruption_and_navigation_resumes() -> None:
    # Battle fires on grass cell (1,0); the Fighter wins, walking resumes to (2,0).
    # The false-wall bug must NOT trigger: no wall is recorded and it arrives.
    world = BattleNavWorld(grass_at=(1, 0), start=(0, 0))
    memory = MapMemory()
    wallmap = WallMap()
    result = navigate_to(
        world, world, wallmap, target=(2, 0), max_steps=50, memory=memory,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "arrived"
    assert world.pos == (2, 0)
    assert memory.cells_labeled("has_grass") == [((0, 0), (1, 0))]
    assert not wallmap.is_blocked((0, 0), (1, 0), "right")


def test_fighter_loss_aborts_navigation() -> None:
    world = BattleNavWorld(grass_at=(1, 0), start=(0, 0), can_win=False)
    result = navigate_to(
        world, world, WallMap(), target=(2, 0), max_steps=50,
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "battle_lost"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: FAIL — `navigate_to()` has no `move_type_fn`/`predict` kwargs (TypeError) and returns `"arrived"` where `"battle_interrupted"`/`"battle_lost"` are now expected.

(If `.venv` is missing in the worktree, use the main repo interpreter: `/Users/_eloi/Projets/Emu/.venv/bin/python`.)

- [ ] **Step 4: Add the helper and wire `navigate_to` in `env/live_navigator.py`**

Add the import near the other `env` imports (line 12-16 block):

```python
from env.battle_player import play_battle
```

Add the helper immediately above `def navigate_to` (after the constants block):

```python
def _handle_battle_interruption(
    emulator: Any, reader: Any, move_type_fn: Any, predict: Any
) -> str | None:
    """If a wild battle is in progress, hand it to the Fighter and report.

    Returns None when there is no battle (or the battle was won) so the caller
    resumes navigating; returns a terminal outcome when navigation must abort:
    "battle_interrupted" (no Fighter supplied), "battle_lost", "battle_timeout".
    """
    if not reader.in_battle():
        return None
    if move_type_fn is None or predict is None:
        return "battle_interrupted"
    result = play_battle(emulator, move_type_fn, predict)
    if result == "won":
        return None
    return "battle_lost" if result == "lost" else "battle_timeout"
```

Change the `navigate_to` signature to add the two optional Fighter deps (keep `memory` last-but-two so existing keyword calls stay valid):

```python
def navigate_to(
    emulator: Any,
    reader: Any,
    wallmap: WallMap,
    target: tuple[int, int],
    max_steps: int = 200,
    memory: MapMemory | None = None,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
```

Extend the docstring's returns line:

```python
    Returns 'arrived' | 'unreachable' | 'left_map' | 'timeout' |
    'battle_lost' | 'battle_timeout' | 'battle_interrupted'.
```

Insert the interruption check in the loop AFTER the `if memory is not None:` recording block and BEFORE `if before.pos == target:`:

```python
        if memory is not None:
            if heal_watcher.observe(reader.party_hp()):
                memory.observe(before, WorldEvent(healed=True))
            if enc_watcher.observe(reader.in_battle()):
                memory.observe(before, WorldEvent(encounter_started=True))
        interruption = _handle_battle_interruption(
            emulator, reader, move_type_fn, predict
        )
        if interruption is not None:
            return interruption
        if before.pos == target:
            return "arrived"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest tests/test_live_navigator.py -q`
Expected: PASS (all live_navigator tests green).

- [ ] **Step 6: Lint**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/ruff check env/live_navigator.py tests/test_live_navigator.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p4-interruptible-nav
git add env/live_navigator.py tests/test_live_navigator.py
git commit -m "$(cat <<'EOF'
feat: navigate_to hands a mid-move wild battle to the Fighter and resumes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `travel_to` threads the Fighter + propagates battle outcomes

**Files:**
- Modify: `env/map_traveler.py`
- Test: `tests/test_map_traveler.py`

- [ ] **Step 1: Write the failing propagation test**

Append to `tests/test_map_traveler.py`:

```python
def _u16b(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


class LostBattleWorld:
    """Same-map world where walking treads grass at grass_at; the Fighter loses.

    Acts as emulator (step + read_bytes) and reader (snapshot). Movement is free
    until the battle starts; the battle is a terminal loss (outcome 2).
    """

    def __init__(self, map_id: tuple[int, int], start: tuple[int, int],
                 grass_at: tuple[int, int]) -> None:
        self.map_id = map_id
        self.pos = start
        self._grass_at = grass_at
        self._battle = False
        self._fought = False
        self._outcome = 0
        self._phase = "menu"

    def step(self, keys: int, frames: int) -> None:
        if self._battle:
            if keys == 0:
                return
            if self._phase == "menu" and keys & buttons.KEY_A:
                self._phase = "moves"
            elif self._phase == "moves" and keys & buttons.KEY_A:
                self._outcome = 2   # terminal loss
                self._battle = False
            return
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.pos == self._grass_at and not self._fought:
            self._battle = True
            self._fought = True

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

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
        for base, hp, mx in ((pbase, 19, 19), (obase, 18, 18)):
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


def test_battle_loss_propagates_from_the_first_hop() -> None:
    # Same-map goal: travel_to delegates to navigate_to, which treads grass at
    # (1,0), loses the battle, and the loss propagates as battle_lost (not "lost").
    world = LostBattleWorld(map_id=(0, 0), start=(0, 0), grass_at=(1, 0))
    result = travel_to(
        world, world, MapMemory(), WallMap(),
        goal_map=(0, 0), goal_cell=(2, 0),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "battle_lost"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest tests/test_map_traveler.py::test_battle_loss_propagates_from_the_first_hop -q`
Expected: FAIL — `travel_to()` has no `move_type_fn`/`predict` kwargs (TypeError).

- [ ] **Step 3: Thread the deps and add propagation in `env/map_traveler.py`**

Add the module constant below `SETTLE_TRIES = 4`:

```python
BATTLE_OUTCOMES = ("battle_lost", "battle_timeout", "battle_interrupted")
```

Extend the `travel_to` signature:

```python
def travel_to(
    emulator: Any,
    reader: Any,
    memory: MapMemory,
    wallmap: WallMap,
    goal_map: tuple[int, int],
    goal_cell: tuple[int, int],
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
```

Extend the docstring returns line:

```python
    Returns 'arrived' | 'unknown_route' | 'unreachable' | 'lost' | 'timeout' |
    'battle_lost' | 'battle_timeout' | 'battle_interrupted'.
```

Pass the deps to all three `navigate_to` calls and add battle-outcome guards. The goal-cell call (delegates directly, so battle outcomes propagate through its return):

```python
        if here.map_id == goal_map:
            return navigate_to(
                emulator, reader, wallmap, goal_cell,
                move_type_fn=move_type_fn, predict=predict,
            )
```

The portal-approach call:

```python
        reached = navigate_to(
            emulator, reader, wallmap, crossing.from_cell,
            move_type_fn=move_type_fn, predict=predict,
        )
        if reached in BATTLE_OUTCOMES:
            return reached
        if reached in ("unreachable", "timeout"):
            return reached
        if reached == "left_map":
            continue   # already crossed a border on the way; re-plan from new map
```

The door-crossing call:

```python
        crossed = navigate_to(
            emulator, reader, wallmap, neighbour, memory=memory,
            move_type_fn=move_type_fn, predict=predict,
        )
        if crossed in BATTLE_OUTCOMES:
            return crossed
        if crossed in ("unreachable", "timeout"):
            return crossed   # the crossing never fired; not a route divergence
```

- [ ] **Step 4: Run to verify it passes (and no regressions)**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest tests/test_map_traveler.py -q`
Expected: PASS (all traveler tests green).

- [ ] **Step 5: Lint**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/ruff check env/map_traveler.py tests/test_map_traveler.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p4-interruptible-nav
git add env/map_traveler.py tests/test_map_traveler.py
git commit -m "$(cat <<'EOF'
feat: travel_to threads the Fighter and propagates battle outcomes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `execute_order` advance path threads the Fighter

**Files:**
- Modify: `env/orders.py`
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing advance tests**

Append to `tests/test_orders.py` (it already has `_KEY_TO_DIR`, `_DELTAS`, `_u16b`, `Order`, `execute_order`, `MapMemory`, `WallMap`, `WorldSnapshot` in scope):

```python
class AdvanceBattleWorld:
    """Walks toward the route_101 destination cell, treading grass on the way.

    Acts as emulator (step + read_bytes) and reader (snapshot). The Fighter (if
    wired) wins the battle and walking resumes to the goal cell.
    """

    _RESOLVE_PRESSES = 2

    def __init__(self, map_id: tuple[int, int], start: tuple[int, int],
                 grass_at: tuple[int, int]) -> None:
        self.map_id = map_id
        self.pos = start
        self._grass_at = grass_at
        self._battle = False
        self._fought = False
        self._opp_hp = 18
        self._outcome = 0
        self._phase = "menu"
        self._resolve_left = 0

    def step(self, keys: int, frames: int) -> None:
        from emulator import buttons

        if self._battle:
            if keys == 0:
                return
            if self._phase == "menu" and keys & buttons.KEY_A:
                self._phase = "moves"
            elif self._phase == "moves" and keys & buttons.KEY_A:
                self._opp_hp = max(0, self._opp_hp - 6)
                if self._opp_hp == 0:
                    self._outcome = 1
                    self._battle = False
                self._phase = "resolving"
                self._resolve_left = self._RESOLVE_PRESSES
            elif self._phase == "resolving" and keys & buttons.KEY_A:
                self._resolve_left -= 1
                if self._resolve_left <= 0 and self._outcome == 0:
                    self._phase = "menu"
            return
        direction = _KEY_TO_DIR.get(keys)
        if direction is None:
            return
        dx, dy = _DELTAS[direction]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.pos == self._grass_at and not self._fought:
            self._battle = True
            self._fought = True

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

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
        for base, hp, mx in ((pbase, 19, 19), (obase, self._opp_hp, 18)):
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


def test_advance_wins_a_grass_battle_and_arrives() -> None:
    # route_101 destination is ((0,16),(5,12)); start one grass cell short.
    world = AdvanceBattleWorld(map_id=(0, 16), start=(3, 12), grass_at=(4, 12))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), WallMap(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "arrived"
    assert world.pos == (5, 12)


def test_advance_without_fighter_reports_battle_interrupted() -> None:
    world = AdvanceBattleWorld(map_id=(0, 16), start=(3, 12), grass_at=(4, 12))
    order = Order(destination="route_101", mode="advance", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "battle_interrupted"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest tests/test_orders.py -k advance_wins_a_grass_battle or advance_without_fighter -q`
Expected: FAIL — `advance` does not pass the Fighter deps to `travel_to`, so `test_advance_wins_a_grass_battle_and_arrives` returns `"battle_interrupted"` instead of `"arrived"`.

- [ ] **Step 3: Thread the deps in `env/orders.py`**

Change the `advance` path (the final `travel_to` call in `execute_order`) to pass the Fighter deps:

```python
    return travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell,
        max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
    )
```

Extend the `execute_order` docstring returns list to include the advance battle outcomes. Change the travel_to pass-through line:

```python
    one of travel_to's outcomes ("arrived" | "unknown_route" | "unreachable" |
    "lost" | "timeout" | "battle_lost" | "battle_timeout" | "battle_interrupted") |
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest tests/test_orders.py -q`
Expected: PASS (all orders tests green).

- [ ] **Step 5: Full suite + lint**

Run: `cd /Users/_eloi/Projets/Emu-p4-interruptible-nav && .venv/bin/python -m pytest -q && .venv/bin/ruff check env tests`
Expected: whole suite PASS (ROM smokes skip without `POKEMON_EMERALD_ROM`), ruff clean.

- [ ] **Step 6: Commit**

```bash
cd /Users/_eloi/Projets/Emu-p4-interruptible-nav
git add env/orders.py tests/test_orders.py
git commit -m "$(cat <<'EOF'
feat: execute_order advance threads the Fighter through interruptible travel

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Helper `_handle_battle_interruption` with the exact four-case contract (not-in-battle→None; no-Fighter→battle_interrupted; won→None; lost→battle_lost; else→battle_timeout) — Task 1 Step 4. ✓
- Loop ordering (recording block BEFORE interruption handler, so has_grass is learned even when the battle is then fought) — Task 1 Step 4 insert point + asserted by `test_fighter_wins_the_interruption_and_navigation_resumes` (has_grass learned) and `test_no_fighter_learns_grass_then_returns_battle_interrupted`. ✓
- New `navigate_to` outcomes — Task 1. ✓
- `travel_to` gains deps, threads to all three calls, `BATTLE_OUTCOMES` propagation, `battle_lost` distinct from `lost` — Task 2. ✓
- `execute_order` advance threads deps + docstring — Task 3. ✓
- Scope non-goals respected: `map_map` NOT touched; no new Fighter capability; no ROM smoke. ✓
- Testing: win-and-resume (arrives, no false wall, grass learned), lost→battle_lost, no-Fighter→battle_interrupted, regression (existing always-False fakes unchanged), traveler loss propagation, advance win + advance no-Fighter. ✓

**Placeholder scan:** none — every step has complete code and exact commands.

**Type/name consistency:** `move_type_fn`/`predict` kwargs match across `navigate_to`, `travel_to`, `execute_order`, `play_battle`. `BATTLE_OUTCOMES` used consistently. `cells_labeled("has_grass")`, `WallMap.is_blocked`, `MapMemory` APIs match existing usage. Fakes serve the same battle-reader addresses as `GrassBattleWorld`.

**Note on interpreter path:** commands assume a `.venv` in the worktree. If absent, substitute `/Users/_eloi/Projets/Emu/.venv/bin/python` and `/Users/_eloi/Projets/Emu/.venv/bin/ruff` (the worktree shares the main repo's tooling).
