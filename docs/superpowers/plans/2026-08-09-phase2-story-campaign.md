# Phase 2 Story Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable, tested campaign machinery so the scripted Strategist can drive Phase 2 from post-rival — return to Littleroot, enter Birch's lab, receive the Pokédex + 5 Poké Balls, and receive the running shoes on Route 101.

**Architecture:** A new `story` Order mode (sibling of `heal`): `travel_to(cell)` then a bounded A-spam loop until an injected target predicate holds. `Order` stays pure data; the predicate threads through `execute_order` as a `story_target` kwarg (exactly like `target_level` threads for `level_up`). Three new `EmeraldReader` detectors (`has_pokedex`, `has_running_shoes`, `has_item` with the Emerald securityKey XOR decrypt) supply the predicates. Because a fresh `post_rival.state` load carries an EMPTY `MapMemory` (it is a live Python object, not serialized), a hand-seeded `seed_return_portals(memory)` registers the 4 southbound return edges before `run_campaign` runs. A `PHASE2_CAMPAIGN` curriculum lives on the scripted campaign path; RL `milestones.py` is untouched.

**Tech Stack:** Python 3.12, pytest, ruff (line-length 100), stable-baselines3 (ROM smoke only), mGBA via the project emulator.

**Anti-circularity guard:** pure tests validate detector arithmetic against the detector's own candidate constants (true by construction). Only the throwaway live probe (Task 6) and the gated ROM smoke (Task 7) are load-bearing. This makes "build machinery with candidate constants first, probe confirms/replaces later" safe.

---

## File Structure

- `env/game_state.py` (modify) — add `SAVE_BLOCK2_PTR`, `_save_block2()`, and detectors `has_pokedex` / `has_running_shoes` / `has_item`, plus their constants. Responsibility: typed RAM reads.
- `env/orders.py` (modify) — add `story` mode: `STORY_*` constants, `_execute_story`, `_advance_story_dialogue`, two `DESTINATIONS` entries, and a `story_target` kwarg on `execute_order`. Responsibility: the Order language + execution.
- `env/campaign.py` (modify) — add `Milestone.story_target`, map-id constants, `seed_return_portals`, `PHASE2_CAMPAIGN`, and a story dispatch branch in `run_campaign`. Responsibility: scripted curriculum sequencing.
- `tests/test_game_state.py` (modify) — detector unit tests with crafted bytes.
- `tests/test_orders.py` (modify) — `story` mode unit tests with a `StoryWorld` fake.
- `tests/test_campaign.py` (modify) — `seed_return_portals` + story-dispatch unit tests.
- `tools/probe_phase2_facts.py` (create, throwaway) — freeze the live Phase 2 facts (flag ids, item offsets, trigger cells, A-press counts).
- `tests/test_phase2_rom.py` (create) — gated ROM smoke: seed portals then `run_campaign(PHASE2_CAMPAIGN)`; assert `has_pokedex()` AND `has_running_shoes()`.

Candidate constants (probe confirms/replaces in Task 6):
- `SAVE_BLOCK2_PTR = 0x03005D90`
- `FLAG_SYS_POKEDEX_GET = 0x801`
- `FLAG_RECEIVED_RUNNING_SHOES = 0x86F`
- `POKE_BALL_ITEM_ID = 0x4`
- `_ITEMS_POCKET_OFFSET = 0x560` (offsetof(SaveBlock1, bagPocket_Items) — candidate)
- `_SECURITY_KEY_OFFSET = 0xAC` (offsetof(SaveBlock2, encryptionKey))
- `_ITEM_SLOT_SIZE = 4` (u16 itemId + u16 quantity)

---

## Task 1: Pokédex + running-shoes flag detectors

**Files:**
- Modify: `env/game_state.py` (after `read_flag`, line ~99)
- Test: `tests/test_game_state.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_game_state.py`:

```python
from env.game_state import (
    EmeraldReader,
    SAVE_BLOCK1_PTR,
    _FLAGS_OFFSET,
    FLAG_SYS_POKEDEX_GET,
    FLAG_RECEIVED_RUNNING_SHOES,
)


def _reader_with_flag(flag_id: int, value: bool) -> EmeraldReader:
    """A reader whose SaveBlock1 has exactly `flag_id` set to `value`."""
    sb1 = 0x02025734
    byte_index, bit_index = divmod(flag_id, 8)
    flag_addr = sb1 + _FLAGS_OFFSET + byte_index
    flag_byte = (1 << bit_index) if value else 0

    def read(addr: int, size: int) -> bytes:
        if addr == SAVE_BLOCK1_PTR and size == 4:
            return sb1.to_bytes(4, "little")
        if addr == flag_addr and size == 1:
            return bytes([flag_byte])
        return bytes(size)

    return EmeraldReader(read)


def test_has_pokedex_true_when_flag_set() -> None:
    assert _reader_with_flag(FLAG_SYS_POKEDEX_GET, True).has_pokedex() is True


def test_has_pokedex_false_when_flag_clear() -> None:
    assert _reader_with_flag(FLAG_SYS_POKEDEX_GET, False).has_pokedex() is False


def test_has_running_shoes_true_when_flag_set() -> None:
    assert _reader_with_flag(FLAG_RECEIVED_RUNNING_SHOES, True).has_running_shoes() is True


def test_has_running_shoes_false_when_flag_clear() -> None:
    assert _reader_with_flag(FLAG_RECEIVED_RUNNING_SHOES, False).has_running_shoes() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_game_state.py -k "pokedex or running_shoes" -v`
Expected: FAIL with `ImportError: cannot import name 'FLAG_SYS_POKEDEX_GET'`.

- [ ] **Step 3: Write minimal implementation**

In `env/game_state.py`, after the `FLAG_SET_WALL_CLOCK = 0x51` block (line ~43), add:

```python
# FLAG_SYS_POKEDEX_GET / FLAG_RECEIVED_RUNNING_SHOES from pret/pokeemerald
# include/constants/flags.h. Candidate ids — the probe (tools/probe_phase2_facts.py)
# confirms or replaces them on BPEF before the ROM smoke becomes load-bearing.
FLAG_SYS_POKEDEX_GET = 0x801
FLAG_RECEIVED_RUNNING_SHOES = 0x86F
```

In `class EmeraldReader`, after `read_flag` (line ~98), add:

```python
    def has_pokedex(self) -> bool:
        """True once the Pokédex has been received (Birch lab cutscene)."""
        return self.read_flag(FLAG_SYS_POKEDEX_GET)

    def has_running_shoes(self) -> bool:
        """True once the running shoes have been received (Route 101 cutscene)."""
        return self.read_flag(FLAG_RECEIVED_RUNNING_SHOES)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_game_state.py -k "pokedex or running_shoes" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add env/game_state.py tests/test_game_state.py
git commit -m "$(cat <<'EOF'
feat: has_pokedex / has_running_shoes flag detectors on EmeraldReader

Candidate flag ids (FLAG_SYS_POKEDEX_GET, FLAG_RECEIVED_RUNNING_SHOES) reuse
the existing read_flag path; the Phase 2 probe confirms them on BPEF before the
ROM smoke goes load-bearing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `has_item` — securityKey XOR quantity decrypt

**Files:**
- Modify: `env/game_state.py` (add `SAVE_BLOCK2_PTR`, `_save_block2`, item constants, `has_item`)
- Test: `tests/test_game_state.py`

Emerald stores bag item quantities XOR-encrypted with the low 16 bits of `gSaveBlock2Ptr->encryptionKey`: `real_qty = stored_qty XOR (securityKey & 0xFFFF)`. An `ItemSlot` is `{u16 itemId; u16 quantity}` = 4 bytes; the Items pocket is a contiguous array in SaveBlock1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_game_state.py`:

```python
from env.game_state import (
    SAVE_BLOCK2_PTR,
    _ITEMS_POCKET_OFFSET,
    _SECURITY_KEY_OFFSET,
    _ITEM_SLOT_SIZE,
    POKE_BALL_ITEM_ID,
)


def _reader_with_item(item_id: int, real_qty: int, security_key: int) -> EmeraldReader:
    """A reader whose Items pocket slot 0 holds (item_id, real_qty) encrypted."""
    sb1 = 0x02025734
    sb2 = 0x02027000
    stored_qty = real_qty ^ (security_key & 0xFFFF)
    slot_addr = sb1 + _ITEMS_POCKET_OFFSET
    key_addr = sb2 + _SECURITY_KEY_OFFSET

    def read(addr: int, size: int) -> bytes:
        if addr == SAVE_BLOCK1_PTR and size == 4:
            return sb1.to_bytes(4, "little")
        if addr == SAVE_BLOCK2_PTR and size == 4:
            return sb2.to_bytes(4, "little")
        if addr == key_addr and size == 4:
            return security_key.to_bytes(4, "little")
        if addr == slot_addr and size == _ITEM_SLOT_SIZE:
            return item_id.to_bytes(2, "little") + stored_qty.to_bytes(2, "little")
        return bytes(size)

    return EmeraldReader(read)


def test_has_item_decrypts_quantity_and_meets_threshold() -> None:
    reader = _reader_with_item(POKE_BALL_ITEM_ID, real_qty=5, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=5) is True


def test_has_item_below_threshold_is_false() -> None:
    reader = _reader_with_item(POKE_BALL_ITEM_ID, real_qty=3, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=5) is False


def test_has_item_absent_is_false() -> None:
    reader = _reader_with_item(item_id=0x0, real_qty=0, security_key=0x1234ABCD)
    assert reader.has_item(POKE_BALL_ITEM_ID, min_qty=1) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_game_state.py -k "has_item" -v`
Expected: FAIL with `ImportError: cannot import name 'SAVE_BLOCK2_PTR'`.

- [ ] **Step 3: Write minimal implementation**

In `env/game_state.py`, after `SAVE_BLOCK1_PTR = 0x03005D8C` (line ~22), add:

```python
# IWRAM pointer to the relocated SaveBlock2 struct (holds the item encryption key).
# Candidate — probe-confirmed on BPEF before the ROM smoke goes load-bearing.
SAVE_BLOCK2_PTR = 0x03005D90
```

After `_EWRAM_END = 0x02040000` (line ~54), add:

```python
# Bag Items pocket: contiguous ItemSlot array inside SaveBlock1. Item quantities
# are XOR-encrypted with the low 16 bits of SaveBlock2.encryptionKey. Offsets are
# candidates from pret/pokeemerald, confirmed by the Phase 2 probe.
POKE_BALL_ITEM_ID = 0x4
_ITEMS_POCKET_OFFSET = 0x560   # offsetof(SaveBlock1, bagPocket_Items) — candidate
_SECURITY_KEY_OFFSET = 0xAC    # offsetof(SaveBlock2, encryptionKey)
_ITEM_SLOT_SIZE = 4            # u16 itemId + u16 quantity
_ITEMS_POCKET_CAPACITY = 30    # bound the scan (code-safety #2)
```

In `class EmeraldReader`, after `party_hp` (line ~117), add:

```python
    def has_item(self, item_id: int, min_qty: int = 1) -> bool:
        """True if the Items pocket holds >= min_qty of item_id (qty XOR-decrypted)."""
        sb1 = self._save_block1()
        sb2 = self._save_block2()
        if sb1 is None or sb2 is None:
            return False
        key = int.from_bytes(self._read(sb2 + _SECURITY_KEY_OFFSET, 4), "little") & 0xFFFF
        base = sb1 + _ITEMS_POCKET_OFFSET
        for slot in range(_ITEMS_POCKET_CAPACITY):
            entry = self._read(base + slot * _ITEM_SLOT_SIZE, _ITEM_SLOT_SIZE)
            slot_id = int.from_bytes(entry[0:2], "little")
            if slot_id != item_id:
                continue
            qty = int.from_bytes(entry[2:4], "little") ^ key
            return qty >= min_qty
        return False
```

After `_save_block1` (line ~123), add:

```python
    def _save_block2(self) -> int | None:
        sb2 = int.from_bytes(self._read(SAVE_BLOCK2_PTR, 4), "little")
        if not _EWRAM_START <= sb2 < _EWRAM_END:
            return None
        return sb2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_game_state.py -k "has_item" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add env/game_state.py tests/test_game_state.py
git commit -m "$(cat <<'EOF'
feat: has_item with SaveBlock2 securityKey XOR quantity decrypt

Reads the Items pocket ItemSlot array and XOR-decrypts each quantity with the
low 16 bits of SaveBlock2.encryptionKey. Candidate offsets confirmed by the
Phase 2 probe before the ROM smoke goes load-bearing.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `seed_return_portals` — hand-seed the southbound return graph

**Files:**
- Modify: `env/campaign.py` (add map-id constants + `seed_return_portals`)
- Test: `tests/test_campaign.py`

A fresh `post_rival.state` load carries an EMPTY `MapMemory` (live object, never serialized). The rival leg used single-map A*, not `travel_to`, so no inherited memory exists. Hand-seed the 4 return edges so `plan_route` finds route_103 → Oldale → route_101 → Littleroot → lab.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_campaign.py`:

```python
from env.campaign import (
    seed_return_portals,
    ROUTE_103,
    OLDALE,
    ROUTE_101,
    LITTLEROOT,
    LAB,
)
from env.map_memory import MapMemory
from env.route_planner import plan_route


def test_seed_return_portals_links_route_103_to_lab() -> None:
    memory = MapMemory()
    seed_return_portals(memory)
    assert plan_route(memory, ROUTE_103, LAB) == [
        ROUTE_103, OLDALE, ROUTE_101, LITTLEROOT, LAB,
    ]


def test_seed_return_portals_registers_each_southbound_crossing() -> None:
    memory = MapMemory()
    seed_return_portals(memory)
    assert memory.portal(ROUTE_103, OLDALE) is not None
    assert memory.portal(OLDALE, ROUTE_101) is not None
    assert memory.portal(ROUTE_101, LITTLEROOT) is not None
    assert memory.portal(LITTLEROOT, LAB) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_campaign.py -k "seed_return_portals" -v`
Expected: FAIL with `ImportError: cannot import name 'seed_return_portals'`.

- [ ] **Step 3: Write minimal implementation**

In `env/campaign.py`, after the imports (line ~17), add:

```python
from env.map_memory import MapMemory

# Map ids on the return path (probe-confirmed live cells replace the candidates
# in seed_return_portals; the map ids themselves are already verified).
ROUTE_103 = (0, 18)
OLDALE = (0, 10)
ROUTE_101 = (0, 16)
LITTLEROOT = (0, 9)
LAB = (1, 4)

# Southbound return crossings, hand-seeded because a fresh savestate load carries
# an empty MapMemory. from_cell/to_cell are candidates the Phase 2 probe pins
# exactly; direction and reversibility are the real overworld/warp semantics.
_RETURN_PORTALS = (
    (ROUTE_103, (0, 18), "down", OLDALE, True, (0, 0)),
    (OLDALE, (0, 9), "down", ROUTE_101, True, (0, 0)),
    (ROUTE_101, (0, 19), "down", LITTLEROOT, True, (10, 1)),
    (LITTLEROOT, (3, 10), "up", LAB, False, (6, 12)),
)


def seed_return_portals(memory: MapMemory) -> None:
    """Register the 4 southbound return edges so travel_to can path home.

    A fresh post_rival.state load has an empty MapMemory (not serialized), so
    the first story milestone's travel_to would return 'unknown_route' on step
    zero. This hand-seeds route_103 -> Oldale -> route_101 -> Littleroot -> lab.
    """
    for from_map, from_cell, direction, to_map, reversible, to_cell in _RETURN_PORTALS:
        memory.record_portal(from_map, from_cell, direction, to_map, reversible, to_cell)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_campaign.py -k "seed_return_portals" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "$(cat <<'EOF'
feat: seed_return_portals hand-seeds the southbound return graph

A fresh post_rival.state load carries an empty MapMemory, so the first story
milestone's travel_to would return unknown_route on step zero. Seed the 4 return
edges (route_103->Oldale->route_101->Littleroot->lab); the probe pins exact cells.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `story` Order mode — travel then A-spam to a target predicate

**Files:**
- Modify: `env/orders.py` (STORY_* consts, DESTINATIONS entries, `_execute_story`, `_advance_story_dialogue`, `story_target` kwarg + dispatch)
- Test: `tests/test_orders.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orders.py`:

```python
# ---------------------------------------------------------------------------
# Story tests
# ---------------------------------------------------------------------------


class StoryWorld:
    """Fake emulator+reader on a single map: a scripted cutscene predicate flips
    true after N A-presses (models a lab/shoes A-spam until control/flag)."""

    def __init__(
        self,
        map_id: tuple[int, int],
        cell: tuple[int, int],
        a_presses_to_done: int = 2,
    ) -> None:
        self.map_id = map_id
        self.pos = cell
        self._to_done = a_presses_to_done
        self._a_count = 0

    def step(self, keys: int, frames: int) -> None:
        if keys & buttons.KEY_A:
            self._a_count += 1

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(map_id=self.map_id, pos=self.pos, tile_behavior=None)

    def party_hp(self) -> list[tuple[int, int]]:
        return [(5, 5)]

    def in_battle(self) -> bool:
        return False

    def battle_starting(self) -> bool:
        return False

    def event_done(self) -> bool:
        return self._a_count >= self._to_done

    @property
    def grid_reader(self) -> _AllFreeGridReader:
        return _AllFreeGridReader(50, 50)


def test_story_unknown_destination_returns_unknown_destination() -> None:
    world = StoryWorld((1, 4), (6, 12))
    order = Order(destination="atlantide", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "unknown_destination"


def test_story_arrives_then_a_spams_until_target_holds() -> None:
    world = StoryWorld((1, 4), (6, 12), a_presses_to_done=2)
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "story_done"


def test_story_target_already_true_returns_immediately() -> None:
    world = StoryWorld((1, 4), (6, 12), a_presses_to_done=0)
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "story_done"


def test_story_that_never_completes_returns_story_timeout() -> None:
    world = StoryWorld((1, 4), (6, 12), a_presses_to_done=10_000)
    order = Order(destination="lab", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "story_timeout"


def test_story_passes_through_travel_failure() -> None:
    # route_101_shoes is on a different map with no seeded route -> unknown_route,
    # surfaced verbatim before any A-spam.
    world = StoryWorld((0, 9), (3, 10))
    order = Order(destination="route_101_shoes", mode="story", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), story_target=lambda r: r.event_done()
    )
    assert result == "unknown_route"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_orders.py -k "story" -v`
Expected: FAIL — `execute_order` raises `TypeError: unexpected keyword argument 'story_target'` (or story mode falls through to advance and returns the wrong value).

- [ ] **Step 3: Write minimal implementation**

In `env/orders.py`, add the destinations (in `DESTINATIONS`, after the `route_103` entry, line ~41):

```python
    "lab": ((1, 4), (6, 12)),            # Birch lab warp-landing tile (probe-pinned)
    "route_101_shoes": ((0, 16), (5, 12)),  # Route 101 running-shoes trigger cell
```

After the `GRIND_*` constants (line ~56), add:

```python
STORY_PRESS_A_FRAMES = 6
STORY_RELEASE_FRAMES = 10
STORY_MAX_PRESSES = 2000   # bound the cutscene A-spam (code-safety #2)
```

In `execute_order`, add the `story_target` parameter (after `max_cycles: int = 50,`, line ~69):

```python
    story_target: Any = None,
```

Add the dispatch branch (before the `dest = DESTINATIONS.get(order.destination)` fallthrough, line ~106):

```python
    if order.mode == "story":
        return _execute_story(
            emulator, reader, memory, order.destination, story_target,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
```

After `_execute_battle_trainer` / `_walk_until_trainer` (end of file), add:

```python
def _execute_story(
    emulator: Any,
    reader: Any,
    memory: Any,
    destination: str,
    story_target: Any,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Travel to a story cell, then A-spam until story_target(reader) holds.

    Sibling of heal: heal presses A until the party is full, story presses A
    until an injected predicate (a flag set / an item held / control regained)
    holds. The lab cutscene locks control on its warp-landing tile, so the story
    cell is that tile and the A-spam starts from wherever travel_to lands.

    Returns "unknown_destination" | a travel_to pass-through | "story_done" |
    "story_timeout".
    """
    dest = DESTINATIONS.get(destination)
    if dest is None:
        return "unknown_destination"
    goal_map, goal_cell = dest
    outcome = travel_to(
        emulator, reader, memory, goal_map, goal_cell,
        max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    return _advance_story_dialogue(emulator, reader, story_target)


def _advance_story_dialogue(emulator: Any, reader: Any, story_target: Any) -> str:
    """Press A until story_target(reader) is satisfied (bounded)."""
    if story_target(reader):
        return "story_done"
    for _ in range(STORY_MAX_PRESSES):
        emulator.step(buttons.KEY_A, STORY_PRESS_A_FRAMES)
        emulator.step(0, STORY_RELEASE_FRAMES)   # release between presses (GBA debounce)
        if story_target(reader):
            return "story_done"
    return "story_timeout"
```

Update the `execute_order` docstring "Returns" list to add:

```
    story adds: "story_done" | "story_timeout".
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_orders.py -k "story" -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "$(cat <<'EOF'
feat: story Order mode — travel then A-spam to an injected target predicate

Sibling of heal: travel_to(cell) then a bounded A-spam loop until story_target
(reader) holds (flag set / item held / control regained). Order stays pure data;
the predicate threads through execute_order as a story_target kwarg, exactly like
target_level threads for level_up. Adds lab + route_101_shoes destinations.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `Milestone.story_target` + `PHASE2_CAMPAIGN` + campaign dispatch

**Files:**
- Modify: `env/campaign.py` (Milestone field, PHASE2_CAMPAIGN, run_campaign story branch)
- Test: `tests/test_campaign.py`

Return-to-Littleroot and enter-lab are pure arrivals → `advance` milestones. Pokédex, Poké Balls, and shoes use `story`. Pokédex and Balls are ONE cutscene, so the Balls milestone is an idempotent post-assert of the same A-spam (its target is already true when reached).

- [ ] **Step 1: Write the failing tests**

The existing `RecordingOrderFn` records `(order.mode, order.destination, kwargs.get("target_level"))`. Append to `tests/test_campaign.py`:

```python
from env.campaign import Milestone, PHASE2_CAMPAIGN, run_campaign


def test_milestone_story_target_defaults_none() -> None:
    assert Milestone("lab", 0).story_target is None


def test_phase2_campaign_covers_return_pokedex_balls_shoes() -> None:
    modes = [(m.destination, m.story_target is not None) for m in PHASE2_CAMPAIGN]
    assert modes == [
        ("littleroot", False),   # advance: return
        ("lab", False),          # advance: enter lab
        ("lab", True),           # story: Pokédex
        ("lab", True),           # story: Poke Balls (idempotent post-assert)
        ("route_101_shoes", True),  # story: running shoes
    ]


def test_story_milestone_emits_story_order_with_predicate() -> None:
    reader = FakeReader([8])   # over-leveled: no level_up
    fn = RecordingOrderFn(["story_done"])
    target = lambda r: True
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("lab", 0, story_target=target),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [("story", "lab", None)]
    assert fn.kwargs[0]["story_target"] is target


def test_story_milestone_failure_aborts_and_surfaces_outcome() -> None:
    reader = FakeReader([8])
    fn = RecordingOrderFn(["story_timeout"])
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("lab", 0, story_target=lambda r: False),),
        order_fn=fn,
    )
    assert result == "story_timeout"
    assert fn.calls == [("story", "lab", None)]


def test_advance_milestone_still_emits_advance_when_no_story_target() -> None:
    reader = FakeReader([8])
    fn = RecordingOrderFn(["arrived"])
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("littleroot", 0),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [("advance", "littleroot", None)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_campaign.py -k "story or phase2" -v`
Expected: FAIL with `ImportError: cannot import name 'PHASE2_CAMPAIGN'` / `AttributeError: 'Milestone' object has no attribute 'story_target'`.

- [ ] **Step 3: Write minimal implementation**

In `env/campaign.py`, add the imports and `Callable` (top, after `from typing import Any`, line ~15):

```python
from collections.abc import Callable
```

Add the `Milestone.story_target` field (last field, after `trainer: bool = False`, line ~27):

```python
    story_target: Callable[[Any], bool] | None = None   # story mode: A-spam until this holds
```

After the `CAMPAIGN` seed (line ~35), add:

```python
# Phase 2 curriculum: return to Littleroot, enter Birch's lab, receive Pokédex +
# 5 Poké Balls (one cutscene -> the Balls target is idempotent), then the running
# shoes on Route 101. Return/enter are pure arrivals (advance); the rest are story.
PHASE2_CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("littleroot", 0),
    Milestone("lab", 0),
    Milestone("lab", 0, story_target=lambda r: r.has_pokedex()),
    Milestone("lab", 0, story_target=lambda r: r.has_item(0x4, 5)),
    Milestone("route_101_shoes", 0, story_target=lambda r: r.has_running_shoes()),
)
```

In `run_campaign`, add the story dispatch. Replace the current advance block (lines ~59-83, from the `if not reached(...)` through the `trainer` block) with:

```python
        if milestone.story_target is not None:
            told = order_fn(
                Order(milestone.destination, "story", "win"),
                emulator, reader, memory,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                story_target=milestone.story_target,
            )
            if told != "story_done":
                return told
            continue
        if not reached(reader.party_levels(), milestone.target_level):
            leveled = order_fn(
                Order(milestone.destination, "level_up", "win"),
                emulator, reader, memory,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                target_level=milestone.target_level, heal_threshold=heal_threshold,
                max_cycles=max_cycles,
            )
            if leveled != "leveled_up":
                return leveled
        advanced = order_fn(
            Order(milestone.destination, "advance", "win"),
            emulator, reader, memory,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
        if advanced != "arrived":
            return advanced
        if milestone.trainer:
            fought = order_fn(
                Order(milestone.destination, "battle_trainer", "win"),
                emulator, reader, memory,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
            )
            if fought != "won":
                return fought
```

Update the `run_campaign` docstring "Returns" to add: `| any non-"story_done" outcome from a story Order`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest -q && .venv/bin/ruff check env tests`
Expected: all passed (ROM/artifact-gated tests skip), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "$(cat <<'EOF'
feat: PHASE2_CAMPAIGN + Milestone.story_target + run_campaign story dispatch

Return/enter-lab stay advance milestones; Pokédex/Balls/shoes use the story mode
with an injected reader predicate. Pokédex and Balls are one cutscene, so the
Balls milestone is an idempotent post-assert of the same A-spam.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Throwaway probe — freeze the live Phase 2 facts

**Files:**
- Create: `tools/probe_phase2_facts.py`
- Modify: `env/game_state.py` and/or `env/campaign.py` and/or `env/orders.py` ONLY if the probe finds a candidate constant/cell is wrong.

This probe is load-bearing: it confirms or replaces every candidate (flag ids, SAVE_BLOCK2_PTR, item pocket offset, security key offset, return-portal cells, lab warp-landing tile, and the A-press counts). It does NOT stay in the suite.

- [ ] **Step 1: Write the probe**

Create `tools/probe_phase2_facts.py`:

```python
"""Throwaway probe: freeze the live Phase 2 facts on BPEF before wiring durables.

Loads states/post_rival.state and, driving the Fighter for any wild along the way,
prints the ground truth the Phase 2 machinery encodes as candidates:
  - has_pokedex / has_running_shoes flag ids (scan a small flag-id window for the
    bit that flips when the Pokédex/shoes are received),
  - SAVE_BLOCK2_PTR validity + securityKey, and has_item(POKE_BALL, 5) after the
    lab cutscene (confirms the Items pocket offset + XOR decrypt),
  - the return-portal cells actually crossed (route_103->...->lab) and the lab
    warp-landing tile,
  - how many A-presses each cutscene needs (sanity-check STORY_MAX_PRESSES).

Run: POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" \
     .venv/bin/python tools/probe_phase2_facts.py states/post_rival.state
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emulator.gba import GbaEmulator
from env.game_state import (
    EmeraldReader,
    FLAG_SYS_POKEDEX_GET,
    FLAG_RECEIVED_RUNNING_SHOES,
    POKE_BALL_ITEM_ID,
    SAVE_BLOCK2_PTR,
)
from env.world_reader import WorldReader


def main() -> None:
    rom = os.environ["POKEMON_EMERALD_ROM"]
    state = sys.argv[1] if len(sys.argv) > 1 else "states/post_rival.state"
    emu = GbaEmulator(rom)
    with open(state, "rb") as fh:
        emu.load_state(fh.read())
    emu.step(0, 4)
    reader = EmeraldReader(emu.read_bytes)
    world = WorldReader(emu.read_bytes)

    here = world.snapshot()
    print(f"start map={here.map_id} pos={here.pos}")
    print(f"SAVE_BLOCK2_PTR=0x{SAVE_BLOCK2_PTR:08x} sb2_valid="
          f"{reader._save_block2() is not None}")
    print(f"has_pokedex(0x{FLAG_SYS_POKEDEX_GET:x})={reader.has_pokedex()} "
          f"has_running_shoes(0x{FLAG_RECEIVED_RUNNING_SHOES:x})="
          f"{reader.has_running_shoes()}")
    print(f"has_item(POKE_BALL={POKE_BALL_ITEM_ID}, 5)="
          f"{reader.has_item(POKE_BALL_ITEM_ID, 5)}")
    # The operator drives the campaign manually here (or wires run_campaign) and
    # re-prints these lines after each cutscene to confirm the flags/items flip.


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && POKEMON_EMERALD_ROM="$(pwd)/roms/pokemon_emerald_fr.gba" .venv/bin/python tools/probe_phase2_facts.py states/post_rival.state`
Expected: prints the start map/pos and the current flag/item readings. Record every printed value.

- [ ] **Step 3: Reconcile candidates with reality**

For each candidate that the probe contradicts, edit the source constant/cell and re-run the affected unit tests:
- Wrong flag id → edit `FLAG_SYS_POKEDEX_GET` / `FLAG_RECEIVED_RUNNING_SHOES` in `env/game_state.py`; re-run `pytest tests/test_game_state.py -k "pokedex or running_shoes"`.
- Wrong SAVE_BLOCK2_PTR / item offsets → edit in `env/game_state.py`; re-run `pytest tests/test_game_state.py -k "has_item"`.
- Wrong return-portal cell / lab tile → edit `_RETURN_PORTALS` in `env/campaign.py` or `DESTINATIONS["lab"]` / `DESTINATIONS["route_101_shoes"]` in `env/orders.py`; re-run `pytest tests/test_campaign.py tests/test_orders.py`.
If every candidate is confirmed, make no source edits.

- [ ] **Step 4: Commit (probe + any reconciliations)**

```bash
git add tools/probe_phase2_facts.py env/game_state.py env/campaign.py env/orders.py
git commit -m "$(cat <<'EOF'
probe: freeze live Phase 2 facts (flag ids, item offsets, return cells, A-counts)

Confirms or replaces the candidate constants the Phase 2 machinery encodes,
before the ROM smoke goes load-bearing. Throwaway diagnostic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Gated ROM smoke — run PHASE2_CAMPAIGN end to end

**Files:**
- Create: `tests/test_phase2_rom.py`

Triple-skip (ROM unset | Fighter checkpoint missing | `states/post_rival.state` missing). Load-bearing once the probe (Task 6) has confirmed the constants: it runs the real campaign and asserts the deliverables landed.

- [ ] **Step 1: Write the gated smoke**

Create `tests/test_phase2_rom.py`:

```python
"""Gated ROM smoke: run PHASE2_CAMPAIGN from post_rival.state end to end.

Load-bearing once tools/probe_phase2_facts.py has confirmed the constants: seeds
the return portals, drives the scripted campaign (return -> lab -> Pokedex ->
Balls -> shoes) with the real Fighter, and asserts the deliverables landed. Dumps
states/post_phase2.state for the next phase.
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


def test_phase2_campaign_delivers_pokedex_and_running_shoes() -> None:
    from stable_baselines3 import PPO

    from agent.train_fighter import make_move_type_fn
    from emulator.gba import GbaEmulator
    from env.campaign import PHASE2_CAMPAIGN, seed_return_portals, run_campaign
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
    seed_return_portals(memory)

    model = PPO.load(FIGHTER_CKPT, device="cpu")
    predict = lambda obs: int(model.predict(obs, deterministic=True)[0])
    move_type_fn = make_move_type_fn(emu)

    # The campaign reads world/game state off `emu`; run_campaign expects a reader
    # that also exposes snapshot/party_* — a thin adapter forwards both readers.
    class _Reader:
        def __getattr__(self, name):
            for src in (world, reader):
                if hasattr(src, name):
                    return getattr(src, name)
            raise AttributeError(name)

    result = run_campaign(
        emu, _Reader(), memory,
        curriculum=PHASE2_CAMPAIGN,
        move_type_fn=move_type_fn, predict=predict,
    )

    assert result == "campaign_complete", result
    assert reader.has_pokedex()
    assert reader.has_running_shoes()

    Path("states/post_phase2.state").write_bytes(emu.save_state())
```

- [ ] **Step 2: Run the smoke (skips without artifacts)**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest tests/test_phase2_rom.py -v`
Expected: SKIP (all three gates) in CI. Locally with ROM + Fighter + `states/post_rival.state`: PASS, and `states/post_phase2.state` is written.

- [ ] **Step 3: Run the full suite + ruff**

Run: `cd /Users/_eloi/Projets/Emu-phase2-story-campaign && .venv/bin/python -m pytest -q && .venv/bin/ruff check env tests tools`
Expected: all passed (gated tests skip), ruff clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase2_rom.py
git commit -m "$(cat <<'EOF'
test: gated ROM smoke — run PHASE2_CAMPAIGN from post_rival.state end to end

Seeds return portals, drives the scripted campaign with the real Fighter, asserts
has_pokedex() AND has_running_shoes(), and dumps states/post_phase2.state.
Triple-skips without ROM / Fighter / post_rival.state.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- `story` mode (travel + A-spam to predicate) → Task 4. ✓
- `has_pokedex` / `has_running_shoes` / `has_item` (securityKey XOR) → Tasks 1-2. ✓
- G1 `seed_return_portals` + MapMemory/plan_route regression → Task 3. ✓
- G2 lab warp-landing tile as the story cell (0 intra-map steps) → `DESTINATIONS["lab"]` in Task 4; A-spam starts from wherever travel lands. ✓
- G3 return/enter as advance, only Pokédex/Balls/shoes as story → `PHASE2_CAMPAIGN` shape asserted in Task 5. ✓
- G4 Pokédex + Balls one cutscene, Balls idempotent post-assert → `has_item` target already true when reached; covered by the campaign shape. ✓
- G5 `SAVE_BLOCK2_PTR` → Task 2. ✓
- Throwaway probe freezes live facts → Task 6. ✓
- Gated ROM smoke load-bearing → Task 7. ✓
- RL `milestones.py` untouched → no task edits it. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows complete code; every run step shows an exact command + expected output. ✓

**Type consistency:** `story_target: Callable[[Any], bool] | None` (campaign) matches the `story_target: Any` kwarg threaded through `execute_order` → `_execute_story` → `_advance_story_dialogue` (predicate called as `story_target(reader)`). `_execute_story` returns `"story_done"|"story_timeout"|"unknown_destination"|<travel pass-through>`; `run_campaign` gates on `== "story_done"`. Constant names (`FLAG_SYS_POKEDEX_GET`, `FLAG_RECEIVED_RUNNING_SHOES`, `SAVE_BLOCK2_PTR`, `POKE_BALL_ITEM_ID`, `_ITEMS_POCKET_OFFSET`, `_SECURITY_KEY_OFFSET`, `_ITEM_SLOT_SIZE`) are used identically across game_state.py and the tests. Map-id constants (`ROUTE_103/OLDALE/ROUTE_101/LITTLEROOT/LAB`) are consistent between `seed_return_portals` and its test. ✓
