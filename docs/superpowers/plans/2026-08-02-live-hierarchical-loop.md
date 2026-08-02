# Live Hierarchical Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scripted-Strategist driver (`run_campaign`) that walks a hand-written milestone curriculum, calling `execute_order` to level up then advance at each step — the first time the Order interface runs in a loop.

**Architecture:** One new module `env/campaign.py` composing the existing `env.orders` API. It adds only sequencing logic (level-check → level_up → advance per milestone); all navigation/combat/RAM logic stays in `execute_order`, injected as `order_fn` for pure testing.

**Tech Stack:** Python 3.12, pytest. No ROM, no SB3 — tests are pure with injected fakes.

**Test command (from worktree `/Users/_eloi/Projets/Emu-p4-live-loop`):**
`/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`

---

### Task 1: Milestone dataclass + CAMPAIGN seed

**Files:**
- Create: `env/campaign.py`
- Test: `tests/test_campaign.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

from env.campaign import CAMPAIGN, Milestone


def test_milestone_holds_destination_and_target_level():
    m = Milestone("route_101", 5)
    assert m.destination == "route_101"
    assert m.target_level == 5


def test_campaign_seed_is_a_tuple_of_milestones():
    assert isinstance(CAMPAIGN, tuple)
    assert all(isinstance(m, Milestone) for m in CAMPAIGN)
    assert CAMPAIGN[0].destination == "route_101"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'env.campaign'`

- [ ] **Step 3: Write minimal implementation**

Create `env/campaign.py`:

```python
"""campaign: a scripted Strategist that drives the Order loop over milestones.

The chef holds a hand-written curriculum of (named destination, required mean
party level). For each milestone: if the team is under the required level, emit a
level_up Order (which grinds + heals itself to the target); then emit an advance
Order to reach the destination. run_campaign composes execute_order — it adds no
navigation, combat, or RAM logic of its own, only the sequencing.

advance is navigation-only in v1 (reach the place); fighting the leader there is
deferred. No trained Strategist, no capture directive here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env.orders import Order, _reached, execute_order


@dataclass(frozen=True)
class Milestone:
    """One curriculum step: reach `destination` once the mean party level is at
    least `target_level`."""

    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # required mean party level before advancing


# Hand-written curriculum. Like DESTINATIONS, a name means something to the chef
# before any exploration. Seeded minimally; extend as destinations are verified.
CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("route_101", 5),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "feat: Milestone dataclass + CAMPAIGN seed curriculum"
```

---

### Task 2: run_campaign driver — under-leveled milestone

**Files:**
- Modify: `env/campaign.py`
- Test: `tests/test_campaign.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_campaign.py` (add `run_campaign` to the existing import from
`env.campaign`, and `Order` from `env.orders`):

```python
from env.campaign import CAMPAIGN, Milestone, run_campaign
from env.orders import Order


class FakeReader:
    """Supplies party_levels(); run_campaign reads nothing else off the reader."""

    def __init__(self, levels: list[int]) -> None:
        self._levels = levels

    def party_levels(self) -> list[int]:
        return self._levels


class RecordingOrderFn:
    """Stand-in for execute_order: records each emitted Order and returns a
    scripted outcome per call, in order."""

    def __init__(self, outcomes: list[str]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, str, int | None]] = []

    def __call__(self, order: Order, emulator, reader, memory, wallmap, **kwargs):
        self.calls.append((order.mode, order.destination, kwargs.get("target_level")))
        return self._outcomes.pop(0)


def test_under_leveled_milestone_emits_level_up_then_advance():
    reader = FakeReader([3])  # mean level 3 < target 5
    fn = RecordingOrderFn(["leveled_up", "arrived"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [
        ("level_up", "route_101", 5),
        ("advance", "route_101", None),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py::test_under_leveled_milestone_emits_level_up_then_advance -v`
Expected: FAIL with `ImportError: cannot import name 'run_campaign'`

- [ ] **Step 3: Write minimal implementation**

Append to `env/campaign.py`:

```python
def run_campaign(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    curriculum: tuple[Milestone, ...] = CAMPAIGN,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
    heal_threshold: float = 0.4,
    max_cycles: int = 50,
    order_fn: Any = execute_order,
) -> str:
    """Walk the curriculum: for each milestone, level_up if under the required
    mean level, then advance to the destination. Abort on the first non-terminal
    outcome, surfaced verbatim so a future Strategist can react.

    Returns "campaign_complete" | any non-"leveled_up" outcome from a level_up
    Order | any non-"arrived" outcome from an advance Order.
    """
    for milestone in curriculum:
        if not _reached(reader.party_levels(), milestone.target_level):
            leveled = order_fn(
                Order(milestone.destination, "level_up", "win"),
                emulator, reader, memory, wallmap,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                target_level=milestone.target_level, heal_threshold=heal_threshold,
                max_cycles=max_cycles,
            )
            if leveled != "leveled_up":
                return leveled
        advanced = order_fn(
            Order(milestone.destination, "advance", "win"),
            emulator, reader, memory, wallmap, max_hops=max_hops,
        )
        if advanced != "arrived":
            return advanced
    return "campaign_complete"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "feat: run_campaign drives level_up then advance per milestone"
```

---

### Task 3: Skip, ordering, and abort behaviors

**Files:**
- Test: `tests/test_campaign.py`

These cases all pass against the Task 2 driver; they lock in the remaining
requirements (over-leveled skip, multi-milestone ordering, abort paths).

- [ ] **Step 1: Write the tests**

Add to `tests/test_campaign.py`:

```python
def test_over_leveled_milestone_skips_level_up():
    reader = FakeReader([8])  # mean level 8 >= target 5
    fn = RecordingOrderFn(["arrived"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [("advance", "route_101", None)]


def test_multiple_milestones_run_in_order():
    reader = FakeReader([9])  # over-leveled for both -> advance-only each
    fn = RecordingOrderFn(["arrived", "arrived"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_101", 5), Milestone("littleroot", 5)),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [
        ("advance", "route_101", None),
        ("advance", "littleroot", None),
    ]


def test_level_up_failure_aborts_without_advancing():
    reader = FakeReader([3])  # under-leveled -> level_up first
    fn = RecordingOrderFn(["lost"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "lost"
    assert fn.calls == [("level_up", "route_101", 5)]


def test_advance_failure_aborts_and_surfaces_outcome():
    reader = FakeReader([8])  # over-leveled -> straight to advance
    fn = RecordingOrderFn(["unknown_route"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "unknown_route"
    assert fn.calls == [("advance", "route_101", None)]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: PASS (7 passed)

- [ ] **Step 3: Run the full suite + ruff**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py tests/test_orders.py -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/campaign.py tests/test_campaign.py`
Expected: all pass, ruff clean

- [ ] **Step 4: Commit**

```bash
git add tests/test_campaign.py
git commit -m "test: campaign skip/ordering/abort behaviors"
```

---

## Self-Review

**1. Spec coverage:**
- Milestone / CAMPAIGN data → Task 1.
- `run_campaign` driver (level-check → level_up → advance, bounded loop, injected `order_fn`, abort-verbatim) → Task 2.
- Under/over-leveled, ordering, abort outcomes → Tasks 2–3.
- Non-goals (no PPO wrap, no gym battle, no capture, no nearest-spot) → nothing implements them, correct.
- ROM smoke deferred → not in plan, matches spec.

**2. Placeholder scan:** none — every step has full code and exact commands.

**3. Type consistency:** `Milestone(destination, target_level)`, `run_campaign(...) -> str`, `_reached`/`Order`/`execute_order` imported from `env.orders`, recording fake records `(mode, destination, target_level)`. Consistent across tasks.
