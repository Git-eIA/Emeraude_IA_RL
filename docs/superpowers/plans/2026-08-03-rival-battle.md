# Rival Battle (`battle_trainer` mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `battle_trainer` Order mode that travels to a trainer, engages, and plays the multi-mon battle via `play_trainer_battle`, plus the campaign milestone that emits it.

**Architecture:** A new self-contained Order mode in `env/orders.py` (twin of grind: travel_to → bounded engage-walk on the in_battle front → `play_trainer_battle`). `env/campaign.py` gains a `Milestone.trainer` flag and emits the mode after `advance` arrives, staying pure sequencing. A disposable capture tool produces `states/trainer_battle.state` (route_103 rival), which makes Brique 3 part 1's existing gated smoke load-bearing. The mode's own ROM smoke is deferred (documented): it would need a pre-trigger overworld state + surveyed nav, which a walk-until-in_battle tool cannot produce.

**Tech Stack:** Python 3.12, pytest (pure, no ROM), existing `env/` modules (`map_traveler.travel_to`, `battle_player.play_trainer_battle`, `encounter_detector.EncounterWatcher`, `emulator.buttons`).

**How to run tests (from the worktree):** the worktree has no local venv — use the main repo interpreter:
`/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -v`

---

### Task 1: `battle_trainer` mode in `env/orders.py`

**Files:**
- Modify: `env/orders.py`
- Test: `tests/test_orders.py`

The mode resolves its own destination (like advance), travels there, engages with a
bounded approach walk, and — with a Fighter wired — plays the trainer battle. Reuses
grind's constants (`GRIND_STEP_FRAMES`, `GRIND_RELEASE_FRAMES`, `GRIND_MAX_STEPS`) and
`EncounterWatcher`. Adds a `route_103` destination (unverified coords, backfilled by the
capture tool) and a `TRAINER_APPROACH` heading table.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_orders.py` (the fakes `NamedWorld`, `GrassBattleWorld` and helper
`_u16b` already exist in this file — reuse them):

```python
# ---------------------------------------------------------------------------
# battle_trainer mode tests
# ---------------------------------------------------------------------------


def test_route_103_is_a_known_destination() -> None:
    assert "route_103" in DESTINATIONS


def test_battle_trainer_wins_at_the_destination() -> None:
    map_id, cell = DESTINATIONS["route_103"]
    world = GrassBattleWorld(map_id, cell, steps_to_encounter=3)
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), WallMap(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "won"


def test_battle_trainer_without_fighter_reports_encounter_started() -> None:
    map_id, cell = DESTINATIONS["route_103"]
    world = GrassBattleWorld(map_id, cell, steps_to_encounter=3)
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "encounter_started"


def test_battle_trainer_without_a_trainer_returns_no_trainer() -> None:
    map_id, cell = DESTINATIONS["route_103"]
    world = GrassBattleWorld(map_id, cell, steps_to_encounter=10_000)
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(
        order, world, world, MapMemory(), WallMap(),
        move_type_fn=lambda mid: 12, predict=lambda obs: 0,
    )
    assert result == "no_trainer"


def test_battle_trainer_unknown_destination() -> None:
    order = Order(destination="atlantide", mode="battle_trainer", combat="win")
    result = execute_order(order, None, None, MapMemory(), WallMap())
    assert result == "unknown_destination"


def test_battle_trainer_passes_through_travel_failure() -> None:
    world = NamedWorld(start_map=(0, 9), start_cell=(0, 10))
    order = Order(destination="route_103", mode="battle_trainer", combat="win")
    result = execute_order(order, world, world, MapMemory(), WallMap())
    assert result == "unknown_route"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -k battle_trainer -v`
Expected: FAIL — `test_route_103_is_a_known_destination` KeyError/assert (route_103 not
in DESTINATIONS), others fail because `execute_order` returns something other than the
expected string for `mode="battle_trainer"` (falls through to the advance path).

- [ ] **Step 3: Add the destination, approach table, and import**

In `env/orders.py`, extend `DESTINATIONS` and add `TRAINER_APPROACH` right after it:

```python
DESTINATIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "littleroot": ((0, 9), (3, 10)),   # Bourg-en-Vol, truck landing cell
    "route_101": ((0, 16), (5, 12)),   # Route 101 south entrance (cell unverified)
    "route_103": ((0, 18), (9, 5)),    # Route 103 rival approach (map+cell unverified)
}

# Engage heading per trainer destination: the d-pad direction that walks into the
# trainer to trigger the battle. Unverified until the capture tool discovers it.
TRAINER_APPROACH: dict[str, int] = {
    "route_103": buttons.KEY_UP,   # walk north into the rival (unverified)
}
```

Update the `play_battle` import line to also import `play_trainer_battle`:

```python
from env.battle_player import play_battle, play_trainer_battle
```

- [ ] **Step 4: Add the mode dispatch**

In `execute_order`, add a branch after the `level_up` branch and before the
`dest = DESTINATIONS.get(order.destination)` line:

```python
    if order.mode == "battle_trainer":
        return _execute_battle_trainer(
            emulator, reader, memory, wallmap, order.destination,
            max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
        )
```

- [ ] **Step 5: Implement the mode and the engage helper**

Add after `_execute_grind` / `_walk_until_encounter` in `env/orders.py`:

```python
def _execute_battle_trainer(
    emulator: Any,
    reader: Any,
    memory: Any,
    wallmap: Any,
    destination: str,
    max_hops: int = 20,
    move_type_fn: Any = None,
    predict: Any = None,
) -> str:
    """Travel to a trainer's approach cell, engage, then—if a Fighter is
    supplied—play the trainer battle to an outcome.

    ROM smoke deferred: a mode-level smoke needs a pre-trigger overworld state
    near the trainer plus a surveyed route so travel_to can path there. A
    walk-until-in_battle capture tool produces an already-in-battle state, on
    which travel_to spins to timeout before the battle. Pure tests cover the
    sequencing; play_trainer_battle on a real rival is covered by Brique 3
    part 1's gated smoke.

    Returns "unknown_destination" | a travel_to pass-through | "no_trainer" |
    "encounter_started" (no Fighter) | a play_trainer_battle outcome
    ("won" | "lost" | "battle_timeout").
    """
    dest = DESTINATIONS.get(destination)
    if dest is None:
        return "unknown_destination"
    goal_map, goal_cell = dest
    outcome = travel_to(
        emulator, reader, memory, wallmap, goal_map, goal_cell,
        max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
    )
    if outcome != "arrived":
        return outcome               # pass-through: unknown_route/unreachable/lost/timeout
    heading = TRAINER_APPROACH.get(destination, buttons.KEY_UP)
    engaged = _walk_until_trainer(emulator, reader, heading)
    if engaged != "engaged":
        return engaged               # no_trainer
    if move_type_fn is None or predict is None:
        return "encounter_started"   # no Fighter wired: stop at the trigger
    return play_trainer_battle(emulator, move_type_fn, predict)


def _walk_until_trainer(emulator: Any, reader: Any, heading: int) -> str:
    """Press a fixed approach heading in place until the battle flag rises."""
    watcher = EncounterWatcher()
    for _ in range(GRIND_MAX_STEPS):
        if watcher.observe(reader.in_battle()):
            return "engaged"
        emulator.step(heading, GRIND_STEP_FRAMES)
        emulator.step(0, GRIND_RELEASE_FRAMES)   # release between presses (GBA debounce)
    return "engaged" if reader.in_battle() else "no_trainer"
```

Also extend the `execute_order` docstring "Returns" list to mention the new mode:
after the `level_up adds:` line, add:

```
    battle_trainer adds: "no_trainer" (Fighter wired, no battle triggered).
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -k battle_trainer -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Run the full orders suite to check for regressions**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_orders.py -v`
Expected: PASS (all prior tests + the 6 new ones).

- [ ] **Step 8: Commit**

```bash
git add env/orders.py tests/test_orders.py
git commit -m "$(cat <<'EOF'
feat: battle_trainer Order mode — travel to a trainer, engage, play the battle

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `Milestone.trainer` + campaign wiring + seed

**Files:**
- Modify: `env/campaign.py`
- Test: `tests/test_campaign.py`

`Milestone` gains a `trainer: bool = False` field. `run_campaign`, for a trainer
milestone, emits a `battle_trainer` Order after `advance` returns `"arrived"`, aborting
verbatim on any non-`"won"` outcome. The `CAMPAIGN` seed gains `route_103`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_campaign.py`:

```python
def test_milestone_trainer_defaults_false():
    assert Milestone("route_101", 5).trainer is False


def test_campaign_seed_has_route_103_trainer_milestone():
    route_103 = next(m for m in CAMPAIGN if m.destination == "route_103")
    assert route_103.trainer is True
    assert route_103.target_level == 5


def test_trainer_milestone_advances_then_battles():
    reader = FakeReader([8])  # over-leveled -> straight to advance
    fn = RecordingOrderFn(["arrived", "won"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_103", 5, trainer=True),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [
        ("advance", "route_103", None),
        ("battle_trainer", "route_103", None),
    ]


def test_trainer_battle_failure_aborts_after_advance():
    reader = FakeReader([8])
    fn = RecordingOrderFn(["arrived", "lost"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_103", 5, trainer=True),),
        order_fn=fn,
    )
    assert result == "lost"
    assert fn.calls == [
        ("advance", "route_103", None),
        ("battle_trainer", "route_103", None),
    ]


def test_non_trainer_milestone_does_not_battle():
    reader = FakeReader([8])
    fn = RecordingOrderFn(["arrived"])
    result = run_campaign(
        None, reader, None, None,
        curriculum=(Milestone("route_101", 5),),  # trainer defaults False
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [("advance", "route_101", None)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: FAIL — `Milestone` has no `trainer` field (TypeError on the keyword), the seed
has no `route_103`, and `run_campaign` never emits `battle_trainer`.

- [ ] **Step 3: Add the `trainer` field**

In `env/campaign.py`, extend the `Milestone` dataclass:

```python
@dataclass(frozen=True)
class Milestone:
    """One curriculum step: reach `destination` once the mean party level is at
    least `target_level`; if `trainer`, fight the trainer there on arrival."""

    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # mean, not max — one powerhouse shouldn't unlock advance
    trainer: bool = False   # end the milestone with a battle_trainer Order
```

- [ ] **Step 4: Extend the seed**

```python
CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("route_101", 5),
    Milestone("route_103", 5, trainer=True),
)
```

- [ ] **Step 5: Wire the trainer battle into `run_campaign`**

In `run_campaign`, after the `advance` block (right after `if advanced != "arrived":
return advanced`) and before the loop closes, add:

```python
        if milestone.trainer:
            fought = order_fn(
                Order(milestone.destination, "battle_trainer", "win"),
                emulator, reader, memory, wallmap,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
            )
            if fought != "won":
                return fought
```

Update the `run_campaign` docstring "Returns" line to mention the new outcome source:

```
    Returns "campaign_complete" | any non-"leveled_up" outcome from a level_up
    Order | any non-"arrived" outcome from an advance Order | any non-"won"
    outcome from a battle_trainer Order.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest tests/test_campaign.py -v`
Expected: PASS (all prior tests + the 4 new ones; `test_campaign_seed_is_a_tuple_of_milestones` still passes since `CAMPAIGN[0]` is unchanged).

- [ ] **Step 7: Commit**

```bash
git add env/campaign.py tests/test_campaign.py
git commit -m "$(cat <<'EOF'
feat: campaign trainer milestone — advance then battle_trainer on arrival

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `tools/capture_rival_battle.py` (disposable) + spec finding

**Files:**
- Create: `tools/capture_rival_battle.py`
- Modify: `docs/superpowers/specs/2026-08-03-rival-battle-design.md`

Disposable scaffolding (no unit test, `chore` commit — the capture-tool precedent). It
walks toward route_103 and, unlike `capture_trainer_battle.py` (which saves on ANY
battle), it verifies the battle is on the route_103 map before saving to
`states/trainer_battle.state`; a wild battle on another map aborts with guidance. This
makes Brique 3 part 1's gated smoke load-bearing once run, and discovers the real
route_103 map/cell/heading to backfill into `DESTINATIONS` + `TRAINER_APPROACH`.

- [ ] **Step 1: Write the tool**

Create `tools/capture_rival_battle.py`:

```python
"""Walk post_starter toward route_103 until the rival trainer battle fires, then
cache states/trainer_battle.state.

Unlike tools/capture_trainer_battle.py (which saves on ANY in_battle), this
verifies the battle is on the target map (route_103) before saving, so a wild
battle in route_101 grass on the way does not get captured by mistake. If a
battle fires on a non-target map, the character is stuck in it and the tool
aborts with guidance (start from a state already past the grass).

The captured states/trainer_battle.state is exactly the artifact that Brique 3
part 1's gated smoke (tests/test_battle_player_rom.py::
test_fighter_wins_a_real_trainer_battle) needs — so running this makes THAT
smoke load-bearing. This run also reveals the real route_103 map_id, approach
cell, and engage heading to backfill into env/orders.DESTINATIONS and
TRAINER_APPROACH (both currently unverified).

IMPORTANT: reader.in_battle() can be a false-positive in some post-cutscene
states. The tool trusts it but prints the map_id; the operator must eyeball the
saved state to confirm it is the genuine rival battle.

One-shot scaffolding: run once locally where the ROM + post_starter.state exist.
Output is gitignored.

Usage (cwd = main repo):
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_rival_battle.py
  POKEMON_EMERALD_ROM=... .venv/bin/python <worktree>/tools/capture_rival_battle.py \\
      --target-map 0 18 --heading up --max-steps 800
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from emulator.buttons import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP
from emulator.gba import GbaEmulator
from env.orders import GRIND_RELEASE_FRAMES, GRIND_STEP_FRAMES
from env.pokemon_env import PokemonEmeraldEnv
from env.world_reader import WorldReader

OUT_PATH = Path("states/trainer_battle.state")
_DIRECTIONS = (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT)
_HEADING_MAP = {"up": KEY_UP, "down": KEY_DOWN, "left": KEY_LEFT, "right": KEY_RIGHT}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Capture the route_103 rival battle.")
    ap.add_argument("--state", default="states/post_starter.state")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument(
        "--heading",
        choices=list(_HEADING_MAP),
        default="up",
        help="Walk direction toward route_103 (default: up).",
    )
    ap.add_argument(
        "--target-map",
        type=int,
        nargs=2,
        default=[0, 18],
        metavar=("GROUP", "NUM"),
        help="route_103 map id to accept the battle on (default: 0 18, unverified).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    target_map = (args.target_map[0], args.target_map[1])

    rom = os.environ["POKEMON_EMERALD_ROM"]
    start = Path(args.state).read_bytes()
    env = PokemonEmeraldEnv(GbaEmulator(rom), [start], max_steps=10_000_000)
    reader = WorldReader(env.emulator.read_bytes)
    env.reset()

    key = _HEADING_MAP[args.heading]
    for i in range(args.max_steps):
        env.emulator.step(key, GRIND_STEP_FRAMES)
        env.emulator.step(0, GRIND_RELEASE_FRAMES)   # release (GBA debounce)
        if reader.in_battle():
            snap = reader.snapshot()
            map_id = None if snap is None else snap.map_id
            if map_id != target_map:
                print(
                    f"battle fired on map {map_id} (not target {target_map}) after "
                    f"{i + 1} steps — likely a wild encounter blocking the path. "
                    f"Start from a state past the grass, or set --target-map.",
                    flush=True,
                )
                raise SystemExit(1)
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_bytes(env.emulator.save_state())
            print(
                f"RIVAL-BATTLE state saved after {i + 1} steps "
                f"(map {map_id}, pos {None if snap is None else snap.pos}) "
                f"-> {OUT_PATH.resolve()}",
                flush=True,
            )
            print(
                "NOTE: verify this is the genuine rival battle (eyeball the state) — "
                "in_battle() can be a false-positive. Backfill the real map/cell/"
                "heading into env/orders.DESTINATIONS + TRAINER_APPROACH.",
                flush=True,
            )
            return

    print(f"no battle in {args.max_steps} steps; nothing saved", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the tool parses (import + argparse, no ROM needed)**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -c "import ast; ast.parse(open('tools/capture_rival_battle.py').read()); print('parse OK')"`
Expected: `parse OK`

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python tools/capture_rival_battle.py --help`
Expected: argparse usage printed (no ROM access, exits 0).

- [ ] **Step 3: Record the finding in the spec**

Append a `## Finding (implementation)` section to
`docs/superpowers/specs/2026-08-03-rival-battle-design.md`:

```markdown
## Finding (implementation)

`tools/capture_trainer_battle.py` already existed and saves `states/trainer_battle.state`
on ANY in_battle. `tools/capture_rival_battle.py` is the route_103-specific variant: it
verifies the battle is on the target map before saving (a wild battle on the way aborts
with guidance), and defaults `--heading up`. Running it locally is what makes Brique 3
part 1's gated `test_fighter_wins_a_real_trainer_battle` load-bearing and reveals the
real route_103 map/cell/heading to backfill into `DESTINATIONS` + `TRAINER_APPROACH`
(currently the unverified placeholders `((0, 18), (9, 5))` and `KEY_UP`). Not run in CI.
```

- [ ] **Step 4: Commit**

```bash
git add tools/capture_rival_battle.py docs/superpowers/specs/2026-08-03-rival-battle-design.md
git commit -m "$(cat <<'EOF'
chore: capture_rival_battle tool + record route_103 capture finding

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full pure suite (no ROM)**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: all tests pass (the ROM-gated smokes skip without `POKEMON_EMERALD_ROM`).

- [ ] **Step 2: Run ruff**

Run: `/Users/_eloi/Projets/Emu/.venv/bin/ruff check env/orders.py env/campaign.py tools/capture_rival_battle.py tests/test_orders.py tests/test_campaign.py`
Expected: no errors.

- [ ] **Step 3: (Optional, if ROM present) run with ROM**

Run: `POKEMON_EMERALD_ROM=/Users/_eloi/Projets/Emu/roms/pokemon_emerald_fr.gba /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: prior counts + these tasks' pure tests pass; the trainer smoke skips until
`states/trainer_battle.state` is captured.

---

## Self-review

**Spec coverage:**
- Component 1 (`battle_trainer` mode) → Task 1. ✓
- Component 2 (campaign `Milestone.trainer` + wiring + seed) → Task 2. ✓
- Component 3 (`DESTINATIONS` + `TRAINER_APPROACH`) → Task 1 Step 3. ✓
- Component 4 (capture tool) → Task 3. ✓
- Component 5 (mode-level ROM smoke deferred, documented) → Task 1 Step 5 docstring +
  Task 3 spec finding. ✓
- Key assumption (destination out of line of sight) → carried in the mode docstring and
  spec; no code beyond travel_to arriving cleanly. ✓

**Placeholder scan:** none — route_103 coords/heading are intentional, flagged-unverified
seed values backfilled by the capture tool, not plan placeholders.

**Type consistency:** `_execute_battle_trainer` / `_walk_until_trainer` signatures and
outcome strings (`unknown_destination`, `no_trainer`, `encounter_started`, `won`/`lost`/
`battle_timeout`) match the spec and the dispatch call. `Milestone(destination,
target_level, trainer=False)` matches every call site (`RecordingOrderFn` records
`(mode, destination, target_level)`; `battle_trainer` passes no `target_level`, recorded
as `None`). `TRAINER_APPROACH` values are `int` (buttons), keyed by destination name.
