# P4 — Grind level-up / auto-heal loop

**Date:** 2026-07-31

## Goal

Turn the single-battle `grind` primitive into a real farming loop: fight wild
battles repeatedly to raise the party, healing when it gets low, until the
party's **average** level reaches a caller-given target. This is the "vrai
grind" a player does — level up just enough, heal just enough — chaining the
Explorer's grass navigation, the treading, the trained Fighter, and the healing
spot into one autonomous cycle.

## Scope (one small step)

Raise the **current** party to a target average level. NOT in scope:
- building a strong team (capturing, PC box management, type coverage,
  evolutions, choosing which 6 to carry) — that is the Strategist's domain plus
  a capture-capable Fighter, a separate subsystem (see Non-goals),
- the `capture` / `min_loss` combat directives (Fighter only knows how to win),
- nearest-spot selection, a real Strategist emitting Orders, carrying the target
  level on the `Order` dataclass.

## Current state

- `env/orders.py::_execute_grind` travels to a known grass cell, treads until a
  wild battle starts, and — when a Fighter (`move_type_fn` + `predict`) is
  supplied — plays that **single** battle to `"won"`/`"lost"`/`"battle_timeout"`.
- `env/orders.py::_execute_heal` travels to a known healing spot and presses A
  until the party reads full HP (`"healed"`/`"heal_failed"`/pass-through).
- `env/world_reader.py::WorldReader` exposes `party_hp()` and `in_battle()` as
  passthroughs, but **not** `party_levels()` (the RAM `EmeraldReader` has it).
- `env/heal_detector.py` has `party_is_full(hp)` and `HealWatcher`.

## Design

### Architecture

New Order mode `"level_up"`, executed by a new `_execute_level_up` in
`env/orders.py` that **composes** the two existing primitives
(`_execute_grind`, `_execute_heal`) without modifying them. `"grind"` stays a
single battle (the building block is untouched, its tests unchanged). Two small
pure decision helpers are added.

### Parameters

`level_up`'s knobs are passed to `execute_order` (mirroring how `move_type_fn`
/`predict` were injected — the Strategist does not emit Orders yet, so the
frozen `Order` dataclass is left alone):

- `target_level: int` — target for the **mean** of the party's levels.
- `heal_threshold: float = 0.4` — total-HP fraction below which we heal.
- `max_cycles: int = 50` — safety bound on the loop (code-safety #2).

These sit alongside the existing `max_hops`, `move_type_fn`, `predict` on
`execute_order` and are forwarded to `_execute_level_up`.

### Brick 1 — read levels through the shared reader

`WorldReader.party_levels() -> list[int]` — passthrough to `self._reader`,
exactly like `party_hp()`. `WorldReader` is the single reader object passed to
`execute_order`/`travel_to`/`navigate_to`, so the loop must read levels through
it.

### Brick 2 — heal decision (pure, in `env/heal_detector.py`)

```python
def party_needs_heal(hp: list[tuple[int, int]], threshold: float) -> bool:
    """True if any member has fainted (0 HP) OR the party's total HP fraction
    is below `threshold`. False for an empty party."""
    if not hp:
        return False
    if any(cur == 0 for cur, _ in hp):
        return True
    total_cur = sum(cur for cur, _ in hp)
    total_max = sum(mx for _, mx in hp)
    return total_max > 0 and total_cur / total_max < threshold
```

Mixed trigger, as decided: a KO forces a heal even if the totals look fine, and
a low team total forces a heal even with nobody KO'd.

### Brick 3 — the loop

```python
def _execute_level_up(emulator, reader, memory, wallmap,
                      target_level, heal_threshold=0.4, max_cycles=50,
                      max_hops=20, move_type_fn=None, predict=None) -> str:
    for _ in range(max_cycles):
        levels = reader.party_levels()
        if _reached(levels, target_level):
            return "leveled_up"
        result = _execute_grind(emulator, reader, memory, wallmap,
                                max_hops=max_hops,
                                move_type_fn=move_type_fn, predict=predict)
        if result == "won":
            if party_needs_heal(reader.party_hp(), heal_threshold):
                healed = _execute_heal(emulator, reader, memory, wallmap,
                                       max_hops=max_hops)
                if healed != "healed":
                    return healed        # no_healing_spot_known / pass-through / heal_failed
        elif result == "no_encounter":
            continue                     # no battle this cycle (RNG) — retry, budget-bounded
        else:
            return result                # lost / battle_timeout / no_grass_spot_known / travel
    return "leveled_up" if _reached(reader.party_levels(), target_level) else "grind_exhausted"
```

`_reached(levels, target)` = `bool(levels) and sum(levels) / len(levels) >= target`.

**Check level BEFORE fighting:** if the party is already at the target average,
return `"leveled_up"` immediately with zero battles.

Every loop is bounded (`max_cycles`, and each sub-call is itself bounded), so
the loop provably terminates (code-safety #2). No reward, no Strategist here.

### Outcomes

`execute_order(mode="level_up")` returns:
- `"leveled_up"` — target average reached.
- `"grind_exhausted"` — `max_cycles` spent without reaching the target.
- `"no_grass_spot_known"` — no grass cell learned (from `_execute_grind`).
- `"no_healing_spot_known"` / `"heal_failed"` — heal needed but unavailable/failed.
- travel pass-through (`"unknown_route"`/`"unreachable"`/`"lost"`/`"timeout"`).
- battle outcome (`"lost"`/`"battle_timeout"`) — the party cannot win; abort so
  the caller (a future Strategist) can react.

NOTE: `level_up` expects a Fighter. If `move_type_fn`/`predict` are missing, the
first `_execute_grind` returns `"encounter_started"`, which falls into the
`else` branch and is returned verbatim (documented, not specially guarded).

### Dispatch

`execute_order` gains the `mode == "level_up"` branch, checked alongside the
existing `heal`/`grind` branches (before `DESTINATIONS.get`, since level_up
ignores `destination` — pure intention like heal/grind).

## Testing

All pure, no ROM.

- `tests/test_heal_detector.py` +3: `party_needs_heal` — KO triggers, low total
  triggers, full party does not; empty party is False.
- `tests/test_world_reader.py` +1: `party_levels()` passthrough returns the RAM
  reader's list.
- `tests/test_orders.py` +N via a scripted `FarmWorld` fake (plays BOTH emulator
  and reader; snapshots at route_101's DESTINATIONS cell so `travel_to` arrives
  immediately; scripts party levels rising after each won battle and HP dropping
  then refilling on heal):
  - target reached after k battles → `"leveled_up"`,
  - already at target → `"leveled_up"` with zero battles fought,
  - HP drops low → detours to heal, then resumes and finishes → `"leveled_up"`,
  - a member KO'd → heals even though totals were fine,
  - heal needed but no healing spot known → `"no_healing_spot_known"`,
  - budget exhausted before target → `"grind_exhausted"`,
  - a battle lost → `"lost"` (loop aborts).

No ROM smoke: the loop is a pure composition of `_execute_grind`/`_execute_heal`,
both already covered; an end-to-end ROM run needs a savestate standing near
grass with a healable party and a deterministic encounter chain, which we do not
have.

## Non-goals

Building a strong team is deliberately excluded and left for later:
- **capturing** new Pokémon (needs the `capture` combat directive; the Fighter
  only knows how to win),
- **PC box management** (deposit/withdraw, which 6 to carry),
- **type coverage / evolutions** reasoning.

Deciding *which* Pokémon to raise and *when* to capture is the **Strategist's**
job (Strategist v2, world-aware) with a capture-capable Fighter — a separate
subsystem, its own spec → plan → implementation cycle. This loop raises whatever
party it is given.

Also out of scope: nearest-spot selection, carrying `target_level`/thresholds on
the `Order` dataclass, branching on the `combat` directive, active search for an
unknown grass/healing spot.
