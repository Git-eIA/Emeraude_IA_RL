# Live Hierarchical Loop — design spec

**Date:** 2026-08-02
**Palier:** P4 étape 6

## Goal

Wire the Order interface into a running loop against the ROM for the first time. A
scripted Strategist reads the game state, picks an `Order`, calls `execute_order`,
and repeats over a hand-written curriculum of milestones. This makes everything
built so far (advance / heal / grind / level_up) load-bearing end-to-end.

Until now, nothing in production emits `Order`s: the full advance/heal/grind/level_up
machinery is built and tested, but no driver runs it in a loop. This palier adds
that driver.

## Non-goals

- Wrapping the trained PPO Strategist. Its observation needs `challenge_level`
  (which important battle is next + its level), which is NOT in RAM. Grounding it
  is "world-aware" work, deferred.
- Fighting gym leaders / trainer battles on arrival. `advance` is navigation-only
  in v1 (reach the place). Beating the leader there needs trainer-battle handling
  and a trainer-capable Fighter.
- Capture / min_loss directives (Fighter only knows how to win).
- Nearest-spot selection (still `spots[0]`, same as heal/grind).

## Architecture

New module `env/campaign.py`. One driver function, one data type, one seed table.
It composes `execute_order` — it adds no navigation, combat, or RAM logic of its
own. The only new logic is the sequencing: for each milestone, level up if
under-leveled, then advance.

### Data

```python
@dataclass(frozen=True)
class Milestone:
    destination: str    # a name in orders.DESTINATIONS
    target_level: int   # required mean party level before advancing

CAMPAIGN: tuple[Milestone, ...] = (
    Milestone("route_101", 5),
)
```

`Milestone` mirrors `Order` / `DESTINATIONS`: a hand-written table that means
something to the chef before any exploration. Seeded minimally; extended as more
named destinations are verified.

### Driver

```python
def run_campaign(
    emulator, reader, memory, wallmap,
    curriculum=CAMPAIGN, max_hops=20,
    move_type_fn=None, predict=None,
    heal_threshold=0.4, max_cycles=50,
    order_fn=execute_order,
) -> str:
    for m in curriculum:
        if not _reached(reader.party_levels(), m.target_level):
            r = order_fn(
                Order(m.destination, "level_up", "win"),
                emulator, reader, memory, wallmap,
                max_hops=max_hops, move_type_fn=move_type_fn, predict=predict,
                target_level=m.target_level, heal_threshold=heal_threshold,
                max_cycles=max_cycles,
            )
            if r != "leveled_up":
                return r
        r = order_fn(
            Order(m.destination, "advance", "win"),
            emulator, reader, memory, wallmap, max_hops=max_hops,
        )
        if r != "arrived":
            return r
    return "campaign_complete"
```

- **Bounded loop** (code-safety #2): the curriculum is a finite tuple.
- **`order_fn` injected** = `execute_order` in production, a fake in tests — the
  same dependency-injection pattern used for the Fighter (`move_type_fn` / `predict`
  injected into orders).
- **Per milestone:** under the required mean level → `level_up` to target (which
  grinds + heals itself); then `advance` to the destination.
- **Abort on any non-terminal outcome:** if `level_up` returns anything but
  `"leveled_up"`, or `advance` anything but `"arrived"`, surface it verbatim and
  stop. This lets a future Strategist react.

`_reached(levels, target)` is reused from `env.orders` (mean-level check) rather
than duplicated.

### Outcomes

- `"campaign_complete"` — every milestone advanced.
- Any non-terminal outcome from `level_up` (`"grind_exhausted"`, `"lost"`,
  `"battle_timeout"`, `"no_grass_spot_known"`, `"no_healing_spot_known"`,
  `"heal_failed"`, travel pass-throughs) surfaced verbatim.
- Any non-terminal outcome from `advance` (`"unknown_destination"`,
  `"unknown_route"`, `"unreachable"`, `"lost"`, `"timeout"`) surfaced verbatim.

## Testing

Pure, no ROM. Inject a fake `order_fn` that records each call as
`(mode, destination, target_level)` and returns scripted outcomes. A fake `reader`
supplies `party_levels()`. This tests the sequencing logic precisely without
rebuilding a full ROM world.

Cases:
1. Under-leveled milestone → emits `level_up` (with the target) then `advance`.
2. Over-leveled milestone → skips `level_up`, emits only `advance`.
3. Multiple milestones, all pass → `"campaign_complete"`, in order.
4. `level_up` returns `"lost"` → aborts with `"lost"`, no `advance` emitted.
5. `advance` returns `"unknown_route"` → aborts with `"unknown_route"`.

Real ROM smoke deferred: needs a savestate with a party on/near a known grass
cell and a deterministic path to route_101 — not available yet (same deferral as
grind/level_up).

## Files

- Create: `env/campaign.py`
- Create: `tests/test_campaign.py`
