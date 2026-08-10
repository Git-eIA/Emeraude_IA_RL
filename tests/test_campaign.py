from __future__ import annotations

from env.campaign import (
    CAMPAIGN,
    LAB,
    LITTLEROOT,
    Milestone,
    OLDALE,
    ROUTE_101,
    ROUTE_103,
    run_campaign,
    seed_return_portals,
)
from env.map_memory import MapMemory
from env.orders import Order
from env.route_planner import plan_route


def test_milestone_holds_destination_and_target_level():
    m = Milestone("route_101", 5)
    assert m.destination == "route_101"
    assert m.target_level == 5


def test_campaign_seed_is_a_tuple_of_milestones():
    assert isinstance(CAMPAIGN, tuple)
    assert all(isinstance(m, Milestone) for m in CAMPAIGN)
    assert CAMPAIGN[0].destination == "route_101"


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
        self.kwargs: list[dict] = []

    def __call__(self, order: Order, emulator, reader, memory, **kwargs):
        self.calls.append((order.mode, order.destination, kwargs.get("target_level")))
        self.kwargs.append(kwargs)
        return self._outcomes.pop(0)


def test_under_leveled_milestone_emits_level_up_then_advance():
    reader = FakeReader([3])  # mean level 3 < target 5
    fn = RecordingOrderFn(["leveled_up", "arrived"])
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [
        ("level_up", "route_101", 5),
        ("advance", "route_101", None),
    ]


def test_over_leveled_milestone_skips_level_up():
    reader = FakeReader([8])  # mean level 8 >= target 5
    fn = RecordingOrderFn(["arrived"])
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [("advance", "route_101", None)]


def test_multiple_milestones_run_in_order():
    reader = FakeReader([9])  # over-leveled for both -> advance-only each
    fn = RecordingOrderFn(["arrived", "arrived"])
    result = run_campaign(
        None, reader, None,
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
        None, reader, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "lost"
    assert fn.calls == [("level_up", "route_101", 5)]


def test_advance_failure_aborts_and_surfaces_outcome():
    reader = FakeReader([8])  # over-leveled -> straight to advance
    fn = RecordingOrderFn(["unknown_route"])
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn,
    )
    assert result == "unknown_route"
    assert fn.calls == [("advance", "route_101", None)]


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
        None, reader, None,
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
        None, reader, None,
        curriculum=(Milestone("route_103", 5, trainer=True),),
        order_fn=fn,
    )
    assert result == "lost"
    assert fn.calls == [
        ("advance", "route_103", None),
        ("battle_trainer", "route_103", None),
    ]


def test_advance_threads_the_fighter():
    # advance must cross grass toward route_103; without the Fighter, navigate_to
    # aborts on a wild battle. The advance Order carries move_type_fn/predict.
    reader = FakeReader([8])
    fn = RecordingOrderFn(["arrived"])
    move_type_fn, predict = object(), object()
    run_campaign(
        None, reader, None,
        curriculum=(Milestone("route_101", 5),),
        order_fn=fn, move_type_fn=move_type_fn, predict=predict,
    )
    advance_kwargs = fn.kwargs[0]
    assert advance_kwargs["move_type_fn"] is move_type_fn
    assert advance_kwargs["predict"] is predict


def test_non_trainer_milestone_does_not_battle():
    reader = FakeReader([8])
    fn = RecordingOrderFn(["arrived"])
    result = run_campaign(
        None, reader, None,
        curriculum=(Milestone("route_101", 5),),  # trainer defaults False
        order_fn=fn,
    )
    assert result == "campaign_complete"
    assert fn.calls == [("advance", "route_101", None)]


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
