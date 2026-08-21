from __future__ import annotations

from types import SimpleNamespace

from emulator import buttons
from env import campaign
from env.campaign import (
    CAMPAIGN,
    LAB,
    LITTLEROOT,
    Milestone,
    OLDALE,
    PHASE2_CAMPAIGN,
    ROUTE_101,
    ROUTE_103,
    _RETURN_DIRECTIONS,
    _drain_mom_event,
    _exit_lab,
    _finish_lab_cutscene,
    _verify_control,
    run_campaign,
    run_pokedex_return,
    run_shoes_leg,
)
from env.map_memory import MapMemory
from env.orders import Order


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


def test_milestone_story_target_defaults_none() -> None:
    assert Milestone("lab", 0).story_target is None


def test_story_milestone_emits_story_order_with_predicate() -> None:
    def target(_r: object) -> bool:
        return True

    reader = FakeReader([8])   # over-leveled: no level_up
    fn = RecordingOrderFn(["story_done"])
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


def test_reach_milestone_dispatches_reach_map(monkeypatch):
    calls = []

    def fake_reach(emu, rdr, mem, goal, directions, **kw):
        calls.append((goal, directions))
        return "arrived"

    monkeypatch.setattr(campaign, "reach_map", fake_reach)
    result = run_campaign(
        None, FakeReader([5]), MapMemory(),
        curriculum=(Milestone("lab", 0, reach=LAB),),
    )
    assert result == "campaign_complete"
    assert calls == [(LAB, _RETURN_DIRECTIONS)]


def test_reach_milestone_aborts_on_non_arrived(monkeypatch):
    monkeypatch.setattr(campaign, "reach_map", lambda *a, **k: "stall")
    result = run_campaign(
        None, FakeReader([5]), MapMemory(),
        curriculum=(Milestone("lab", 0, reach=LAB),),
    )
    assert result == "stall"


def test_return_directions_are_route101_down_littleroot_up():
    assert _RETURN_DIRECTIONS == {ROUTE_101: "down", LITTLEROOT: "up"}


def test_phase2_campaign_is_a_single_reach_home_milestone():
    # post_starter is before the route_103 rival, so the Pokédex cutscene is not armed
    # (proven: entering the lab fires nothing). The B2 deliverable is the descent itself,
    # so the campaign is exactly one reach-lab milestone — no story milestones.
    assert PHASE2_CAMPAIGN == (Milestone("lab", 0, reach=LAB),)


# ---------------------------------------------------------------------------
# A2: run_pokedex_return orchestration
# ---------------------------------------------------------------------------


class _Emu:
    """Fake emulator: run_pokedex_return calls .step for the settle frames."""

    def step(self, keys, frames):
        pass


def _wire_return(monkeypatch, *, hop, flora, descent, cutscene):
    """Stub the four legs of run_pokedex_return; record their calls in order."""
    calls = []
    monkeypatch.setattr(
        campaign, "hop_via_explore",
        lambda *a, **k: (calls.append(("hop", a[3], a[4])), hop)[1],
    )
    monkeypatch.setattr(
        campaign, "cross_scripted_npc",
        lambda *a, **k: (calls.append(("flora", a[3])), flora)[1],
    )
    monkeypatch.setattr(
        campaign, "reach_map",
        lambda *a, **k: (calls.append(("reach", a[3])), descent)[1],
    )
    monkeypatch.setattr(
        campaign, "_finish_lab_cutscene",
        lambda *a, **k: (calls.append(("cutscene",)), cutscene)[1],
    )
    return calls


def test_run_pokedex_return_delivers_on_the_happy_path(monkeypatch):
    calls = _wire_return(monkeypatch, hop="arrived", flora=True,
                         descent="arrived", cutscene=True)
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "pokedex_delivered"
    assert calls == [
        ("hop", ROUTE_103, OLDALE), ("flora", OLDALE),
        ("reach", LAB), ("cutscene",),
    ]


def test_run_pokedex_return_propagates_the_oldale_hop_outcome(monkeypatch):
    # hop_via_explore's own status (stall/no_portal/battle_lost/...) surfaces verbatim.
    _wire_return(monkeypatch, hop="stall", flora=True,
                 descent="arrived", cutscene=True)
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "stall"


def test_run_pokedex_return_stops_if_flora_never_crosses(monkeypatch):
    _wire_return(monkeypatch, hop="arrived", flora=False,
                 descent="arrived", cutscene=True)
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "flora_no_cross"


def test_run_pokedex_return_propagates_the_descent_outcome(monkeypatch):
    _wire_return(monkeypatch, hop="arrived", flora=True,
                 descent="stall", cutscene=True)
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "stall"


def test_run_pokedex_return_stops_if_the_pokedex_is_not_delivered(monkeypatch):
    _wire_return(monkeypatch, hop="arrived", flora=True,
                 descent="arrived", cutscene=False)
    assert run_pokedex_return(_Emu(), None, MapMemory()) == "pokedex_not_delivered"


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

    def has_poke_balls(self, min_qty: int = 1) -> bool:
        # Pin the production threshold: only 5 balls are ever delivered.
        return min_qty <= 5 and self._emu.a_presses >= self._presses_to_done

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
