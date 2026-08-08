from __future__ import annotations

import pytest

from agent.train import load_initial_states, select_milestones


def test_select_milestones_route103_is_the_north_push_chain():
    ms = select_milestones("route103")
    assert [m.name for m in ms] == ["reach_oldale", "reach_route_103", "beat_rival"]
    # beat_rival is terminal and gated on the env's rival_beaten latch.
    assert ms[-1].terminal
    assert ms[-1].env_condition is not None


def test_select_milestones_default_is_the_starter_chain():
    starter = select_milestones("starter")
    assert "starter_obtained" in {m.name for m in starter}
    assert select_milestones("route103") != starter


def test_load_returns_truck_plus_frontier(tmp_path):
    truck = tmp_path / "initial.state"
    truck.write_bytes(b"truck")
    frontier = tmp_path / "explorer"
    frontier.mkdir()
    (frontier / "reach_route_101.state").write_bytes(b"r101")
    (frontier / "meet_rival.state").write_bytes(b"rival")

    states = load_initial_states(truck, frontier)

    assert states[0] == b"truck"  # truck always first
    assert set(states) == {b"truck", b"r101", b"rival"}


def test_load_truck_only_when_no_frontier_dir(tmp_path):
    truck = tmp_path / "initial.state"
    truck.write_bytes(b"truck")
    states = load_initial_states(truck, tmp_path / "explorer")  # dir absent
    assert states == [b"truck"]


def test_load_raises_when_truck_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_initial_states(tmp_path / "initial.state", tmp_path / "explorer")
