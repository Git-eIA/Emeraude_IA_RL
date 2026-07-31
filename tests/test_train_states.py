from __future__ import annotations

import pytest

from agent.train import load_initial_states


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
