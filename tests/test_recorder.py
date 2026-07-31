from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from env.capture.recorder import RecorderCallback


class FakeVecEnv:
    """Minimal VecEnv double: num_envs + env_method('render')."""

    def __init__(self, num_envs: int, frame_shape=(160, 240, 3)) -> None:
        self.num_envs = num_envs
        self._frame_shape = frame_shape
        self.render_calls = 0

    def env_method(self, method_name, *args, indices=None):
        assert method_name == "render"
        idxs = range(self.num_envs) if indices is None else indices
        self.render_calls += 1
        return [np.full(self._frame_shape, 7, dtype=np.uint8) for _ in idxs]


def _info(pos=(5, 5), map=(0, 10), milestones=(), visited=1):
    return {"pos": pos, "map": map, "milestones": list(milestones), "visited_tiles": visited}


def _make_cb(tmp_path: Path, n_envs=2, **kw) -> RecorderCallback:
    cb = RecorderCallback(run_dir=tmp_path / "run1", meta={"argv": ["train.py"]}, **kw)
    cb.training_env = FakeVecEnv(n_envs)
    cb._on_training_start()
    return cb


def _step(cb, t, infos, rewards):
    cb.num_timesteps = t
    cb.locals = {"infos": infos, "rewards": rewards}
    return cb._on_step()


def test_run_json_written_on_start(tmp_path):
    cb = _make_cb(tmp_path)
    data = json.loads((tmp_path / "run1" / "run.json").read_text())
    assert data["schema_version"] == 1
    assert data["run_id"] == "run1"
    assert data["n_envs"] == 2
    assert data["argv"] == ["train.py"]
    assert "git_commit" in data
    assert "start_wall_time" in data


def test_steps_csv_one_row_per_env_per_step(tmp_path):
    cb = _make_cb(tmp_path)
    _step(cb, t=2, infos=[_info(pos=(1, 2)), _info(pos=(3, 4))], rewards=[0.5, -0.5])
    cb._on_training_end()
    rows = list(csv.reader((tmp_path / "run1" / "steps.csv").open()))
    assert rows[0] == ["t", "env", "map_g", "map_n", "x", "y", "reward", "visited_tiles"]
    assert rows[1] == ["2", "0", "0", "10", "1", "2", "0.5", "1"]
    assert rows[2] == ["2", "1", "0", "10", "3", "4", "-0.5", "1"]


def test_run_json_finalised_on_end(tmp_path):
    cb = _make_cb(tmp_path)
    cb.num_timesteps = 8
    cb._on_training_end()
    data = json.loads((tmp_path / "run1" / "run.json").read_text())
    assert data["final_timestep"] == 8
    assert "end_wall_time" in data


# ---------------------------------------------------------------------------
# Batch A (Task 3) — milestone logging + clip arming
# ---------------------------------------------------------------------------


def test_new_milestone_writes_row_and_arms_clip(tmp_path):
    cb = _make_cb(tmp_path, n_envs=2, clip_len=5)
    _step(cb, t=100, infos=[_info(milestones=("meet_rival",)), _info()], rewards=[1.0, 0.0])
    cb._milestones_fh.flush()
    rows = list(csv.reader((tmp_path / "run1" / "milestones.csv").open()))
    assert rows[0] == ["t", "env", "milestone", "wall_time"]
    assert rows[1][0:3] == ["100", "0", "meet_rival"]
    assert cb._clip_tag[0] == "meet_rival_100"
    # arming set clip_remaining, then _record_clips consumed one frame this step
    assert cb._clip_remaining[0] == 4


def test_seen_milestone_not_duplicated(tmp_path):
    cb = _make_cb(tmp_path, n_envs=1, clip_len=1)
    _step(cb, t=10, infos=[_info(milestones=("meet_rival",))], rewards=[0.0])
    _step(cb, t=11, infos=[_info(milestones=("meet_rival",))], rewards=[0.0])
    cb._milestones_fh.flush()
    rows = list(csv.reader((tmp_path / "run1" / "milestones.csv").open()))
    assert len(rows) == 2  # header + exactly one milestone row

