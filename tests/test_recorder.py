from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

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
    _make_cb(tmp_path)
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


# ---------------------------------------------------------------------------
# Batch B (Task 4) — frame sampling, post-roll clips, disk cap
# ---------------------------------------------------------------------------


def _count(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.jpg")) if path.exists() else 0


def test_frames_sampled_at_cadence(tmp_path):
    cb = _make_cb(tmp_path, n_envs=2, capture_every=50)
    _step(cb, t=50, infos=[_info(), _info()], rewards=[0.0, 0.0])   # bucket 1 -> sample
    _step(cb, t=52, infos=[_info(), _info()], rewards=[0.0, 0.0])   # still bucket 1 -> no sample
    _step(cb, t=100, infos=[_info(), _info()], rewards=[0.0, 0.0])  # bucket 2 -> sample
    frames = tmp_path / "run1" / "frames"
    assert (frames / "env0" / "000000050.jpg").exists()
    assert (frames / "env0" / "000000100.jpg").exists()
    assert not (frames / "env0" / "000000052.jpg").exists()
    assert _count(frames / "env0") == 2 and _count(frames / "env1") == 2


def test_clip_writes_clip_len_frames(tmp_path):
    cb = _make_cb(tmp_path, n_envs=1, clip_len=3, capture_every=10_000)
    _step(cb, t=1, infos=[_info(milestones=("m",))], rewards=[0.0])  # arm + frame seq 0
    _step(cb, t=2, infos=[_info(milestones=("m",))], rewards=[0.0])  # seq 1
    _step(cb, t=3, infos=[_info(milestones=("m",))], rewards=[0.0])  # seq 2, remaining -> 0
    _step(cb, t=4, infos=[_info(milestones=("m",))], rewards=[0.0])  # nothing
    clip = tmp_path / "run1" / "clips" / "m_1" / "env0"
    assert sorted(p.name for p in clip.glob("*.jpg")) == ["0000.jpg", "0001.jpg", "0002.jpg"]


def test_disk_cap_stops_frames_keeps_numeric(tmp_path):
    cb = _make_cb(tmp_path, n_envs=1, capture_every=1, max_frame_gb=1e-9)  # ~1 byte cap
    _step(cb, t=1, infos=[_info(pos=(1, 1))], rewards=[0.0])  # writes 1 frame, trips cap
    _step(cb, t=2, infos=[_info(pos=(2, 2))], rewards=[0.0])  # frames disabled
    assert cb._disabled_frames is True
    assert _count(tmp_path / "run1" / "frames") == 1
    cb._on_training_end()
    rows = list(csv.reader((tmp_path / "run1" / "steps.csv").open()))
    assert len(rows) == 3  # header + 2 numeric rows (numeric kept)


# ---------------------------------------------------------------------------
# Batch C (Task 5) — fail-safe
# ---------------------------------------------------------------------------


class ExplodingVecEnv(FakeVecEnv):
    def env_method(self, method_name, *args, indices=None):
        raise RuntimeError("render boom")


def test_render_error_does_not_crash_and_keeps_numeric(tmp_path):
    cb = RecorderCallback(run_dir=tmp_path / "run1", capture_every=1)
    cb.training_env = ExplodingVecEnv(1)
    cb._on_training_start()
    result = _step(cb, t=1, infos=[_info(pos=(9, 9))], rewards=[0.0])
    assert result is True                 # training continues
    assert cb._errors == 1                # render failure noted
    cb._on_training_end()
    rows = list(csv.reader((tmp_path / "run1" / "steps.csv").open()))
    assert rows[1][4:6] == ["9", "9"]     # numeric written before the failing frame block


def test_disables_after_error_threshold(tmp_path):
    from env.capture.recorder import _MAX_ERRORS

    cb = RecorderCallback(run_dir=tmp_path / "run1", capture_every=1)
    cb.training_env = ExplodingVecEnv(1)
    cb._on_training_start()
    for t in range(1, _MAX_ERRORS + 1):
        _step(cb, t=t, infos=[_info()], rewards=[0.0])
    assert cb._disabled is True
    # once disabled, further steps are no-ops and still return True
    assert _step(cb, t=999, infos=[_info()], rewards=[0.0]) is True


# ---------------------------------------------------------------------------
# Fix A — global milestone dedup (first-time-ever per env, not per episode)
# ---------------------------------------------------------------------------


def test_milestone_not_refired_after_episode_reset(tmp_path):
    cb = _make_cb(tmp_path, n_envs=1, clip_len=5)
    _step(cb, t=10, infos=[_info(milestones=("meet_rival",))], rewards=[0.0])  # fires
    _step(cb, t=11, infos=[_info(milestones=())], rewards=[0.0])               # episode reset
    _step(cb, t=12, infos=[_info(milestones=("meet_rival",))], rewards=[0.0])  # same milestone again
    cb._milestones_fh.flush()
    rows = list(csv.reader((tmp_path / "run1" / "milestones.csv").open()))
    assert len(rows) == 2  # header + exactly one row despite firing twice across episodes


# ---------------------------------------------------------------------------
# Fix B — teardown is a no-op if init never completed
# ---------------------------------------------------------------------------


def test_failed_init_teardown_is_noop(tmp_path):
    cb = RecorderCallback(run_dir=tmp_path / "run1")
    cb.training_env = FakeVecEnv(1)
    # sabotage: point run_dir at a path whose parent is a file, so mkdir fails
    bad = tmp_path / "afile"
    bad.write_text("x")
    cb._run_dir = bad / "sub"
    cb._on_training_start()
    assert cb._disabled is True
    assert cb._started is False
    cb._on_training_end()  # must not raise
    assert not (bad / "sub" / "run.json").exists()


# ---------------------------------------------------------------------------
# Fix C — capture_every=0 rejected in __init__
# ---------------------------------------------------------------------------


def test_capture_every_zero_rejected(tmp_path):
    with pytest.raises(ValueError):
        RecorderCallback(run_dir=tmp_path / "run1", capture_every=0)
