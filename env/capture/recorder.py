"""SB3 callback that records video-ready training data. Fail-safe: never crashes training."""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

log = logging.getLogger("env.capture")

SCHEMA_VERSION = 1
_MAX_ERRORS = 20
_STEPS_HEADER = ["t", "env", "map_g", "map_n", "x", "y", "reward", "visited_tiles"]
_MILESTONES_HEADER = ["t", "env", "milestone", "wall_time"]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip()
    except Exception:  # git absent / not a repo — metadata is best-effort
        return ""


class RecorderCallback(BaseCallback):
    """Writes captures/<run_id>/ during training. All I/O is fail-safe."""

    def __init__(
        self,
        run_dir: Path,
        capture_every: int = 200,
        clip_len: int = 48,
        max_frame_gb: float = 20.0,
        frame_format: str = "jpg",
        meta: dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._run_dir = Path(run_dir)
        self._capture_every = capture_every
        self._clip_len = clip_len
        self._max_frame_bytes = int(max_frame_gb * 1_000_000_000)
        self._frame_format = frame_format
        self._meta = dict(meta or {})
        self._seen: list[set[str]] = []
        self._clip_remaining: list[int] = []
        self._clip_tag: list[str] = []
        self._frame_bytes = 0
        self._last_bucket = -1
        self._errors = 0
        self._disabled = False
        self._disabled_frames = False
        self._steps_fh = None
        self._steps_writer = None
        self._milestones_fh = None
        self._milestones_writer = None
        # NOTE: SB3 >=2.4 makes training_env a read-only property on BaseCallback;
        # we shadow it here so tests and _on_training_start can inject a VecEnv directly.
        self._training_env = None

    @property  # type: ignore[override]
    def training_env(self):
        if self._training_env is not None:
            return self._training_env
        return super().training_env  # type: ignore[misc]

    @training_env.setter
    def training_env(self, env) -> None:
        self._training_env = env

    # --- lifecycle -----------------------------------------------------------
    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._seen = [set() for _ in range(n)]
        self._clip_remaining = [0] * n
        self._clip_tag = [""] * n
        for sub in ("frames", "clips", "checkpoints"):
            (self._run_dir / sub).mkdir(parents=True, exist_ok=True)
        self._steps_fh = (self._run_dir / "steps.csv").open("w", newline="")
        self._steps_writer = csv.writer(self._steps_fh)
        self._steps_writer.writerow(_STEPS_HEADER)
        self._milestones_fh = (self._run_dir / "milestones.csv").open("w", newline="")
        self._milestones_writer = csv.writer(self._milestones_fh)
        self._milestones_writer.writerow(_MILESTONES_HEADER)
        self._write_run_json(start=True)

    def _on_step(self) -> bool:
        if self._disabled:
            return True
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        t = int(self.num_timesteps)
        self._guard(self._record_numeric, t, infos, rewards)
        self._guard(self._record_milestones, t, infos)
        self._guard(self._maybe_sample_frames, t)
        self._guard(self._record_clips)
        return True

    def _on_training_end(self) -> None:
        for fh in (self._steps_fh, self._milestones_fh):
            if fh is not None:
                fh.flush()
                fh.close()
        self._write_run_json(start=False)

    # --- guard ---------------------------------------------------------------
    def _guard(self, fn, *args) -> None:
        if self._disabled:
            return
        try:
            fn(*args)
        except Exception as exc:  # capture must never crash training
            self._errors += 1
            log.warning("capture error (%d/%d): %s", self._errors, _MAX_ERRORS, exc)
            if self._errors >= _MAX_ERRORS:
                self._disabled = True
                log.warning("capture disabled after repeated errors; training continues")

    # --- records -------------------------------------------------------------
    def _record_numeric(self, t: int, infos, rewards) -> None:
        for i, info in enumerate(infos):
            m = info.get("map")
            map_g, map_n = (m if m else ("", ""))
            pos = info.get("pos")
            x, y = (pos if pos else ("", ""))
            self._steps_writer.writerow(
                [t, i, map_g, map_n, x, y, float(rewards[i]), info.get("visited_tiles", "")]
            )

    def _record_milestones(self, t: int, infos) -> None:
        for i, info in enumerate(infos):
            current = set(info.get("milestones", []))
            for name in sorted(current - self._seen[i]):
                self._milestones_writer.writerow([t, i, name, time.time()])
                self._clip_remaining[i] = self._clip_len
                self._clip_tag[i] = f"{name}_{t}"
            self._seen[i] = current

    def _maybe_sample_frames(self, t: int) -> None:
        if self._disabled_frames:
            return
        bucket = t // self._capture_every
        if bucket == self._last_bucket:
            return
        self._last_bucket = bucket
        for i, frame in enumerate(self.training_env.env_method("render")):
            d = self._run_dir / "frames" / f"env{i}"
            d.mkdir(parents=True, exist_ok=True)
            self._save_frame(frame, d / f"{t:09d}.{self._frame_format}")

    def _record_clips(self) -> None:
        if self._disabled_frames:
            return
        for i, remaining in enumerate(self._clip_remaining):
            if remaining <= 0:
                continue
            frame = self.training_env.env_method("render", indices=[i])[0]
            d = self._run_dir / "clips" / self._clip_tag[i] / f"env{i}"
            d.mkdir(parents=True, exist_ok=True)
            seq = self._clip_len - remaining
            self._save_frame(frame, d / f"{seq:04d}.{self._frame_format}")
            self._clip_remaining[i] = remaining - 1

    def _save_frame(self, frame, path: Path) -> None:
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
        self._frame_bytes += path.stat().st_size
        if self._frame_bytes > self._max_frame_bytes:
            self._disabled_frames = True
            log.warning(
                "frame disk cap reached (%.1f GB); numeric logging continues",
                self._max_frame_bytes / 1e9,
            )

    def _write_run_json(self, start: bool) -> None:
        path = self._run_dir / "run.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        if start:
            data.update(self._meta)
            data.update(
                {
                    "run_id": self._run_dir.name,
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": _git_commit(),
                    "n_envs": self.training_env.num_envs,
                    "start_wall_time": time.time(),
                }
            )
        else:
            data.update({"end_wall_time": time.time(), "final_timestep": int(self.num_timesteps)})
        path.write_text(json.dumps(data, indent=2))
