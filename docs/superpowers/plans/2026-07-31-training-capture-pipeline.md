# Training Capture Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrumenter l'entraînement Explorer pour qu'au prochain `train.py` la data video-ready (positions, reward, jalons, frames échantillonnées, clips) sorte sur disque toute seule, sans jamais faire planter le RL.

**Architecture:** Un seul `RecorderCallback` SB3 dans le processus principal lit `infos`/`rewards` (gratuit, déjà transférés) pour le numérique et appelle `env_method("render")` pour les frames échantillonnées + clips post-roll sur jalons. L'env expose `pos`/`step` dans son `info`. `train.py` câble le callback + rend chaque run auto-contenu sous `captures/<run_id>/`.

**Tech Stack:** Python 3.12, stable-baselines3 (PPO, `BaseCallback`, `CallbackList`), Pillow, numpy, csv/json stdlib, pytest (fakes sans ROM, smoke ROM gated).

---

## Contexte codebase (à lire avant de commencer)

- `env/pokemon_env.py` : `PokemonEmeraldEnv`. `step()` → `_info()` renvoie déjà `visited_tiles`, `badges`, `map=(map_group,map_num)`, `milestones`. `render()` → `emulator.screenshot()` (RGB `(160,240,3)` uint8). `PlayerState` (dans `env/game_state.py`) a `x`, `y`, `map_group`, `map_num`.
- `agent/train.py` : PPO `CnnPolicy`, `SubprocVecEnv` de N `Monitor(PokemonEmeraldEnv(...))`, `CheckpointCallback(save_path="checkpoints", name_prefix="ppo_emerald")`, `tensorboard_log="runs"`.
- `tests/conftest.py` : `FakeEmulator` (joueur mobile, `screenshot()` déterministe `(160,240,3)`), marqueur `requires_rom` (skip si `POKEMON_EMERALD_ROM` absent), fixture `rom_path`.
- `tests/test_env.py` : tests de `PokemonEmeraldEnv` sans ROM (via `FakeEmulator`).

**Commande de test (depuis le worktree)** — le venv et la ROM vivent dans le repo principal :

```bash
POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba \
PYTHONPATH=/Users/_eloi/Projets/Emu-training-capture \
/Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q
```

Pour un fichier précis, ajouter son chemin. Lint : `/Users/_eloi/Projets/Emu/.venv/bin/ruff check .`

## Structure des fichiers

- **Créer** `env/capture/__init__.py` — package vide.
- **Créer** `env/capture/recorder.py` — `RecorderCallback` (toute la logique de capture).
- **Modifier** `env/pokemon_env.py` — ajouter `pos` et `step` à `_info()`.
- **Modifier** `agent/train.py` — flags + câblage callback + run auto-contenu + `tb_log_name`.
- **Créer** `tests/test_recorder.py` — unitaires sans ROM (fake vec-env).
- **Créer** `tests/test_recorder_rom.py` — smoke ROM gated.
- **Modifier** `tests/test_env.py` — assertions `pos`/`step` dans `info`.

Contrat de données figé (voir spec) — `captures/<run_id>/` : `run.json`, `steps.csv` (`t,env,map_g,map_n,x,y,reward,visited_tiles`), `milestones.csv` (`t,env,milestone,wall_time`), `frames/envK/<t>.jpg`, `clips/<milestone>_<t0>/envK/<seq>.jpg`, `checkpoints/`.

---

## Task 1 : Exposer `pos` et `step` dans l'info de l'env

**Files:**
- Modify: `env/pokemon_env.py` (méthode `_info`, ~lignes 99-105)
- Test: `tests/test_env.py`

- [ ] **Step 1: Écrire le test qui échoue**

Ajouter à `tests/test_env.py` :

```python
def test_info_exposes_pos_and_step():
    from tests.conftest import FakeEmulator
    from env.pokemon_env import PokemonEmeraldEnv

    env = PokemonEmeraldEnv(FakeEmulator(), initial_state=b"x", max_steps=10)
    _, info = env.reset()
    assert info["pos"] == (5, 5)  # FakeEmulator starts at (5, 5)
    assert info["step"] == 0

    # action index for "right" moves x from 5 -> 6
    right = PokemonEmeraldEnv.ACTIONS.index("right")
    _, _, _, _, info = env.step(right)
    assert info["pos"] == (6, 5)
    assert info["step"] == 1
```

- [ ] **Step 2: Lancer le test → échec**

Run: `... -m pytest tests/test_env.py::test_info_exposes_pos_and_step -q`
Expected: FAIL (`KeyError: 'pos'`)

- [ ] **Step 3: Implémenter**

Dans `env/pokemon_env.py`, remplacer le corps de `_info` :

```python
    def _info(self, state: PlayerState | None) -> dict[str, Any]:
        return {
            "visited_tiles": self._tracker.visited_count,
            "badges": state.badges if state else 0,
            "map": (state.map_group, state.map_num) if state else None,
            "pos": (state.x, state.y) if state else None,
            "step": self._step_count,
            "milestones": sorted(self._milestones.fired),
        }
```

- [ ] **Step 4: Lancer le test → succès**

Run: `... -m pytest tests/test_env.py::test_info_exposes_pos_and_step -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add env/pokemon_env.py tests/test_env.py
git commit -m "feat: expose pos and step in env info for capture"
```

---

## Task 2 : `RecorderCallback` — squelette, `run.json`, `steps.csv`

**Files:**
- Create: `env/capture/__init__.py`
- Create: `env/capture/recorder.py`
- Test: `tests/test_recorder.py`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/test_recorder.py` :

```python
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
```

- [ ] **Step 2: Lancer → échec**

Run: `... -m pytest tests/test_recorder.py -q`
Expected: FAIL (`ModuleNotFoundError: env.capture`)

- [ ] **Step 3: Implémenter**

Créer `env/capture/__init__.py` (vide).

Créer `env/capture/recorder.py` :

```python
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
```

- [ ] **Step 4: Lancer → succès**

Run: `... -m pytest tests/test_recorder.py -q`
Expected: PASS (3 tests de cette task ; les tests des tasks 3-5 ne sont pas encore écrits)

- [ ] **Step 5: Commit**

```bash
git add env/capture/__init__.py env/capture/recorder.py tests/test_recorder.py
git commit -m "feat: RecorderCallback skeleton, run.json + steps.csv"
```

---

## Task 3 : Jalons → `milestones.csv` + armement de clip

**Files:**
- Modify: `tests/test_recorder.py` (le code de `_record_milestones` est déjà écrit en Task 2 ; cette task ajoute ses tests)

- [ ] **Step 1: Écrire les tests qui échouent... puis vérifier qu'ils passent**

Note : `_record_milestones` a été implémenté en Task 2. Ces tests verrouillent son comportement. Ajouter à `tests/test_recorder.py` :

```python
def test_new_milestone_writes_row_and_arms_clip(tmp_path):
    cb = _make_cb(tmp_path, n_envs=2, clip_len=5)
    _step(cb, t=100, infos=[_info(milestones=("meet_rival",)), _info()], rewards=[1.0, 0.0])
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
    rows = list(csv.reader((tmp_path / "run1" / "milestones.csv").open()))
    assert len(rows) == 2  # header + exactly one milestone row
```

- [ ] **Step 2: Lancer → succès (comportement déjà implémenté en Task 2)**

Run: `... -m pytest tests/test_recorder.py -k milestone -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_recorder.py
git commit -m "test: milestone logging + clip arming"
```

---

## Task 4 : Frames échantillonnées, clips post-roll, cap disque

**Files:**
- Modify: `tests/test_recorder.py` (le code `_maybe_sample_frames`/`_record_clips`/`_save_frame` est déjà écrit en Task 2 ; cette task ajoute ses tests)

- [ ] **Step 1: Écrire les tests**

Ajouter à `tests/test_recorder.py` :

```python
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
```

- [ ] **Step 2: Lancer → succès**

Run: `... -m pytest tests/test_recorder.py -k "frame or clip or cap" -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_recorder.py
git commit -m "test: frame sampling, post-roll clips, disk cap"
```

---

## Task 5 : Fail-safe (la capture ne crashe jamais l'entraînement)

**Files:**
- Modify: `tests/test_recorder.py`

- [ ] **Step 1: Écrire les tests**

Ajouter à `tests/test_recorder.py` :

```python
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
```

- [ ] **Step 2: Lancer → succès**

Run: `... -m pytest tests/test_recorder.py -k "error or threshold" -q`
Expected: PASS

- [ ] **Step 3: Lancer toute la suite recorder + lint**

Run: `... -m pytest tests/test_recorder.py -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check env/capture/`
Expected: PASS, 0 erreur ruff

- [ ] **Step 4: Commit**

```bash
git add tests/test_recorder.py
git commit -m "test: capture fail-safe never crashes training"
```

---

## Task 6 : Câbler `train.py` + smoke ROM

**Files:**
- Modify: `agent/train.py`
- Create: `tests/test_recorder_rom.py`

- [ ] **Step 1: Écrire le smoke ROM qui échoue**

Créer `tests/test_recorder_rom.py` :

```python
from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import requires_rom


@requires_rom
def test_capture_smoke_produces_run_dir(tmp_path, monkeypatch):
    """Tiny real train with capture ON writes a non-empty run dir."""
    import agent.train as train

    monkeypatch.chdir(tmp_path)
    (tmp_path / "states").mkdir()
    # reuse the real initial savestate from the main repo
    src = Path.home() / "Projets" / "Emu" / "states" / "initial.state"
    (tmp_path / "states" / "initial.state").write_bytes(src.read_bytes())

    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--envs", "1", "--timesteps", "512", "--max-steps", "64",
         "--run-id", "smoke", "--capture-every", "50"],
    )
    assert train.main() == 0

    run = tmp_path / "captures" / "smoke"
    assert (run / "steps.csv").read_text().count("\n") > 1        # header + >=1 row
    data = json.loads((run / "run.json").read_text())
    assert data["run_id"] == "smoke" and "final_timestep" in data
    assert any((run / "frames").rglob("*.jpg"))                   # at least one frame
```

- [ ] **Step 2: Lancer → échec**

Run: `... -m pytest tests/test_recorder_rom.py -q`
Expected: FAIL (`train.main` ne connaît pas `--run-id`/`--capture-every` → `SystemExit 2`)

- [ ] **Step 3: Modifier `agent/train.py`**

Remplacer les imports du haut et la fonction `main` par :

```python
"""Train PPO on Pokémon Emerald. Requires POKEMON_EMERALD_ROM and states/initial.state."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from emulator.gba import GbaEmulator
from env.capture.recorder import RecorderCallback
from env.pokemon_env import PokemonEmeraldEnv

log = logging.getLogger("agent.train")

STATE_PATH = Path("states/initial.state")


def make_env(rom_path: str, initial_state: bytes, max_steps: int):
    def _init() -> Monitor:
        # Monitor records episode rewards/lengths so SB3 logs rollout/ep_rew_mean.
        env = PokemonEmeraldEnv(GbaEmulator(rom_path), initial_state, max_steps=max_steps)
        return Monitor(env)

    return _init


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint .zip to resume")
    parser.add_argument("--run-id", default=None, help="capture run id (default: timestamp)")
    parser.add_argument("--capture", dest="capture", action="store_true", default=True)
    parser.add_argument("--no-capture", dest="capture", action="store_false")
    parser.add_argument("--capture-every", type=int, default=200)
    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--max-frame-gb", type=float, default=20.0)
    args = parser.parse_args()

    rom = os.environ.get("POKEMON_EMERALD_ROM")
    if not rom:
        log.error("Set POKEMON_EMERALD_ROM")
        return 1
    if not STATE_PATH.is_file():
        log.error("Missing %s — create it with tools/play_interactive.py", STATE_PATH)
        return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("captures") / run_id

    initial_state = STATE_PATH.read_bytes()
    vec = SubprocVecEnv(
        [make_env(rom, initial_state, args.max_steps) for _ in range(args.envs)]
    )
    device = pick_device()
    log.info("Training on device=%s with %d envs (run=%s)", device, args.envs, run_id)

    if args.resume:
        model = PPO.load(args.resume, env=vec, device=device)
    else:
        model = PPO(
            "CnnPolicy",
            vec,
            n_steps=512,
            batch_size=512,
            ent_coef=0.01,
            learning_rate=3e-4,
            device=device,
            verbose=1,
            tensorboard_log="runs",
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_emerald",
        )
    ]
    if args.capture:
        meta = {
            "argv": sys.argv,
            "total_timesteps": args.timesteps,
            "rom": rom,
            "initial_state": str(STATE_PATH),
            "max_steps": args.max_steps,
        }
        callbacks.append(
            RecorderCallback(
                run_dir=run_dir,
                capture_every=args.capture_every,
                clip_len=args.clip_len,
                max_frame_gb=args.max_frame_gb,
                meta=meta,
            )
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=run_id,
        reset_num_timesteps=False,
    )
    model.save(str(run_dir / "checkpoints" / "ppo_emerald_final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Lancer le smoke ROM → succès**

Run: `... -m pytest tests/test_recorder_rom.py -q`
Expected: PASS (skip proprement si `POKEMON_EMERALD_ROM` absent)

- [ ] **Step 5: Vérifier la non-régression du smoke train existant + lint**

Run: `... -m pytest tests/test_train_smoke.py -q && /Users/_eloi/Projets/Emu/.venv/bin/ruff check .`
Expected: PASS, 0 erreur ruff.
Note : si `tests/test_train_smoke.py` asserte l'ancien `save_path="checkpoints"`, ajuster l'assertion vers `captures/<run_id>/checkpoints/` (les checkpoints sont désormais sous le run).

- [ ] **Step 6: Suite complète**

Run: `POKEMON_EMERALD_ROM=$HOME/Projets/Emu/roms/pokemon_emerald_fr.gba PYTHONPATH=/Users/_eloi/Projets/Emu-training-capture /Users/_eloi/Projets/Emu/.venv/bin/python -m pytest -q`
Expected: tous verts (les tests recorder + les tests existants).

- [ ] **Step 7: Commit**

```bash
git add agent/train.py tests/test_recorder_rom.py
git commit -m "feat: wire RecorderCallback into train.py, self-contained runs"
```

---

## Self-review (couverture de la spec)

- Enrichir `info` (pos/step) → **Task 1** ✓
- `RecorderCallback` : `run.json`, `steps.csv` → **Task 2** ✓
- `milestones.csv` + armement clip → **Task 3** ✓
- Frames échantillonnées + clips post-roll + cap disque → **Task 4** ✓
- Fail-safe (jamais de crash, seuil d'erreurs) → **Task 5** ✓
- Câblage `train.py` (flags, `CallbackList`, checkpoints sous le run, `tb_log_name`) → **Task 6** ✓
- Contrat de données (`captures/<run_id>/…`) → produit par Tasks 2-4-6, vérifié par le smoke ROM (Task 6) ✓
- Tests sans ROM + smoke ROM gated → Tasks 2-5 (sans ROM) + Task 6 (gated) ✓

Non-goals respectés : aucun outil de rendu, pas de fog-of-war WallMap, pas de pré-roll, pas d'encodage mp4, `train_fighter`/`train_strategist` non touchés.
```
