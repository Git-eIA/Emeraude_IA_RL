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
