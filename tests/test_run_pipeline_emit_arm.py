"""run_pipeline's emit step: call #3 runs framed (--render-chained) by
default, schema-only on request, and run.json records which. Subprocesses
are captured, never run."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_pipeline as rp  # noqa: E402


@pytest.mark.parametrize("extra, framed", [([], True),
                                          (["--emit-arm", "schema-only"], False)])
def test_emit_arm(tmp_path, monkeypatch, extra, framed):
    calls = []

    class Done:
        returncode = 0

    monkeypatch.setattr(rp.subprocess, "run", lambda args, env=None: calls.append(args) or Done())
    monkeypatch.setattr(rp, "ISSUED", [])
    monkeypatch.setattr(rp, "RUN", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--scene", str(ROOT / "scenes/insert_marker.json"),
        "--task-spec", str(ROOT / "tasks/insert_marker.json"),
        "--from", "emit", "--until", "emit", *extra])
    rp.main()
    (emit,) = [c for c in calls if any("emit_constraints.py" in a for a in c)]
    assert ("--render-chained" in emit) is framed
    if framed:
        assert emit[emit.index("--render-chained") + 1] == "outputs/frames/insert_marker_chained"
    (run_json,) = tmp_path.glob("outputs/runs/*/run.json")
    assert json.loads(run_json.read_text())["emit_arm"] == ("framed-chained" if framed else "schema-only")
