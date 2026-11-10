import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import benchstudio as bs
from benchstudio.processes import ProcessCommand, ProcessFailure
from benchstudio.refinement import RefinementWorkspace, RefinementPreparation
from refinement_refresh_benchstudio import create_pipeline


def recipe(tmp_path, script=None):
    root = tmp_path / "paper checkout"
    root.mkdir()
    target = root / "experiments/run_r22r25.py"
    target.parent.mkdir()
    target.write_text(script or '# --pi05-server-python\nraise SystemExit(7)')
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=synthetic", "-c",
                    "user.email=synthetic@example.test", "commit", "-qm", "synthetic"], check=True)
    rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return create_pipeline(repository=root, revision=rev, smolvla_python="/smol/python",
                           pi05_client_python="/client/python", pi05_server_python="/server/python")


def preparation(technical=False, authorized=False):
    return RefinementPreparation({model: {"issues": [] if technical else [{
        "message": "native evidence missing", "next_action": "retain evidence"}],
        "offline_inputs_complete": technical, "execution_authorized": authorized,
        "commands": {}} for model in ("smolvla", "pi05")})


@pytest.mark.parametrize("technical,authorized", [(False, False), (True, False), (False, True)])
def test_readiness_and_permission_are_independent_and_never_launch(tmp_path, technical, authorized):
    pipeline = recipe(tmp_path)
    with patch.object(RefinementWorkspace, "prepare", return_value=preparation(technical, authorized)), \
            patch.object(ProcessCommand, "execute") as run:
        with pytest.raises(RuntimeError, match="not ready/authorized"):
            pipeline.run()
        run.assert_not_called()


def test_discovery_construction_and_check_do_not_execute_campaign(tmp_path, monkeypatch):
    backend = recipe(tmp_path)
    monkeypatch.setattr("benchstudio.backend_loading.metadata.entry_points", lambda **kw: [
        SimpleNamespace(load=lambda: lambda **kw: backend)])
    pipeline = bs.pipeline("refinement-refresh")
    with patch.object(RefinementWorkspace, "prepare", return_value=preparation()), \
            patch.object(ProcessCommand, "execute") as run:
        assert not pipeline.check().configuration_valid
        run.assert_not_called()


def test_single_frontend_routes_three_interpreters_and_returns_separate_native_analyses(tmp_path):
    script = '''# --pi05-server-python
import json, sys
from pathlib import Path
Path("received.json").write_text(json.dumps(sys.argv[1:]))
for model in ("smolvla", "pi05"):
    folder = Path("experiments/runs/cross_policy_confirmation_v2") / model
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "analysis.json").write_text(json.dumps({"model": model, "verdict": "SYNTHETIC_ONLY"}))
'''
    pipeline = recipe(tmp_path, script)
    with patch.object(RefinementWorkspace, "prepare", return_value=preparation(True, True)):
        result = pipeline.run()
    args = json.loads((pipeline.repository / "received.json").read_text())
    assert args == ["--smolvla-python", "/smol/python", "--pi05-client-python", "/client/python",
                    "--pi05-server-python", "/server/python", "--execute"]
    assert set(result.analyses) == {"smolvla", "pi05"}
    assert all(len(value) == 64 for value in result.analysis_sha256.values())
    assert "smolvla: SYNTHETIC_ONLY" in result.summary()


def test_checkout_mismatch_refuses_before_preparation(tmp_path):
    pipeline = recipe(tmp_path)
    object.__setattr__(pipeline, "revision", "0" * 40)
    with patch.object(RefinementWorkspace, "prepare") as prepare:
        with pytest.raises(ValueError, match="revision"):
            pipeline.check()
        prepare.assert_not_called()


def test_existing_runner_failure_propagates_without_fake_results(tmp_path):
    pipeline = recipe(tmp_path)
    with patch.object(RefinementWorkspace, "prepare", return_value=preparation(True, True)):
        with pytest.raises(ProcessFailure, match="exited 7"):
            pipeline.run()
    assert not (pipeline.repository / "experiments/runs/cross_policy_confirmation_v2/pi05/analysis.json").exists()


def test_cli_preparation_uses_same_installed_recipe_and_never_launches(tmp_path, monkeypatch):
    from benchstudio.commands import main
    from refinement_refresh_benchstudio import recipe as factory
    pipeline = recipe(tmp_path)
    monkeypatch.setattr("benchstudio.backend_loading.metadata.entry_points", lambda **kw: [
        SimpleNamespace(load=lambda: factory)])
    args = ["evaluate", "--recipe", "refinement-refresh", "--repository", str(pipeline.repository),
            "--revision", pipeline.revision, "--smolvla-python", "/smol/python",
            "--pi05-client-python", "/client/python", "--pi05-server-python", "/server/python", "--check"]
    with patch.object(RefinementWorkspace, "prepare", return_value=preparation()), \
            patch.object(ProcessCommand, "execute") as run:
        assert main(args) == 1
        run.assert_not_called()


@pytest.mark.parametrize("kind", ["unstaged", "staged", "untracked", "deleted"])
def test_runtime_drift_refuses_before_preparation_and_launch(tmp_path, kind):
    pipeline = recipe(tmp_path)
    source = pipeline.repository / "experiments/run_r22r25.py"
    if kind == "deleted":
        source.unlink()
    elif kind == "untracked":
        (source.parent / "shadow.py").write_text("raise RuntimeError('unreviewed')")
    else:
        source.write_text(source.read_text() + "\n# local change")
        if kind == "staged":
            subprocess.run(["git", "-C", str(pipeline.repository), "add", "."], check=True)
    with patch.object(RefinementWorkspace, "prepare") as prepare, \
            patch.object(ProcessCommand, "execute") as run:
        with pytest.raises(ValueError, match="runtime sources differ"):
            pipeline.run()
        prepare.assert_not_called()
        run.assert_not_called()


def test_retained_json_and_documentation_do_not_invalidate_runtime_pin(tmp_path):
    pipeline = recipe(tmp_path)
    (pipeline.repository / "retained.json").write_text('{"fixture": true}')
    (pipeline.repository / "notes.md").write_text("operator notes")
    with patch.object(RefinementWorkspace, "prepare", return_value=preparation()) as prepare:
        assert not pipeline.check().configuration_valid
        prepare.assert_called_once()

