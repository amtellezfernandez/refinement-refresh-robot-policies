"""Thin integration with the existing native paper runner. No native imports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Mapping

import benchstudio as bs


@dataclass(frozen=True)
class NativeCampaignResult:
    """Separate native analyses; never one generic trial or pooled model effect."""
    analyses: Mapping[str, dict]
    analysis_sha256: Mapping[str, str]

    def summary(self):
        return "\n".join(f"{model}: {report['verdict']}" for model, report in self.analyses.items())


@dataclass(frozen=True)
class NativeRecipe:
    repository: Path
    revision: str
    smolvla_python: str
    pi05_client_python: str
    pi05_server_python: str

    def _verify_checkout(self):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repository,
                                text=True, capture_output=True, check=True)
        if result.stdout.strip() != self.revision:
            raise ValueError("paper checkout differs from the requested recipe revision")
        # HEAD alone does not identify the code that Python will execute.
        # Inspect staged, unstaged and untracked Python/configuration changes;
        # retained run data and evidence JSON are intentionally outside this guard.
        paths = ("*.py", "pyproject.toml", "integrations/benchstudio/pyproject.toml")
        changed = subprocess.run(
            ["git", "diff", "--name-only", self.revision, "--", *paths],
            cwd=self.repository, text=True, capture_output=True, check=True)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
            cwd=self.repository, text=True, capture_output=True, check=True)
        if changed.stdout.strip() or untracked.stdout.strip():
            raise ValueError("paper runtime sources differ from the requested recipe revision; "
                             "use a clean pinned checkout")
        script = self.repository / "experiments/run_r22r25.py"
        if not script.is_file() or "--pi05-server-python" not in script.read_text("utf-8"):
            raise ValueError("integrate the reviewed three-interpreter frontend (paper PR #14) first")

    def check(self):
        """Offline inventory only. No native qualification or identity generation."""
        self._verify_checkout()
        return bs.RefinementWorkspace(self.repository).prepare()

    def run(self):
        # The helper reads retained evidence; the existing native runner independently
        # checks releases, interpreter roles, manifest hashes and qualification again.
        report = self.check()
        if not report.technical_ready or not report.execution_authorized:
            raise RuntimeError("native campaign is not ready/authorized:\n" + report.summary())
        workspace = self.repository / "experiments/runs/cross_policy_confirmation_v2"
        logs_root = workspace / "operator-logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        logs = Path(tempfile.mkdtemp(prefix="benchstudio-", dir=logs_root))
        argv = (sys.executable, str(self.repository / "experiments/run_r22r25.py"),
                "--smolvla-python", self.smolvla_python,
                "--pi05-client-python", self.pi05_client_python,
                "--pi05-server-python", self.pi05_server_python, "--execute")
        bs.ProcessCommand("refinement-refresh", argv, self.repository).run(logs)
        analyses, digests = {}, {}
        for model in ("smolvla", "pi05"):
            path = workspace / model / "analysis.json"
            raw = path.read_bytes()
            analysis = json.loads(raw)
            if analysis.get("model") != model or not isinstance(analysis.get("verdict"), str):
                raise ValueError("invalid native analysis for " + model)
            analyses[model] = analysis
            digests[model] = hashlib.sha256(raw).hexdigest()
        return NativeCampaignResult(MappingProxyType(analyses), MappingProxyType(digests))

    def __call__(self):
        return self.run()


def create_pipeline(*, repository, revision, smolvla_python, pi05_client_python, pi05_server_python):
    """Configure the existing two-model native recipe. Construction is inert."""
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be the full pinned paper checkout commit")
    for interpreter in (smolvla_python, pi05_client_python, pi05_server_python):
        if not isinstance(interpreter, str) or not interpreter.strip():
            raise ValueError("each native interpreter must be explicitly selected")
    return NativeRecipe(Path(repository).resolve(), revision,
                        smolvla_python, pi05_client_python, pi05_server_python)


class RecipeFactory:
    def __call__(self, **options):
        return create_pipeline(**options)

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--repository", required=True)
        parser.add_argument("--revision", required=True)
        parser.add_argument("--smolvla-python", required=True)
        parser.add_argument("--pi05-client-python", required=True)
        parser.add_argument("--pi05-server-python", required=True)

    @staticmethod
    def options_from_args(args):
        return {name: getattr(args, name) for name in (
            "repository", "revision", "smolvla_python", "pi05_client_python", "pi05_server_python")}


recipe = RecipeFactory()
