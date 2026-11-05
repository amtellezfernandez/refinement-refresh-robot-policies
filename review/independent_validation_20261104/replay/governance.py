"""Fail-closed governance for the *archived* paper-evidence replay.

The frozen PROTOCOL.md remains unchanged. This implementation repairs enforcement,
not scientific decision rules. Synthetic fixtures used by Warren's prior tests are
not paper evidence and are not subject to the archived-data release.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_SHA256 = "ede3eb8570487b7e2eda8e47f39c6afc1d12e40bece590e8c4590d34341ea737"
BUNDLES_SHA256 = "99587b7e3ea594572f18ee3b933109eaf57dfa0ce4cc74deea83b3bc1b43f372"
ARCHIVE_SHA256 = "36c1d5f2dcaef85664ec7a1bfe7409c849b3be17831566e783d24a00ea4e8df0"
ASSET_MANIFEST_SHA256 = "ad609f416342dcd970eef0fa12d5ab93088a776454350d4c5f6a82b0a076a503"
SAMPLE_SELECTION_SHA256 = "3ffe84b0ef2d055a6c22667406c9d8eca370bf66ab7999c46e99a7114f73df56"
SAMPLE_SALT = "rrrp-replay-stage1-20261104|"
ENV_VERSIONS = {"mujoco": "3.8.1", "robosuite": "1.4.0", "hf_libero": "0.1.4",
                "bddl": "1.0.1", "numpy": "2.2.6", "torch": "2.7.1"}
EXECUTION_FILES = ("governance.py", "run_protocol.py", "replay.py", "libero_replay.py",
                   "extract_bundles.py")
AMENDMENT_FILES = ("PROTOCOL_AMENDMENT_01.md", "PROTOCOL_AMENDMENT_02.md")
APPROVER = "Alba María Téllez Fernández"
PRIMARY = {"h2_d2", "h2_d5"}
# Internal capability prevents accidentally invoking the direct replay CLI on archived
# evidence; this is NOT a security boundary against someone editing Python source.
_ORCHESTRATOR_CAPABILITY = object()


class ReplayRefused(ValueError):
    pass


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def is_real_archive(bundles):
    # Preserve synthetic fixture drills that deliberately relabel their provenance.
    # A real attempt must have the archive key and must not contain fixture provenance.
    return any(b.get("provenance") != "synthetic_fixture"
               and not (str(b.get("episode_key", "")).startswith("fixture.")
                        and b.get("fixture_source") is not None)
               for b in bundles)


FROZEN_CONDITIONS = ((1, 2), (1, 5), (1, 10), (2, 2), (2, 5), (5, 2), (5, 5))


def expected_all_keys():
    return {f"confirmation.{suite}.task{task:03d}.state{state:03d}.seed{seed}.h{h}_d{d}"
            for suite in ("libero_goal", "libero_object")
            for task in (5, 6, 7, 8) for state in range(20, 30)
            for seed in (3000, 3001) for h, d in FROZEN_CONDITIONS}


def expected_primary_keys():
    return {f"confirmation.{suite}.task{task:03d}.state{state:03d}.seed{seed}.h2_d{d}"
            for suite in ("libero_goal", "libero_object")
            for task in (5, 6, 7, 8) for state in range(20, 30)
            for seed in (3000, 3001) for d in (2, 5)}


def verify_population(bundles, sample):
    """Check population, pairing and exactly the frozen outcome-stratified sample.

    No replayed outcome enters this calculation; the stratification uses archived
    labels as it did when the original sample was frozen.
    """
    if len(bundles) != 1120 or not all(b.get("provenance") == "archived_paper_evidence" for b in bundles):
        raise ReplayRefused("archive population is not 1,120 recorded attempts")
    keys = [b.get("episode_key") for b in bundles]
    if any(not isinstance(k, str) for k in keys) or len(set(keys)) != 1120:
        raise ReplayRefused("duplicate/missing archive attempt identity")
    if set(keys) != expected_all_keys():
        raise ReplayRefused("archive identities do not exactly match the seven frozen settings")
    by_key = dict(zip(keys, bundles))
    primary = {k for k, b in by_key.items() if b.get("condition") in PRIMARY}
    if primary != expected_primary_keys():
        raise ReplayRefused("primary population is not the 320 frozen paired attempts")
    starts = {}
    for b in bundles:
        if b.get("suite") not in ("libero_goal", "libero_object") or b.get("task_id") not in (5, 6, 7, 8):
            raise ReplayRefused("archive includes a task outside the confirmation population")
        sid = b.get("reset_source", {}).get("initial_state_id")
        if sid not in range(20, 30):
            raise ReplayRefused("archive uses an unplanned initial-state ID")
        if b.get("reset_source", {}).get("kind") != "official":
            raise ReplayRefused("archive must use official, immutable init-state IDs")
        model_key = ".{}.".format(b["suite"])
        if (model_key not in b["episode_key"]
                or f".task{b['task_id']:03d}.state{sid:03d}." not in b["episode_key"]
                or not b["episode_key"].endswith("." + str(b.get("condition")))):
            raise ReplayRefused("archive row metadata disagrees with the planned identity")
        key = (b["suite"], b["task_id"], sid)
        start = b.get("recorded_initial_state", {}).get("sha256")
        if not isinstance(start, str) or len(start) != 64:
            raise ReplayRefused("missing initial-state hash")
        if key in starts and starts[key] != start:
            raise ReplayRefused("paired initial states differ within a task-state")
        starts[key] = start
    if len(starts) != 80:
        raise ReplayRefused("expected 80 distinct task-state resets")

    sample_keys = sample.get("stage1_sample")
    if (not isinstance(sample_keys, list) or len(sample_keys) != 57
            or len(set(sample_keys)) != 57 or not set(sample_keys).issubset(primary)):
        raise ReplayRefused("Stage 1 manifest is not 57 distinct primary attempts")
    if (sample.get("population"), sample.get("primary_frame"),
            sample.get("strata_nonempty"), sample.get("stage1_sample_size"),
            sample.get("salt"), sample.get("per_stratum")) != (
            1120, 320, 29, 57, SAMPLE_SALT, 2):
        raise ReplayRefused("Stage 1 manifest design metadata differs from the frozen design")
    if sample.get("stage1_sample_sha256") != SAMPLE_SELECTION_SHA256:
        raise ReplayRefused("frozen sample digest field differs")
    if sha_json(sorted(sample_keys)) != SAMPLE_SELECTION_SHA256:
        raise ReplayRefused("actual sample selection differs from the frozen digest")

    strata = defaultdict(list)
    for k in primary:
        b = by_key[k]
        status = "success" if b["label"]["success"] else "failure"
        strata[(b["suite"], b["task_id"], b["condition"], status)].append(k)
    if len(strata) != 29:
        raise ReplayRefused("unexpected number of nonempty strata")
    picked = []
    expected_table = []
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=lambda k: hashlib.sha256((SAMPLE_SALT + k).encode()).hexdigest())
        selected = ordered[:2]
        picked.extend(selected)
        expected_table.append({"stratum": list(stratum), "frame": len(strata[stratum]), "sampled": selected})
    if sample_keys != picked or sample.get("strata") != expected_table:
        raise ReplayRefused("Stage 1 manifest deviates from the frozen hash-stratified draw")
    return {"bundles": 1120, "resets": 80, "primary": 320,
            "sample": len(sample_keys), "sample_selection_sha256": SAMPLE_SELECTION_SHA256}


def _strict_release(release_path, protocol_path, bundles_path, sample_path, through, mode="frozen"):
    if through not in (0, 1, 2):
        raise ReplayRefused("requested --through must be one of 0, 1, 2")
    if mode not in ("frozen", "sample_first_57_v1"):
        raise ReplayRefused("unknown execution mode")
    if mode == "sample_first_57_v1" and through != 1:
        raise ReplayRefused("sample-first diagnostic is limited to 57 attempts (stage 1)")
    if not all((release_path, protocol_path, bundles_path, sample_path)):
        raise ReplayRefused("release, protocol, bundle and sample file paths are mandatory")
    try:
        release = json.loads(Path(release_path).read_text("utf-8"))
        sample = json.loads(Path(sample_path).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayRefused(f"cannot read approved inputs: {exc}") from exc
    if not isinstance(release, dict) or not isinstance(sample, dict):
        raise ReplayRefused("release and sample must be JSON objects")
    if mode == "sample_first_57_v1":
        alternate = ROOT / "protocol" / "SAMPLE_FIRST_57_PROTOCOL.md"
        if Path(protocol_path).resolve() != alternate.resolve():
            raise ReplayRefused("sample-first run requires the separate, versioned protocol")
        expected_protocol = sha_file(alternate)
        if release.get("execution_mode") != mode:
            raise ReplayRefused("release does not approve sample-first execution mode")
    else:
        expected_protocol = PROTOCOL_SHA256
        if release.get("execution_mode") not in (None, "frozen"):
            raise ReplayRefused("frozen release has incompatible execution mode")
    if sha_file(protocol_path) != expected_protocol or release.get("protocol_sha256") != expected_protocol:
        raise ReplayRefused("chosen versioned protocol hash mismatch")
    if release.get("approved_by") != APPROVER or release.get("decision") != "APPROVED":
        raise ReplayRefused("release lacks the named author's explicit approval")
    scope = release.get("scope")
    if mode == "sample_first_57_v1":
        if scope != [1]:
            raise ReplayRefused("sample-first diagnostic requires scope [1] only")
    else:
        if (not isinstance(scope, list) or not scope or
                scope != list(range(len(scope))) or any(s not in (0, 1, 2) for s in scope)):
            raise ReplayRefused("approval scope must be exactly [0], [0,1] or [0,1,2]")
        if through > max(scope):
            raise ReplayRefused(f"Stage {through} not included in approval scope {scope}")
    if sha_file(bundles_path) != BUNDLES_SHA256 or release.get("bundles_sha256") != BUNDLES_SHA256:
        raise ReplayRefused("bundle bytes are not the pinned archived population")
    if release.get("sample_manifest_sha256") != sha_file(sample_path):
        raise ReplayRefused("sample manifest is not the approved one")
    if release.get("sample_selection_sha256") != SAMPLE_SELECTION_SHA256:
        raise ReplayRefused("sample selection is not approved")
    expected_code = {name: sha_file(HERE / name) for name in EXECUTION_FILES}
    if release.get("code_sha256") != expected_code:
        raise ReplayRefused("replay executable code has drifted since approval")
    expected_amendments = {name: sha_file(ROOT / "protocol" / name) for name in AMENDMENT_FILES}
    if release.get("amendments_sha256") != expected_amendments:
        raise ReplayRefused("protocol amendment files are not bound to approval")
    if release.get("render") is not True:
        raise ReplayRefused("approved confirmatory replay must use 360px rendering")
    if platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise ReplayRefused("protocol declares an arm64 macOS host; other hosts need a versioned decision")
    return release, sample, expected_code, expected_amendments


def runtime_versions():
    import sys
    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != "3.12.3":
        raise ReplayRefused("pinned replay interpreter must be Python 3.12.3, found " + python_version)
    versions = {}
    for name, expected in ENV_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ReplayRefused("required replay package missing: " + name) from error
        if actual != expected:
            raise ReplayRefused(f"version drift for {name}: expected {expected}, found {actual}")
        versions[name] = actual
    return versions


def verify_assets(asset_root, manifest_path):
    """Verify the pinned complete asset tree, ignoring downloader metadata only."""
    root, manifest = Path(asset_root), Path(manifest_path)
    if sha_file(manifest) != ASSET_MANIFEST_SHA256:
        raise ReplayRefused("asset manifest bytes differ from the pinned manifest")
    expected = {}
    for line in manifest.read_text("utf-8").splitlines():
        relative, digest = line.split("\t")
        relative = relative.removeprefix("./")
        parts = Path(relative).parts
        if not parts or Path(relative).is_absolute() or ".." in parts or relative in expected:
            raise ReplayRefused("invalid or repeated asset manifest path")
        expected[relative] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and not ({".cache", ".git"} & set(p.relative_to(root).parts))}
    if actual != set(expected):
        raise ReplayRefused("asset tree does not match the pinned file set")
    for relative, digest in expected.items():
        if sha_file(root / relative) != digest:
            raise ReplayRefused("asset file hash mismatch: " + relative)
    return {"path": str(root.resolve()), "files": len(expected), "manifest_sha256": ASSET_MANIFEST_SHA256}


def verify_task_inputs(archive_path):
    """Match each task BDDL and official state-array hash to the pinned archive."""
    import libero_replay as lr
    expected = defaultdict(set)
    phases = ("confirmation", "primary_deadline", "confirmation_continuation_partial",
              "confirmation_continuation_rest")
    with tarfile.open(archive_path, "r:gz") as archive:
        for phase in phases:
            with archive.extractfile(phase + "/episodes.jsonl") as stream:
                for line in stream:
                    episode = json.loads(line)
                    ident = episode["episode_identity"]
                    key = (ident["suite"], ident["task_id"])
                    expected[key].add((episode["task_bddl_sha256"], episode["initial_state_set_sha256"]))
    wanted = {(suite, task) for suite in ("libero_goal", "libero_object") for task in (5, 6, 7, 8)}
    if set(expected) != wanted or any(len(values) != 1 for values in expected.values()):
        raise ReplayRefused("archive task input identities are missing or inconsistent")
    checked = []
    for suite, task in sorted(wanted):
        _, _, bddl = lr.task_bddl(suite, task)
        states, init_path = lr.official_init_states(suite, task)
        actual = (sha_file(bddl), lr.array_sha(states))
        if actual not in expected[(suite, task)]:
            raise ReplayRefused(f"task or init-state hash mismatch: {suite}:{task}")
        checked.append({"suite": suite, "task_id": task, "bddl_sha256": actual[0],
                        "initial_state_set_sha256": actual[1], "init_file_sha256": sha_file(init_path)})
    return checked


def verify_run(bundles, bundles_path, sample_path, release_path, protocol_path, through, render,
               mode="frozen", archive_path=None):
    """Validate everything before any output directory or simulator can be created."""
    if not render:
        raise ReplayRefused("confirmatory replay requires rendering")
    release, sample, code_hashes, amendments = _strict_release(
        release_path, protocol_path, bundles_path, sample_path, through, mode=mode)
    if mode == "sample_first_57_v1":
        # The separate protocol imports the frozen tolerance-mismatch adjudication
        # while also requiring immediate stop. Resolve this before any simulation;
        # an approval alone cannot make an unqualified implementation safe.
        raise ReplayRefused("sample-first implementation not qualified: mismatch adjudication versus immediate stop requires a protocol decision")
    pop = verify_population(bundles, sample)
    versions = runtime_versions()
    config_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_dir or not (Path(config_dir) / "config.yaml").is_file():
        raise ReplayRefused("LIBERO_CONFIG_PATH must name an existing explicit config")
    from libero.libero import get_libero_path
    assets = verify_assets(get_libero_path("assets"), ROOT / "protocol" / "libero_assets_0b3ea86_manifest.tsv")
    tasks = None
    if mode == "sample_first_57_v1":
        if not archive_path or sha_file(archive_path) != ARCHIVE_SHA256:
            raise ReplayRefused("original archive does not match the pinned bytes")
        tasks = verify_task_inputs(archive_path)
    return {"approval_scope": release["scope"], "requested_through": through,
            "execution_mode": mode,
            "release_sha256": sha_file(release_path),
            "protocol_sha256": sha_file(protocol_path),
            "amendments_sha256": amendments, "code_sha256": code_hashes,
            "bundles_sha256": BUNDLES_SHA256, "sample_manifest_sha256": sha_file(sample_path),
            "source_archive_sha256_declared": ARCHIVE_SHA256, "assets": assets,
            "task_inputs": tasks,
            "sample_selection_sha256": SAMPLE_SELECTION_SHA256, "population": pop,
            "platform": platform.platform(), "machine": platform.machine(),
            "runtime_versions": versions,
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def record_preflight(path, report):
    """Write counted identity before Stage 0, with O_EXCL and fsync."""
    import os
    path = Path(path)
    data = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    parent_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def checkpoint_evidence(root):
    """Atomically seal all currently complete files; called after each durable row."""
    root = Path(root)
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in ('SHA256SUMS', '.SHA256SUMS.tmp'):
            entries.append(f"{sha_file(path)}  {path.relative_to(root).as_posix()}\n")
    tmp = root / '.SHA256SUMS.tmp'
    with tmp.open('w', encoding='utf-8') as stream:
        stream.writelines(entries)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, root / 'SHA256SUMS')
    fd = os.open(str(root), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
