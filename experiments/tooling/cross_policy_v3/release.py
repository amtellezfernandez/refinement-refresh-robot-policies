"""Release and authorisation checks for v3 entry points (G0-B, E-AUTH). Python 3.8+, stdlib only.

Protocol B: Alba's release must bind the Protocol B document's sha256 and, per model, the schedule's row digest
and file bytes. The schedule check reuses the frozen `launch_guard.verify_released_schedule` unchanged.
Protocol E and the attesting server's standalone modes: an authorisation file bound to the governing document
and to named input files. No file of either kind exists in this repository; only Alba's review creates one.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3common import PROTOCOL_B, PROTOCOL_B_DOCUMENT, REPO_ROOT, file_sha  # noqa: E402
import launch_guard  # noqa: E402  (frozen v2 module, imported unchanged)


class Refused(SystemExit):
    pass


def release_path(root=REPO_ROOT):
    return Path(root) / "review/releases" / (PROTOCOL_B + ".release.json")


def check_release(schedule_path, model, root=REPO_ROOT, document=PROTOCOL_B_DOCUMENT):
    """G0-B: refuse unless Alba's Protocol B release binds this document and this model's schedule."""
    path = release_path(root)
    if not path.is_file():
        raise Refused("REFUSED: {} has no release at {}".format(PROTOCOL_B, path))
    try:
        release = json.loads(path.read_text("utf-8"))
    except ValueError as error:
        raise Refused("REFUSED: release is not JSON: {}".format(error))
    if release.get("approved_by") != "Alba":
        raise Refused("REFUSED: release is not Alba's")
    if release.get("protocol") != PROTOCOL_B:
        raise Refused("REFUSED: release names protocol {!r}".format(release.get("protocol")))
    if release.get("protocol_document_sha256") != file_sha(document):
        raise Refused("REFUSED: release does not match the Protocol B document hash")
    if model not in release.get("released_models", []):
        raise Refused("REFUSED: {} is not released".format(model))
    try:
        bound = launch_guard.verify_released_schedule(Path(root), "confirmation", PROTOCOL_B, release,
                                                      schedule_path, model)
    except launch_guard.LaunchRefused as error:
        raise Refused(str(error))
    bound["release_sha256"] = file_sha(path)
    return bound


def check_authorisation(path, document, bound_files, scope):
    """E-AUTH and standalone server modes: an authorisation bound to a document and to named files.

    {"approved_by": "Alba", "scope": scope, "document_sha256": ..., "bound": {name: sha256, ...}}
    """
    path = Path(path) if path else None
    if path is None or not path.is_file():
        raise Refused("REFUSED: no authorisation file for {}".format(scope))
    try:
        record = json.loads(path.read_text("utf-8"))
    except ValueError as error:
        raise Refused("REFUSED: authorisation is not JSON: {}".format(error))
    if record.get("approved_by") != "Alba" or record.get("scope") != scope:
        raise Refused("REFUSED: authorisation is not Alba's, or not for {}".format(scope))
    if record.get("document_sha256") != file_sha(document):
        raise Refused("REFUSED: authorisation does not match {}".format(Path(document).name))
    bound = record.get("bound", {})
    for name, file_path in bound_files.items():
        if bound.get(name) != file_sha(file_path):
            raise Refused("REFUSED: {} is not the authorised file".format(name))
    return {"authorisation_sha256": file_sha(path), "scope": scope}
