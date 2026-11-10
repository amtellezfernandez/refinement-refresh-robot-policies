#!/usr/bin/env python3
"""Check every parity video and build contact sheets for a person to look at.

    inspect_videos.py <videos dir>

Machine checks catch missing, unreadable, blank and frozen files. They do not replace looking: the
contact sheets, one per task, are for human inspection, which is recorded separately.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import atomic_write_json, file_sha  # noqa: E402


def frame_checks(frames):
    import numpy as np
    if not frames:
        return {"ok": False, "frames": 0, "reason": "no frames"}
    stack = np.asarray(frames)
    blank = float(stack.std()) < 1.0
    frozen = len(frames) > 1 and float(np.abs(stack[-1].astype(float) - stack[0].astype(float)).mean()) == 0.0
    return {"ok": not blank and not frozen, "frames": len(frames), "shape": list(stack.shape[1:]), "blank": blank, "frozen": frozen}


def contact_sheet(frames, columns=6):
    import numpy as np
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, columns)]
    return np.concatenate(picks, axis=1)


def main(directory):
    import imageio.v2 as imageio
    import numpy as np
    directory = Path(directory)
    sheets = directory / "contact_sheets"
    sheets.mkdir(exist_ok=False)
    rows, by_task = [], {}
    for path in sorted(directory.glob("*.mp4")):
        try:
            frames = [np.asarray(frame) for frame in imageio.get_reader(str(path))]
            checks = frame_checks(frames)
        except Exception as error:  # An unreadable video is a finding, recorded and counted.
            frames, checks = [], {"ok": False, "reason": repr(error)}
        rows.append(dict(checks, file=path.name, sha256=file_sha(path)))
        match = re.match(r"(parity\.[^.]+\.[^.]+\.task\d+)\.ep(\d+)", path.name)
        if match and frames and match.group(1) not in by_task:
            by_task[match.group(1)] = contact_sheet(frames)
    for task, sheet in by_task.items():
        imageio.imwrite(str(sheets / (task + ".png")), sheet)
    atomic_write_json(directory / "video_manifest.json", {
        "videos": len(rows), "failed": sum(not row["ok"] for row in rows), "contact_sheets": len(by_task), "rows": rows,
        "human_inspection": "NOT YET DONE"})
    print(json.dumps({"videos": len(rows), "failed": sum(not row["ok"] for row in rows)}))


if __name__ == "__main__":
    main(sys.argv[1])
