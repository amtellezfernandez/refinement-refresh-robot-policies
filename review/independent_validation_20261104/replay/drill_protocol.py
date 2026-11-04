#!/usr/bin/env python3
"""Fire drill for run_protocol.py on synthetic fixtures relabelled as 'archived'.
Drill A: clean -> every stage runs.  Drill B: one recorded initial state forged (+1e-3,
rehashed) -> stage 0 FAIL_RESET, exit 6, stage 1 never starts.
Usage: drill_protocol.py FIXTURES.jsonl SCRATCH OUT.json"""
import copy, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import libero_replay as lr
fx = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
scr = Path(sys.argv[2]); scr.mkdir(parents=True, exist_ok=False)
proto = scr / "PROTOCOL_DRILL.md"; proto.write_text("drill protocol\n")
rel = scr / "release.json"
rel.write_text(json.dumps({"protocol_sha256": hashlib.sha256(proto.read_bytes()).hexdigest(),
                           "approved_by": "drill", "decision": "APPROVED"}))
arch = []
for i, b in enumerate(fx):
    b = copy.deepcopy(b); b["provenance"] = "archived_paper_evidence"; b["condition"] = "h2_d2" if i % 2 else "h2_d5"
    arch.append(b)
sample = scr / "sample.json"; sample.write_text(json.dumps({"stage1_sample": [b["episode_key"] for b in arch[::3]]}))
def go(name, bundles):
    p = scr / f"{name}.jsonl"; p.write_text("\n".join(json.dumps(b) for b in bundles) + "\n")
    r = subprocess.run([sys.executable, "-I", str(HERE / "run_protocol.py"), str(p), str(sample), str(scr / name),
                        "--release", str(rel), "--protocol", str(proto), "--no-render"], capture_output=True, text=True)
    return r.returncode, [l for l in r.stdout.splitlines() if l.startswith("stage")], r.stderr.strip().splitlines()[-1:]
res = {}
code, out, err = go("drillA", arch)
res["A_clean"] = {"exit": code, "stdout": out, "stage1_started": (scr / "drillA" / "stage1").exists(),
                  "stage2_started": (scr / "drillA" / "stage2").exists(), "pass": code == 0 and len(out) == 3}
forged = copy.deepcopy(arch)
st = lr.array_from_record(forged[-1]["recorded_initial_state"]).astype(np.float64); st[12] += 1e-3
a = np.ascontiguousarray(st)
forged[-1]["recorded_initial_state"] = {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(), "sha256": lr.array_sha(a)}
code, out, err = go("drillB", forged)
res["B_forged_reset"] = {"exit": code, "stdout": out, "stderr_tail": err,
                         "stage1_started": (scr / "drillB" / "stage1").exists(),
                         "pass": code == 6 and not (scr / "drillB" / "stage1").exists()}
mis = copy.deepcopy(arch)
assert not mis[0]["label"]["success"] and mis[0]["episode_key"] in json.loads(sample.read_text())["stage1_sample"]
mis[0]["recorded_success_flags"][-1] = True
mis[0]["label"].update(success=True, termination_reason="success")
code, out, err = go("drillD", mis)
res["D_planted_mislabel_in_sample"] = {"exit": code, "stdout": out, "stderr_tail": err,
                                       "stage2_started": (scr / "drillD" / "stage2").exists(),
                                       "pass": code == 6 and any("DISAGREEMENT" in l for l in out)
                                       and not (scr / "drillD" / "stage2").exists()}
drift = copy.deepcopy(mis)
for b in drift:  # every recorded start moved 1e-9 (inside RESET_TOL) and rehashed
    st = lr.array_from_record(b["recorded_initial_state"]).astype(np.float64); st[12] += 1e-9
    a = np.ascontiguousarray(st)
    b["recorded_initial_state"] = {"dtype": a.dtype.str, "shape": list(a.shape), "values": a.tolist(), "sha256": lr.array_sha(a)}
code, out, err = go("drillE", drift)
adj = json.loads((scr / "drillE" / "stage1" / "stage_verdict.json").read_text())["adjudication"] if (scr / "drillE" / "stage1" / "stage_verdict.json").exists() else None
res["E_drift_plus_mislabel_probe"] = {"exit": code, "stdout": out, "adjudication": adj,
                                      "pass": code == 6 and out and "PASS_WITHIN_TOLERANCE" in out[0]
                                      and any("DISAGREEMENT" in l for l in out) and bool(adj)
                                      and adj[0]["archived_outcome_reachable"] is False}
r = subprocess.run([sys.executable, "-I", str(HERE / "run_protocol.py"), str(scr / "drillA.jsonl"), str(sample),
                    str(scr / "drillC"), "--release", str(scr / "missing.json"), "--protocol", str(proto)],
                   capture_output=True, text=True)
res["C_missing_release"] = {"exit": r.returncode, "stderr_tail": r.stderr.strip().splitlines()[-1:],
                            "pass": r.returncode == 5 and not (scr / "drillC").exists()}
Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
