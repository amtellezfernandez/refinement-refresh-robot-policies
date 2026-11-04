#!/usr/bin/env python3
"""Branch test for the CHAOTIC_INCONCLUSIVE verdict of run_protocol.episode_stage.

The physics probe is REPLACED by a stub (no fixture whose outcome flips under 1e-9 start noise
is known), so this tests the classification and stage-gate logic only, not the probe.
Setup: synthetic fixtures relabelled 'archived', Stage 0 declared PASS_WITHIN_TOLERANCE, one
consistent false success planted.  Stub A says the archived outcome IS reachable in the
ensemble -> CHAOTIC_INCONCLUSIVE, and the Stage 2 condition refuses it.  Stub B says it is
not reachable -> DISAGREEMENT (the real-probe path is covered by drill E).
Usage: test_stage_logic.py FIXTURES.jsonl SCRATCH OUT.json"""
import copy, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_protocol as P
fx = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
scr = Path(sys.argv[2]); scr.mkdir(parents=True, exist_ok=False)
proto = scr / "P.md"; proto.write_text("stage logic test protocol\n")
rel = scr / "r.json"; rel.write_text(json.dumps({"protocol_sha256": hashlib.sha256(proto.read_bytes()).hexdigest(),
                                                  "approved_by": "test", "decision": "APPROVED"}))
bs = []
for b in fx[:10]:
    b = copy.deepcopy(b); b["provenance"] = "archived_paper_evidence"; b["condition"] = "h2_d2"; bs.append(b)
target = next(b for b in bs if not b["label"]["success"])
target["recorded_success_flags"][-1] = True
target["label"].update(success=True, termination_reason="success")
out, calls = {}, []
for name, reachable, want in (("A_reachable", True, "CHAOTIC_INCONCLUSIVE"), ("B_unreachable", False, "DISAGREEMENT")):
    def stub(b, render, rng, r=reachable):
        calls.append(b["episode_key"])
        arch = bool(b["label"]["success"])
        return ([arch] if r else [not arch]) + [not arch] * (P.SENSITIVITY_DRAWS - 1)
    P.sensitivity_probe = stub
    res = P.episode_stage(1, bs, scr / name, False, str(rel), str(proto), "PASS_WITHIN_TOLERANCE")
    stage2_allowed = res["verdict"] in P.PROCEED | {"PASS_OUTCOME_ONLY"}
    out[name] = {"verdict": res["verdict"], "expected": want, "stage2_allowed": stage2_allowed,
                 "pass": res["verdict"] == want and not stage2_allowed}
    print(name, out[name], flush=True)
out["probe_called_only_on_mismatch"] = {"calls": list(calls), "pass": calls == [target["episode_key"]] * 2}
# under an exact Stage 0 the probe must not run and the mismatch is a DISAGREEMENT
calls.clear()
res = P.episode_stage(1, bs, scr / "C_exact", False, str(rel), str(proto), "PASS_EXACT")
out["C_exact_stage0_no_probe"] = {"verdict": res["verdict"], "calls": list(calls),
                                  "pass": res["verdict"] == "DISAGREEMENT" and not calls}
print("C", out["C_exact_stage0_no_probe"])
out["passed"] = sum(v["pass"] for v in out.values() if isinstance(v, dict)); out["total"] = 4
Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + "\n")
sys.exit(0 if out["passed"] == 4 else 1)
