# Ledger v3: completion, configuration, and reference checks

V1/v2 artifacts remain unchanged. The current validator requires v3 and does not
retroactively certify those earlier runs. No performance sweep is authorized by
a successful infrastructure diagnostic alone.

Every environment transition records reward, success, termination and truncation.
An episode must end at termination/truncation or the declared evaluator horizon;
prefix-only trajectories are rejected even if their action/call counts reconcile.
Finalization checks episode completion, per-task outcomes and aggregate success
against the transition trace before writing a completed-run marker. Journals
without finalization are interrupted diagnostics, not completed results.

The run contract stores configuration (excluding only output directory/job name),
checkpoint file hashes, policy/environment/evaluator/ledger source hashes and key
package versions. Paired comparison permits only policy prefix length and solver
step changes; replay requires identical contracts. This is an improvement in
provenance, not a full container/driver/environment-asset lock.

`ROBOPAD_REFERENCE_CHECK=1` enables an expensive diagnostic: each instrumented
action is compared exactly with original policy methods using the same queue
state and tick-keyed RNG. Queues and RNG are restored afterward. Reference
sampling calls are counted separately; diagnostic wall time is not deployment
performance. This checks the measurement wrappers, not unchanged-upstream RNG
semantics or published benchmark scores. Warm-up remains separately accounted.

Commands (isolated LeRobot Python environment):

Verified on 2026-09-10: eight unit-test groups pass. The two earlier audit
attacks (changed success report and coordinated trajectory-prefix truncation)
are rejected on in-memory copies of a real run. A two-episode Spatial-task-0
diagnostic covered initial states 0/1 and seeds 1000/1001: state 0 succeeded at
86 actions; state 1 failed at the evaluator's 280-step horizon (neither native
termination nor truncation was set). All 366 instrumented actions exactly
matched calls through the original policy methods under matched queues/noise;
37 extra reference sampling calls are separately recorded. Both episode endings
and the final reported 1/2 success passed v3 validation. This is an infrastructure
test, not a performance claim or a full upstream benchmark reproduction.

Raw ledgers and the exact recorded ledger source are archived under
`results/infrastructure_v3/`. Remote journals/videos are preserved at
`/mnt/NVME1/projectx_robopad/logs/ledger_v3_2eqbn2UE/`. The recorded source differs
from the current source only in its introductory version-label docstring.

```bash
python -m unittest -v test_ledger
python validate_ledger.py RUN/eval_info.json
python validate_ledger.py RUN/eval_info.json --replay REPEAT/eval_info.json
```

Before a broad sweep: validate repeated complete episodes including a timeout,
multiple initial states, reference-equivalence diagnostics, and an explicit
upstream-versus-controlled-RNG comparison. Only then expand balanced task/state/
seed blocks. Neither unchanged task quality nor safety follows from these checks.
