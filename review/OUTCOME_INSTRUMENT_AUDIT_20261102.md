# Outcome instrumentation audit, 2 November 2026

## Finding

The retained evidence is stronger than an aggregate recount alone. The complete
confirmation archive contains a step-level trace for every completed attempt.
An independent standard-library audit checks each episode against that trace and
passes without a mismatch.

This supports the integrity of the recorded outcome path. It does not constitute
an independent simulator replay from archived actions and initial states.

## Reproduced checks

Run:

```bash
python3 experiments/tooling/audit_outcome_instrument.py \
  experiments/runs/refinement_refresh_final_001/confirmation_completed.tar.gz \
  --output review/outcome_instrument_audit_20261102.json
```

Across 1,122 completed raw episodes and 172,243 completed steps:

- every episode has matching start and completion events;
- episode, base-environment hook, vector evaluator and completion recordings of
  the same upstream success value agree;
- producer and environment-consumer action hashes agree at every step;
- episode, independent-driver and journal step counts agree;
- initial identity, simulator-state hash and observation hash agree;
- every nonsuccess step has reward 0 and every success step has reward 1; this
  is a consistency check, not an independent outcome oracle, because LIBERO's
  sparse reward uses the same success predicate;
- one additional ten-step interrupted stream has no completed outcome and stays
  outside these totals.

The four phase manifests all bind the LeRobot LIBERO wrapper to SHA-256
`97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71`.
That hash matches the file at pinned upstream LeRobot commit
`b6ec0060779550c0a157ae34feb89e0cf86012a8`. The wrapper computes
`is_success = self._env.check_success()` after every simulator step. The recorded
environment package is `hf_libero==0.1.4`; its published wheel routes
`check_success()` to the task environment's BDDL goal-predicate evaluation.
The multiple success fields and sparse reward therefore verify propagation of
that result through the recorded execution path; they do not supply independent
semantic judgments of the terminal simulator state.

The earlier development parity artifact adds one end-to-end native-setting check:
the instrumented runner and pinned upstream rollout have identical initial state,
initial observation, 70-action stream, episode length and successful outcome.
This is a useful positive control, but it covers one Spatial task, state and seed.

## Remaining gap

The evidence rules out many logging, aggregation, action-dispatch and wrapper-path
errors. The remaining high-value validation is a model-free replay of archived
actions from archived initial states in a separately instantiated, version-matched
LIBERO environment, followed by direct evaluation of the task predicates and
inspection of terminal simulator states. A
stratified replay should include successes and horizon failures from every one of
the eight confirmation tasks and both primary conditions. Full 320-attempt replay
would directly cover the reported primary comparison.

The archive records the LIBERO package version but does not retain a cryptographic
hash of the installed LIBERO package tree. A replay should therefore record the
wheel hash, BDDL file hashes, simulator versions and per-attempt final predicate
values. Until that replay is complete, outcome-instrument validity is strongly
supported internally but not fully independently established.
