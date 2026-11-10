# Prepare the unreleased R22/R25 round

Use the [installed BenchStudio recipe](../../../integrations/benchstudio/README.md)
for the full round. The paper frontend owns its scientific sequence and uses
BenchStudio's process infrastructure. Preparation inventories retained artifacts;
it does not generate states, run native checks, create authority or qualify a stack.

## Existing workspace inputs

The canonical workspace is
`REPOSITORY/experiments/runs/cross_policy_confirmation_v2/`, outside frozen
evidence. Each model's folder needs its existing native `manifest.json`,
`schedule.json`, `initial_states.npz`, `fixtures.json` and pinned `checkpoint/`.
Schedules describe 400 pairs / 800 attempts per model. The models are analyzed
separately.

The pi05 server's **run-specific** gate is
`pi05/server/startup_gate.json`. The owned server creates it after its startup
checks during an independently released round. Do not supply a historical gate
or create that file to satisfy preparation. The frontend refuses existing
server/run outputs. Prior count qualification is a separate mandatory item in
the retained qualification record; absence of a future startup file does not
waive that item. The native client still verifies the live gate and each
request's attestation before applying actions.

## One operator interface

In a separate controlling Python >=3.10 environment containing the reviewed
BenchStudio and paper integration, substitute the actual checkout revision and
three native interpreter paths:

```sh
benchstudio evaluate --recipe refinement-refresh \
  --repository /path/to/paper --revision FULL_REVIEWED_PAPER_COMMIT_SHA \
  --smolvla-python /path/to/smolvla/bin/python \
  --pi05-client-python /path/to/libero-client/bin/python \
  --pi05-server-python /path/to/openpi-server/bin/python --check
```

The matching Python interface is `bs.pipeline("refinement-refresh", ...).check()`
after `import benchstudio as bs`; the integration guide contains the complete
example. The recipe reports offline input completeness as
`configuration_valid` and authority as `execution_authorized`, separately.
Neither implies verified native imports, checkpoints, determinism, action
semantics or evaluator validity. Interpret the listed issues and retained native
evidence before making a readiness decision.

The underlying stdlib inventory helper remains available for a model-specific
diagnostic report:

```sh
/path/to/python experiments/tooling/cross_policy_v3/prepare_campaign.py --model pi05
```

An exit code of zero means a report was produced. Inspect
`offline_inputs_complete`, qualification items and `execution_authorized`.
The helper's argument arrays and shell commands are individual components,
not another full-round launcher. It does not start the server or execute those
commands. The full frontend rechecks both stacks before starting either model,
then owns server readiness, client lifetime, logs and separate analyses.

## Remaining boundary

Use only a trusted, clean checkout at the reviewed full revision. Keep the
Python 3.8 LIBERO client separate from the OpenPI/JAX server (Python >=3.11)
and the controlling environment. The offline inventory does not attest those
installed native environments.

The model-keyed native state generator remains to be implemented and reviewed.
Native manifests, Protocol A/E, parity, action mapping and the served compiled
counter still need their retained evidence. The report must leave missing items missing. Neither an implementation merge nor
a passing unit test releases the round. E0 stays on its separately approved,
immutable staging package and exclusive single-use registry. This workflow
does not execute it or change that permission.

No preparation step writes frozen evidence, a release, a qualification verdict
or reserved identities. The frontend's documented cross-workspace/cross-host
duplicate-campaign limitation remains unresolved.
