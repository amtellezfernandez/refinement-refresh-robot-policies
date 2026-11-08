# BenchStudio integration for the native R22/R25 recipe

This package connects the existing paper frontend to BenchStudio's installed
pipeline interface. It introduces no alternative scientific runner, generic trial
conversion, state generation or permission file. Construction and check() never
run a native campaign. The native runner remains the authority for experimental
identities, 400 pairs/800 attempts per model, gates and analysis.

Install the reviewed BenchStudio version supporting installed pipelines in a
separate controlling Python >=3.10 environment. Install this integration there:

```sh
python -m pip install --no-deps -e /path/to/paper/integrations/benchstudio
```

Do not install it into the existing Python 3.8 LIBERO client or alter that client's
package pins. This package needs the paper frontend from the reviewed PR #14
three-interpreter stack, and the offline preparation helper. It does not make
unmerged native implementation drafts available on paper main.

```python
import benchstudio as bs

campaign = bs.pipeline(
    "refinement-refresh",
    repository="/path/to/paper",
    revision="FULL_REVIEWED_PAPER_COMMIT_SHA",
    smolvla_python="/path/to/smolvla/bin/python",
    pi05_client_python="/path/to/libero-client/bin/python",
    pi05_server_python="/path/to/openpi-server/bin/python",
)
status = campaign.check()
print(status.summary())
```

Replace the illustrative revision and paths with the actual pinned checkout and
native interpreters. They cannot be inferred safely from another host's logs. No
source filename, launcher, result converter or manually constructed workspace is
required from the operator. The recipe derives the existing canonical workspace.
The full checkout revision is checked before invoking its preparation helper.

Only for a separately authorized and natively qualified campaign:

```python
results = campaign.run()
print(results.summary())
```

run() refuses unless the existing preparation reports for both models say the
technical inputs are complete and execution is authorized. It then invokes
experiments/run_r22r25.py through bs.ProcessCommand. The existing frontend checks
both native stacks again before either model, manages the pi0.5 server/client,
and invokes each model's existing analysis. The package does not bypass those
checks, add retries or generate identities. A native failure propagates, retaining
its logs. Missing or malformed analysis files do not yield fabricated results.

results.analyses retains the two native reports separately; analysis_sha256 binds
their actual files. They are not pooled, converted into one success trial or
claimed to be a comparable cross-model treatment effect. Technical readiness and
execution authorization are exposed separately on the preparation report.

## Current limitation

The published native evidence still does not qualify either model. Required
native state-generation tooling, state manifests/schedules, Protocol E,
parity/action-unit and compiled-counter verification remain substantive blockers.
This package wires the invocation; it does not supply that missing evidence or
approve any run. E0 remains on its existing independent, single-use path; it is
not executed by construction, check(), installation or this package's tests.

Maintainers own this integration. Warren should operate the reviewed recipe and
return host evidence rather than write another integration script.

## Offline tests

Run tests in the controlling environment with BenchStudio and this package installed:

```sh
python -m pytest integrations/benchstudio/tests -q
```

The tests use synthetic git checkouts, fake readiness and a stdlib-only fake
frontend. They do not instantiate a simulator or load a policy/checkpoint.


## One installed CLI command

The same recipe is available through the unified CLI (substitute the actual pins
and host paths; the revision placeholder is deliberately not executable):

```sh
benchstudio evaluate --recipe refinement-refresh \
  --repository /path/to/paper --revision FULL_REVIEWED_PAPER_COMMIT_SHA \
  --smolvla-python /path/to/smolvla/bin/python \
  --pi05-client-python /path/to/libero-client/bin/python \
  --pi05-server-python /path/to/openpi-server/bin/python --check
```

This preparation command never invokes the campaign. Only after the separately
required native qualification, inputs and authorization exist can the operator
remove --check to invoke the same release-gated native recipe. E0 is not a step
in this command and its single-use permission cannot authorize this campaign.
