# Trace v0.2 examples

These are synthetic software fixtures. Numeric constants are illustrative and
are marked `derived` with a synthetic derivation; they are not robot results.

- `hybrid_system_trace.json` exercises memory retrieval, policy generation,
  world-model imagination, evaluation, selection, real execution, and learning.
  It is complete only for its explicitly declared resource requirements.
- `deployment_trace.json` is a frozen-policy simulator example. Its unknown
  invocation time is `null`; its real-interaction resource is `not_applicable`.
  It passes structural validation and fails `--require-complete` with exit 2.
- `invalid_update_trace.json` has no data input to its update. It must fail
  structural/semantic validation with exit 1.

From the parent directory:

```bash
python validate_trace.py examples/hybrid_system_trace.json --require-complete
python validate_trace.py examples/deployment_trace.json
python -m unittest -v test_trace
```

The validator uses the standard library. Installing `jsonschema` additionally
enables an independent schema-parity test; that test explicitly skips when the
optional package is absent. JSON Schema validates record structure. The Python
validator also checks references, operator signatures, unique producers, and
acyclic artifact lineage.

## Migration from v0.1

Declare a top-level `boundary` containing scope, description, inclusions, and
exclusions. Frozen-policy deployment is a valid measurement boundary; it is not
automatically labeled adaptation. Every event declares `required_resources`, a
nonempty list of resource names selected for that event's comparison boundary.
Each supplied resource contains its name, kind, amount, unit, boundary, status,
and aggregation rule. `measured` and `derived` require finite nonnegative
numbers; `derived` also requires a derivation. `unknown` and `not_applicable`
require `null`, while a measured zero remains numeric zero. `EXECUTE` explicitly
declares `real` or `simulated`, with a consistent physical-world-effect flag.

The `sum`, `max`, `snapshot`, and `interval` aggregation labels state intended
semantics. The validator does not combine measurements, convert units, check
overlapping timers, or infer a total from these labels. In particular, outer wall
time is not the sum of parallel event durations, and retained storage is not the
sum of repeated snapshots.

Structural validity and declared-resource completeness are separate results.
Deleting a required resource produces an incomplete report, not a zero. Deleting
an entire event can still leave a valid graph: independent driver counters are
needed to detect such omissions. Validity and completeness do not establish
measurement accuracy, sufficient accounting boundaries, justified applicability,
or causal effects. Historical robot ledgers use a different schema and are not
retroactively certified by these fixture tests.
