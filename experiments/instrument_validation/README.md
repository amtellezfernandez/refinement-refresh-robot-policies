# Executed offline accounting-validation experiment

This experiment runs deterministic corruptions of the four archived v2 SmolVLA
ledgers through four cumulative validation layers. It performs local software
validation only: no new policy inference, robot rollout, or human study occurs.
Original inputs under `../real_smolvla/results/infrastructure_v2/` are read only.

## Executed result

The fixed suite contains **68 constructed trials and 272 layer checks**:
13 observable corruptions per source (52 trials), one untouched control per source
(4 trials), and three explicitly unobservable scenarios per source (12 trials).

| Cumulative layer | Observable defects rejected | Untouched controls accepted | Unobservable probes | Validator errors |
|---|---:|---:|---:|---:|
| Strict JSON syntax | 4/52 | 4/4 | 12 unknown | 0 |
| + Structural schema | 12/52 | 4/4 | 12 unknown | 0 |
| + Aggregate reconciliation | 28/52 | 4/4 | 12 unknown | 0 |
| + Full ledger reconciliation | 52/52 | 4/4 | 12 unknown | 0 |

These are finite-suite counts, not independent sample estimates or expected
detection rates for naturally occurring mistakes. There are no confidence
intervals or hypothesis tests. All four sources share one task/state, and two
are same-condition repeats.

The aggregate layer catches missing/duplicate consumer events, sampler-count
drift, and contradictory success summaries beyond structural failures. Full
reconciliation additionally catches stale hashes, producer/consumer disagreement
despite a recomputed valid hash, identity drift, reordered steps, balanced-count
duplicate chunks, and interleaved warm-up. Each expected violation is declared
in the frozen protocol, and every decision and rejection reason is retained.

## What "unknown" means

All layers technically **accept** the 12 unobservable probes. The experiment
annotates their requested claim as **unknown** according to the prespecified
scope; the ledger validator does not automatically infer this epistemic label.
These probes are excluded from the 52 observable-defect denominator.

- Joint action rewriting changes both recorded streams and recomputes valid
  hashes. Without an external expected-action record, consistency does not
  establish which action was actually dispatched.
- The preprocessing and rendering probes leave input bytes unchanged and ask
  whether every relevant pipeline cost was recorded. The archive lacks an
  independent expected-stage manifest or corresponding stage-level costs.
  These are unknown coverage queries, not invented measured omissions.

The four layers apply to one artifact at a time. An independent reference can
make otherwise unobservable discrepancies detectable, but that is a different
comparison contract. Pairing/replay against an external reference is not smuggled
into this single-artifact experiment.

## Design and provenance

`protocol.json` was frozen **before the first validation trial**. It records all
17 case definitions, applicability, estimands, source digests, structural-schema
digest, and runner/validator-code digests. Hashes normalize CRLF to LF for Git
checkouts. Source files or validation code that differ from the frozen protocol
cause execution to fail. Input artifacts are cloned in memory; no corruption is
written over an original.

The four layers are:

1. Strict JSON parsing, including rejection of nonstandard numeric tokens.
2. The supplied JSON Schema: required fields, local types and ranges, without
   cross-record equality, hashing, chronology, or semantic reconciliation.
3. Structural checks plus task/group/overall outcomes, episode totals, producer
   and consumer stream lengths, selection totals, and per-phase event/counter
   agreement.
4. Those checks plus the existing `validate_completed` implementation: hashes,
   producer/consumer action equality, identities, seed consistency, ordering,
   warm-up placement, and the action-chunk schedule.

The aggregate layer is an explicit ablation of v2 evidence. It does not imply
that historical v1 artifacts contained the same independent streams. V1 is not
used because its missing actions/identities make these mutation classes
inapplicable. Unexpected validator exceptions are stored separately and never
credited as successful detection.

Mutations were chosen with knowledge of the validator contract, so rejecting
them is a mechanism check with deliberately known ground truth. This is not an
adversarial coverage guarantee or evidence of reader benefit, adaptation quality,
robot safety, real-world collection completeness, or lifecycle accounting.

## Outputs and reproduction

- `trials.json`: every source/case/layer result, reason, ground-truth category,
  input and mutated-payload digests, and dependency versions.
- `summary.json`: layer totals and per-case outcomes.
- `mutation_results_table.tex`: generated table for the paper.
- `mutation_detection.pdf` and `mutation_detection.svg`: vector heatmap.
- `mutation_detection.png`: rendered preview of the heatmap.

Use Python with NumPy, jsonschema, and matplotlib. The checked-in protocol and
results already exist; `--check` reruns all 272 checks and compares the output
without overwriting anything:

```bash
python run_experiment.py --check
python -m unittest -v test_experiment
python plot_results.py
```

For a fresh copy with no protocol or result files, `--freeze` creates the protocol
with exclusive creation, then `--run` executes and exclusively creates results.
Neither flag silently overwrites an existing experiment. A changed scientific
protocol belongs in a new versioned experiment directory. Plotting reads executed
summary data and never fills missing results. If source outcomes differ within a
case, plotting stops rather than hiding the disagreement.
