# Recorded experiments

The main paper studies one frozen SmolVLA checkpoint, revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Each runtime manifest records
the pinned LeRobot sources and evaluation controls.

## Main campaign: refinement_refresh_final_001

- `discovery/`: 480 attempts, 12 configurations, four Spatial tasks;
  all 87,364 dispatched actions reconcile.
- `calibration.json`, frozen plans and predictions, and language-shape audits:
  resource accounting and selection provenance.
- `confirmation_completed.tar.gz`: completed-confirmation snapshot from the
  experiment host, including all four phase journals/manifests, original plan,
  scheduling/deduplication amendments, canonical records, coverage audit,
  calibration, and frozen predictions.
- `paper_analysis/`: reproducible discovery and confirmation reports, figures,
  and manuscript tables. The full-confirmation report records input, snapshot,
  analyzer, and validator hashes.

Confirmation contains **1,120 unique completed attempts**: seven settings on
eight tasks, ten states per task, and two seeds. The four raw phase counts are
799, 92, 6, and 225, totaling 1,122. Two repeated completed identities have
identical outcomes and action streams; documented phase precedence excludes the
later records. The original journal also retains a trailing incomplete attempt
with ten action events and no outcome. It is not an additional completed attempt.
Continuation completes the original schedule; it is not an independent cohort.

The audit reconciles all completed raw journals and 171,805 canonical actions.
It reproduces the unchanged primary result, all three original frozen tests,
four amended descriptive contrasts, task counts, and all-attempt resource costs.

From the repository root, without a GPU or model download:

```bash
python3 experiments/tooling/analyze_confirmation.py --check
```

Omitting `--check` regenerates five files in the existing `paper_analysis/`
directory. The input archive is read-only; extraction uses an automatically
removed temporary directory. Earlier discovery evidence can be reproduced with
`python3 experiments/tooling/port_discovery.py`.

Success intervals resample states within eight fixed tasks, retain paired seeds,
and weight tasks equally. Original comparisons use seed 20260912; amended
descriptive contrasts use 20260913. There is no multiplicity correction or
prespecified quality-preservation margin. Arithmetic uses the length-20 reference
calibration. GPU-monitoring gaps exclude clean aggregate elapsed-efficiency
claims. The separate stale-input diagnostic is complete on 40 discovery
identities: 20/40 successes with five-tick input refresh versus 28/40 matched
fresh-input successes, paired difference -20.0 percentage points and descriptive
task-conditional state-bootstrap interval [-32.5, -7.5]. Initial state and
observation hashes match all 40 pairs. Its step journals do not record supplied
input hashes or observation ages, so the result is a mechanism diagnostic rather
than a complete causal decomposition.

## Earlier pilot and validity checks

- `development_gate_001/`: deterministic action checks, work counters,
  profiler reconciliation, image perturbations, and zero-action control.
- `development_parity_001/`: four-condition development block and exact
  controlled-noise parity with the pinned upstream evaluator.
- `paired_pilot_001/`: frozen 80-attempt schedule and complete journals for two
  Spatial tasks, five states, two seeds, and four configurations.
- `analysis_001/`: paired pilot analysis and information-recovery audit from
  [analyze_execution.py](../tooling/analyze_execution.py).

Pilot timings overlapped another GPU process and are not clean efficiency
evidence. Pilot intervals condition on two tasks and do not establish
equivalence, non-inferiority, or task transfer.
