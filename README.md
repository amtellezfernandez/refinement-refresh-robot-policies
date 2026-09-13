# Refinement versus Refresh: Compute Allocation for Closed-Loop Robot Policies

The [submitted workshop version](submissions/robopad_2026_20260913/README.md)
is archived separately. `main.tex` and `output/` are the continuing development
version. See the [NeurIPS development plan](NEURIPS_DEVELOPMENT.md) for the next
experiments, evidence checks, and manuscript updates.

The paper asks how a frozen robot policy should divide inference computation
between prediction refinement and observation-conditioned refresh. It presents
the 12-configuration SmolVLA/LIBERO study, full-generation dense-arithmetic
calibration, paired quality/resource measurements, and allocation predictions.
Pilot and trace-validation evidence support the methods in the appendix.
The current manuscript has seven main-text pages, one reference page, and three
appendix pages. Historical resource-framework materials remain in the repository
but are not presented as empirical adaptation results.

The [transfer amendment](audit/TRANSFER_AMENDMENT_20260913.md) records additional
secondary contrasts and selector baselines after collection began. Discovery-only
diagnostics reproduce the original scores and summarize fixed-input timing; they
do not change the active experiment or establish transfer outcomes.

Start with the [revision plan](audit/NARRATIVE_PLAN.md),
[manuscript source](main.tex), [most recently built PDF](output/resource_optimal_post_training.pdf),
[revision audit](audit/REVISION_AUDIT.md),
[allocation literature notes](audit/ALLOCATION_LITERATURE.md), and
[bounded experiment design](experiments/EXPERIMENT_DESIGN.md).

The [relevance revision](audit/RELEVANCE_REVISION_20260913.md) records the current
structure and removal of obsolete planning and unrelated conceptual material.

## Evidence and next test

The September 13 manuscript snapshot adds the completed **480-attempt discovery
study**: four Spatial tasks, five states, two seeds, and twelve `(h,d)` settings.
All 87,364 dispatched actions reconcile with the episode records. Dense neural
arithmetic calibration measures 641.675 GFLOPs per native generation, with 82.1%
independent of denoising. The paper states excluded operations and sampled GPU
monitoring limits explicitly. Complete discovery records and generated tables,
figure, and audit are under
`experiments/executed/refinement_refresh_final_001/`.

The frozen recommendations are `(5,2)`, `(2,2)`, and `(1,2)` at quarter, half,
and full native compute-rate ceilings. The primary half-budget comparison is
complete on eight transfer tasks: `(2,2)` records 139/160 successes versus
122/160 for `(2,5)`, with descriptive paired interval [+4.4,+16.9] percentage
points. Secondary confirmation settings and the stale-input diagnostic remain
unfinished and are not claimed in the paper.

- The archive illustrates a cost-proxy ambiguity: three runs each record 19
  generations but different host inference time totals. This is a retrospective
  observation, not a replicated solver comparison.
- Offline reconciliation covers 28 completed archived artifacts (22 v1, four v2,
  two v3), including exact replay of 86 recorded actions in one repeated v2 run.
- The frozen mutation suite comprises 68 constructed trials / 272 checks.
  Structural checks reject 12/52 observable defects, aggregate checks 28/52,
  and full execution trace verification 52/52; all accept four controls. Twelve probes
  remain technically accepted but their requested claims need external evidence.
  This tests specified mechanisms, not natural error rates.
- The original hypothetical adaptation example keeps correction labor, training
  GPU-hours, and deployment seconds separate. A has lower values on both time
  measures; B uses less labor. It supports no crossover or aggregate GPU-time
  claim. Deployment totals of 120/135 seconds refer to 100 attempts.

The executed 80-attempt pilot compares prediction refinement and execution-prefix
length. It uses
two fixed Spatial tasks, five initial states, two policy seeds, and four `(h,d)`
conditions. A one-action prefix achieves higher success at both denoising levels
on this set. The principal `(5,5)` work reduction lowers success by 25 percentage
points relative to native `(1,10)`; halving denoising at `h=1` leaves the quality
difference unresolved. Concurrent GPU use excludes elapsed timings from clean
efficiency claims. The 320-attempt broader plan and a matched-cost comparison
remain unexecuted confirmation designs.
Those earlier designs are distinct from the new completed discovery and ongoing
confirmation campaign described above.

Trace v0.2 supports structure and declared-resource completeness checks. It cannot
infer omitted events or independently verify measurements. Generic traces and
deployment execution traces are separate formats; an integrated lifecycle recorder remains
future work. Historical resource-disclosure labels remain unverified.

## Artifacts

- `main.tex`, `references.bib`, `output/resource_optimal_post_training.pdf`: paper.
- `RESOURCE_SCALING_PROFILE.md`: comparison-specific reporting template and
  completed archive illustration.
- `post_training_trace.schema.json`, `validate_trace.py`, `test_trace.py`:
  generic trace v0.2 contract, validator and fault tests.
- `examples/`: explicitly synthetic hybrid and frozen-deployment examples;
  invalid update fixture. These values are not experimental results.
- `experiments/real_smolvla/ARCHIVE_AUDIT.md` and `archive_audit.json`:
  generated reconciliation with input digests, counters and limits.
- `experiments/instrument_validation/`: frozen protocol, executed trial-level
  results, generated table, and vector PDF/SVG heatmap; inputs remain unchanged.
- `experiments/execution/`: GPU benchmark/evaluation runners, completed-run
  analysis, and offline integrity tests.
- `experiments/executed/`: passed development gates, complete 80-attempt paired
  journals, runtime provenance, and reproduced paired analysis.
- `experiments/execution/port_discovery.py`: independently reconciles completed
  discovery journals and generates the new manuscript tables, figure, and audit;
  run `python experiments/execution/port_discovery.py` from this paper directory.
- `experiments/make_evaluation_plan.py`, `planned_evaluation.json`:
  reproducible proposed schedule; availability/holdout not certified.
- `experiments/make_pilot_plan.py`, `pilot_evaluation.json`:
  frozen schedule used for the executed focused pilot.
- `audit/BIBLIOGRAPHY_AUDIT.md`: 37-entry bibliography with primary-source metadata notes.
- `audit/comparison_extraction_template.csv`: blank comparison-level extraction
  template; `audit/pilot_audit.csv` preserves provisional legacy codes.
- `EXPERIMENT_AUDIT.md`, `experiments/temporal_reuse/`: historical withdrawn
  toy study and diagnostic, retained for traceability.

## Validation and build

Use Python 3.12+ with NumPy, jsonschema, and matplotlib. The mutation experiment
records NumPy 2.5.3 and jsonschema 4.26.0; exact replay checks preserve those
dependency versions in trial provenance. The optional original RNG-isolation test
additionally needs Torch. Offline runner tests use explicit test doubles where
appropriate; they do not invoke a learned model or GPU.

```bash
python -m pip install numpy==2.5.3 jsonschema==4.26.0 matplotlib
python verify_artifacts.py
# Or run individual components:
python -m unittest -v test_trace
python validate_trace.py examples/hybrid_system_trace.json
python validate_trace.py examples/deployment_trace.json
python -m unittest discover -s experiments -p test_evaluation_plan.py
python -m unittest discover -s experiments/real_smolvla -p 'test_*.py'
python experiments/real_smolvla/audit_archive.py --check
python experiments/instrument_validation/run_experiment.py --check
python -m unittest discover -s experiments/execution -p 'test_*.py'
latexmk -pdf -outdir=build main.tex
```

The integrated offline verification passes: 79 tests, two Torch-dependent skips,
28 archived results reconciled, and 68 mutation trials / 272 layer checks
reproduced. This command does not run model inference or launch GPU experiments.

The invalid fixture must return exit 1. `--require-complete` returns exit 2 for
structurally valid traces with missing/unknown declared resource rows; ordinary
structural validation returns 0. Completeness is relative to declared requirements,
not proof that authors disclosed every relevant cost.

Tectonic is an alternative build: `tectonic --keep-logs --outdir build main.tex`.
After visual review, copy `build/main.pdf` to the stable output path. The current
13-page PDF (six main-content pages, two reference pages, five appendix pages)
has been rebuilt and visually checked;
see [submission notes](SUBMISSION.md)
for workshop format and validation status.
