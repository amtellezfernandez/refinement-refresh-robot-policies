# Proposal 2: refinement, refresh, and supported deployment choices

Revision: 2026-09-12. **Prospective GPU design, not results or preregistration.** The
historical toy, v1 pilot, and v2 smoke test were already observed. This design
supersedes the earlier 600-development/4,800-confirmatory sweep and speculative
scaling agenda. Archive results remain unchanged. See
[recomputed archive audit](real_smolvla/ARCHIVE_AUDIT.md).

## Research question and defensible claims

For a fixed deployment budget, is computation more useful for refining an action
prediction or for refreshing it using a new observation? The current paper supplies
measurement infrastructure for this question. Its frozen-policy experiment tests
deployment choices; it does not test an adaptation algorithm or lifecycle crossover.

The allocation question is local to the four declared deployment configurations:
which comparisons are supported when the resource limit uses measured deployment
cost instead of generation counts? These configurations are not automatically
iso-compute; per-generation work and trajectory length can both vary. Report
agreement, disagreement, and unresolved comparisons. This is not a scaling-law
fit. Before making a budget-feasibility or preserved-quality recommendation,
freeze the relevant budgets, clock boundaries, quality requirement, and precision
criterion. Those application thresholds are not supplied by the archived results.

The [deployment decision protocol](DEPLOYMENT_DECISION_PROTOCOL.md) separates the
initial four-cell study from a matched-cost extension and confirmation evaluation.
It specifies the proposed principal comparison, quality and cost requirements,
and unresolved outputs. Numeric application thresholds are still unset; this is
not a frozen confirmatory analysis or a validated configuration selector.

Four separately assessable targets:

1. **Correctness:** recorded counts/actions/outcomes reconcile with independently
   collected driver records; injected inconsistencies are detected.
2. **Overhead:** measure the effect of instrumentation on the same computation.
   Report its distribution instead of assuming tracing is free.
3. **Information recovery:** on the same rollouts, quantify which resource queries are
   recoverable from progressively richer summaries. Report whether richer
   accounting changes any supported comparison, including a null result.
4. **Decision utility (requires the additional decision contract):** compare
   generation-based and measured-cost choices under the same quality requirement,
   then assess chosen configurations on confirmation data. Agreement is a valid
   result. Query recovery alone does not establish better choices.

Do not promote predictable `ceil(episode_steps / h)` call reductions into a new
algorithm, nor equal observed success into quality equivalence. No scaling
exponent, broad LIBERO ranking, safety guarantee, or two-point noninferiority
conclusion is an endpoint of this bounded study.

## Evidence available now

- Legacy v1: 22 completed artifacts; aggregate outcomes and counts can be
  reconciled, but initial-state/randomness pairing and action replay are unverified.
- V2: four completed artifacts on Spatial task 0, state 0, seed 1000. Counts are
  92/92, 66/14, 86/9, and repeated 86/9 selections/generations. All initial-state
  hashes agree; the repeated h=10 condition matches all 86 dispatched actions.
- Twenty warm-up generations per v2 job are separate from rollout generations;
  total `eval_s` still includes warm-up and other pipeline work.
- No completed broad h=1 reference is archived. No independent episode terminal
  outcome stream, runtime lock, hardware metadata, or checkpoint hash is embedded
  in each v2 result. These cannot be reconstructed by editing the JSON.
- The old 16-paper resource codes are provisional; bibliography verification
  does not revalidate their values or applicability.
- A separate frozen offline mechanism check has now been executed: 52 observable
  constructed corruptions, four untouched controls and 12 probes requiring
  unavailable external evidence. See `instrument_validation/README.md`; these
  are validation cases, not new learned-policy rollouts.

## Gate 0: freeze a reproducible runtime

Use `HuggingFaceVLA/smolvla_libero`, model revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, with strict weight loading. Record
the resolved model/config/normalization file hashes, LeRobot/LIBERO/MuJoCo commits,
package lock, GPU/driver/CUDA versions, cameras, action mode, task horizon and
initial-state file hashes in every run manifest. The checkpoint predicts 50
actions; the native deployment is h=1, d=10. Verify these settings in the loaded
config; refuse silent defaults or missing/random weights.

Historical server capacity is not a current hardware reservation. The previous
project directory was `/mnt/NVME1/projectx_robopad/`; an earlier local setup targeted
a WSL RTX 5070 Laptop GPU. Neither describes a verified current reservation.
On 2026-09-12, the current Linux host's 8 GB RTX 2000 Ada Laptop GPU passed a CUDA
matrix-multiplication check in its existing environment. The pinned source files
were verified and a separate Python 3.12 environment created. The requested
Torch 2.8 runtime remains uninstalled after bounded download retries. Tokenizer,
configuration, and normalization files are available, but the model file failed
its publisher SHA-256 check and was quarantined. A fresh HTTP transfer also timed
out; no verified checkpoint or revision marker is present. This is setup progress,
not a model-loading or policy-validity result.
See [the current revision audit](../audit/REVISION_AUDIT.md) for the precise scope.

References: [checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero),
[official evaluation guide](https://huggingface.co/docs/lerobot/libero), and
[reported reproduction issue](https://github.com/huggingface/lerobot/issues/3264).
The issue is a diagnostic lead, not proof that a particular local run is invalid.

## Gate 1: instrument correctness before performance collection

Use development task IDs 0--4 only. Never tune against the reserved evaluation
tasks below. Preserve an unmodified upstream evaluator at the same pinned revision.

1. Compare upstream and instrumented actions, reset state and outcomes with
   identical keyed randomness. Declare exact equality or numerical tolerance
   before comparison. Same-condition replay alone does not check upstream parity.
2. Record deterministic reset state and separate environment/policy RNG streams.
   Key noise by `(suite, task, state_id, policy_seed, simulation_tick)`, never
   the number of previous policy calls. Reset queues/caches at each episode.
3. Check image-shuffle and zero-action diagnostics for policy connectivity.
   Interpret task-appropriate changes rather than requiring every control to fail.
   Exclude these diagnostics from performance tables.
4. Record terminal outcome, truncation cause and actual episode ID in an
   independent driver stream. Match these to evaluator summaries; task-level
   success agreement is weaker and must be labeled as such.
5. Count policy selections, chunk wrapper calls, actual model-entry calls,
   encoder calls, and action-expert/solver evaluations separately. Reconcile
   actual dispatches with proposed actions and counts with an independent
   profiler on a subset. A single `sample_actions` call is not an internal NFE.
6. Inject missing/duplicate model events, changed action bytes, orphan IDs,
   inconsistent outcomes, incomplete journals, missing units, nonfinite costs,
   and cached outputs mislabeled as new predictions. Schema validation alone
   cannot detect a syntactically valid omitted event: require driver totals.
7. Retain all original corrupted fixtures and report which layer detected each
   fault. Include an intentionally unmeasured field that remains unknown.

Stop and diagnose if baseline parity, pairing, counter reconciliation or policy
connectivity fails. No amount of later success-rate data repairs these gates.

## Experiment A: timing boundaries and tracing overhead

Save a fixed development observation. For each of two solver settings (d=5,10),
compare additional model probes on/off: four cells per block. Run 20 warm-up
predictions then 100 measured predictions, resetting queues so
every timed operation is a real generation. Repeat five blocks with condition
order randomized, paired instrumentation-on/off runs, and identical inputs and
policy noise. This is a microbenchmark, not closed-loop success evidence.

Measure outer host wall time, prediction host latency, CUDA-event stream elapsed
time and peak allocated/reserved memory separately. Record preprocessing,
rendering, environment stepping, initialization and warm-up in their own boundaries.
CUDA-event time is neither kernel occupancy nor energy. Do not add overlapping
parent/child durations. Report median and tail call latency with block variability;
100 calls from one block are not 100 independent hardware replicates.

Both modes retain common outer timing and output collection. Report overhead of
the additional model probes `(on - off) / off` within matched
blocks. Investigate >5% median overhead as an engineering diagnostic; retain
measurements regardless of threshold and never subtract overhead from policy
results to invent an unobserved runtime. No energy claim without measured power.

Use measured wall cost, including startup/warm-up and a recorded reserve, to
estimate the full benchmark allocation. If it does not fit, reduce a balanced
design and freeze a new manifest **before** evaluating reserved task/state IDs.
Do not stop after favorable test outcomes.

## Experiment B: four-condition closed-loop benchmark

| Item | Locked design |
|---|---|
| Refresh prefix h | 1, 5 successive predicted actions |
| Denoising steps d | 5, 10; native reference is (1,10) |
| Suites | Spatial, Object, Goal, LIBERO-10 |
| Reserved task IDs | 5 and 6 within each suite |
| Initial-state IDs | 10--14 inclusive |
| Policy seeds | 2000 and 2001, independently keyed by tick |
| Total | 8 tasks x 5 states x 2 seeds x 4 conditions = 320 attempts |
| Attempts per configuration | 80; the focused 80-attempt plan has 20 per configuration |
| Target population | Exactly the declared eight tasks, equally weighted |

Check IDs exist and were not used in **any** historical/development run, including
unarchived server runs. The proposed IDs are not certified held-out merely by
appearing here. If overlap/unavailability exists, amend and refreeze before seeing
new outcomes. **Execution amendment, before any new policy outcomes:** unarchived
server use cannot currently be verified. Proceed on the explicitly named fixed
task set while recording `not_certified_against_unarchived_runs`; do not call it
an independently held-out test or make confirmatory generalization claims.
`make_evaluation_plan.py` emits the proposed balanced schedule and
digest; its status remains `requires_runtime_and_holdout_verification` until those
external gates are documented. It does not launch or certify experiments.

One block is a `(suite,task,state_id,policy_seed)` shared across all four cells.
Randomize block order and condition order using a recorded scheduling seed. Preserve
all attempts, failure logs, actual state hashes, policy-noise keys, timestamps,
terminal outcomes, consumed chunk indexes, prediction origin ticks and cache
invalidation. A process crash is missing evidence, not policy failure; report it
and rerun the identical identity with a separate attempt ID. Never overwrite it.
Final comparisons require complete blocks; report all incomplete blocks and any
budget-driven truncation separately. Do not discard bad outcomes because timing
was contaminated; flag timing and repeat timing separately.

Offline synchronous simulation measures simulator steps and host throughput.
It does not establish a real-time deadline or safety. Report action age in ticks
as an instrumented schedule measure. A future asynchronous experiment needs
independent physics, timestamped observations, declared deadlines/fallbacks, and
disturbance schedules before responsiveness claims are possible.

## Analysis locked before new outcomes

Primary accounting queries on exactly the same records:

| Query | Success only | Success + generation counts | Full reconciled ledger |
|---|---|---|---|
| Task success / attempts | Recoverable | Recoverable | Recoverable |
| Actual neural generations | Unknown | Recoverable | Recoverable |
| Internal neural evaluations | Unknown | Unknown without solver instrumentation | Recoverable if independently counted |
| Outer rollout wall cost | Unknown | Unknown | Recoverable if its boundary is measured |
| Cost per attempt / total cost per success | Unknown | Calls only, if attempts known | Measured coordinates only |

This table is the protocol's prediction, not an observed reader-study result.
For each query return a value with unit/boundary, `unknown`, or an explicit
inconsistency. A supplied config value is not a measured count. Compare query
recovery with independent counters and report both numerator and denominator of
recoverable queries. An optional reader study requires actual independent readers;
an LLM recheck is not inter-annotator reliability.

Report success, executed steps, generations, internal work, outer wall cost,
latency, and memory for every cell and task. Preserve individual outcomes.
Show cost per attempt and all-attempt cost divided by successes (null when zero).
Do not pool the old pilot into the new experiment or treat calls as independent
replicates. Solver work at equal call count is a prespecified contrast; report
whether measured cost actually changes, including null or contrary outcomes.

For this fixed-task estimand, average seeds within state, states within task,
then weight eight tasks equally. Paired differences use the same task/state and
retain both policy seeds and all conditions together. A stratified paired
state-block bootstrap (10,000 draws, analysis seed 731) provides descriptive 95%
intervals conditional on this task set. Only five states/task makes tail intervals
unstable; report raw paired outcomes and sensitivity, not population guarantees.
No task-generalization inference or confirmatory superiority test is planned.
Any later noninferiority claim requires a justified margin, prospective precision
analysis using discordant pairs, sufficient independent confirmation data, and
multiplicity control when selecting across candidates. The descriptive state
bootstrap is not promoted to a confirmatory certificate, even if its interval
collapses on identical observed outcomes.
Failure to reject a difference never establishes equivalence.

Adverse contacts, safety, and recovery are **unmeasured** until task-appropriate
predicates are implemented and validated. Contact alone is not an adverse event.
Do not invent safety scores from the existing success ledgers.

## Literature experiment: repair the unit of analysis

Preserve the 16-paper CSV as legacy coding, with no completeness headline. Use
`audit/comparison_extraction_template.csv` to store a named contrast, exact version,
source passage/locator, value/unit/denominator/phase, measurement versus derivation,
and inherited versus incremental boundary. Define applicability before scoring.
Have two independent annotators extract a bounded six-comparison pilot; report
disagreements and adjudication. Do not replace missing sources with inferred zeros.
Score full-lifecycle and contrast-specific disclosure separately. Metadata fixes
in `references.bib` are not evidence that resource codes were verified.

## Later adaptation extension, not required for this workshop position paper

After validity gates, compare the same pretrained root with one compatible fixed
adaptation recipe, three training seeds, and native/locked alternative deployment
settings. Freeze data subset, validation split, optimizer steps and batch size;
record inherited demonstrations, unique examples, exposure and measured training
cost. Demonstration counts are not newly acquired robot interaction or human time.
Do not reuse test failures as training examples. All tuning runs are study cost,
separate from the selected recipe. No budget is promised until compatibility and
memory are measured.

At volume N, report adaptation cost plus N deployments in matching cost units.
Break-even `N = additional_adaptation_cost / deployment_saving` exists only for
positive saving under acceptable quality and compatible hardware/workloads; show
uncertainty and undefined cases. This does not exchange person-hours, transitions
and GPU time or establish an optimal lifecycle allocation.

## Required artifacts

Versioned protocol and manifest digest; runtime/config/checkpoint hashes; original
and instrumented source diff; immutable episode and event journals; independent
driver/profiler records; warm-up and overhead blocks; all attempts/exclusions;
per-task tables; analysis command and environment; fault-test outputs. Clearly
label archived observations, newly executed measurements and plans in every report.
