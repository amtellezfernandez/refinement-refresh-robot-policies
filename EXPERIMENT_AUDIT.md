# Historical validity audit: the earlier efficiency claim is withdrawn

This document preserves the earlier toy-experiment audit. For the current
manuscript revision, instrument fixes, and executed archive validation, start
with [the revision audit](audit/REVISION_AUDIT.md) and
[recomputed archive audit](experiments/real_smolvla/ARCHIVE_AUDIT.md).
The current [experiment design](experiments/EXPERIMENT_DESIGN.md) supersedes
the larger proposed sweep described below. No legacy run has been retroactively
paired or upgraded to a newer ledger version.

Date: 2026-09-10. Scope: Proposal 2's experiment, its numerical claims, and the
literature audit's measurement contract. Proposal 1 and AlpaSim are unchanged.

The previous experiment ran and produced the stored numbers, but its design does
not support the claimed foundation-model efficiency result. More runs of it
would increase precision around an inappropriate experiment. The paper's success
and speedup claims have been removed, and the original code and outputs remain
available for inspection. This audit includes new executed diagnostics; it is
not an independent second annotation of the literature corpus.

## Findings and evidence

| Issue | Direct evidence | Consequence |
| --- | --- | --- |
| Policy is not pretrained or trained | `make_model()` initializes every weight randomly; actions add hand-coded attraction and repulsion using simulator state | No robot foundation model is evaluated |
| Neural output is negligible | With the stored run's padding=10, max absolute output is 2.883e-11 across 256 observations; padding=80 gives zero | Compute is spent on an effectively irrelevant branch |
| Neural ablation leaves outcomes unchanged | Replacing output with zero preserves every recorded episode outcome tuple in 100/100 episodes at each of five intervals | Model compute has no observed effect on these diagnostic outcomes |
| Episodes are not paired | Only 2/100 start-of-episode RNG fingerprints match interval 1 at each other interval | Resetting a single global seed did not hold initial episodes fixed |
| Reuse changes a held action | The script has no action trajectory output; one cached vector is repeatedly applied | Does not evaluate AlpaSim trajectory reanchoring or VLA chunk execution |
| Nominal 20 Hz is not measured feedback frequency | `control_dt` only accumulates simulated time; neither movement nor execution is timed by it | The experiment cannot support a real-time controller claim |
| Quality is not preserved as a vector | Stored interval-8 run: collisions 0/1,000 to 72/1,000 | Similar success cannot establish equal adverse outcomes |
| Cheap failures reduce average cost | Rollouts end on collision or success; interval-8 mean steps 19.866 versus 32.423 | Per-attempt compute savings mix refresh savings with different episode duration |
| Timing is confounded | Artificial matrix padding, fixed condition order, no warm-up/timing repeats or device timeline | The measured ratio is not a VLA latency or throughput result |
| Aggregation prevents recovery | Only means/rates are saved, with no episode IDs, action streams, or step timestamps | Paired confidence intervals cannot be reconstructed from original outputs |

Executed evidence: [diagnostic JSON](experiments/temporal_reuse/audit_results.json)
and [reproducible diagnostic](experiments/temporal_reuse/audit_experiment.py).
The source checksum in the diagnostic corresponds to the unchanged original
script. Diagnostics ran on the server CPU using its existing PyTorch environment;
they do not remeasure the historical GPU timing. Outcome equality does not assert
bit-identical complete trajectories.

## What the original numbers actually permit

The reported success difference is 11 successes in 1,000 attempts per arm, or
+1.1 percentage points. Even an illustrative unpaired normal interval, ignoring
the experiment's larger validity problems, spans approximately -1.26 to +3.46
points. It does not demonstrate superiority or quality equivalence. A valid
replacement needs a declared noninferiority margin and paired observations.

The collision increase is 7.2 percentage points. This prevents calling the
quality vector equivalent regardless of the success-rate difference. Observing
zero baseline collisions is not proof that the true baseline risk is zero.

The mean runtime of 1.38 ms describes host time per accelerated synthetic
episode. The episode averages 19.866 nominal 50 ms steps, or about 0.993 simulated
seconds. These two clocks are different. No feedback-deadline evidence exists.
The 90.9% call reduction also includes shorter episodes; at a fixed horizon an
eight-step schedule trivially reduces scheduled calls to approximately one eighth.
That arithmetic by itself is not the scientific contribution of Proposal 2.

## Literature-audit issues

The existing CSV has 16 rows, but a paper is not a well-defined comparison unit.
It has no per-field source locator, stored value, unit, phase, or derivation, and
uses `Q` for quantities with different meanings. Deployment latency is not an
episode budget; total experiment GPU-hours are not necessarily one recipe's cost.
Demonstration counts do not measure teleoperator labor. The maximal checklist
therefore cannot establish that a paper's original comparison is irreproducible.

Recomputing completeness from the existing labels gives:

| Required fields | Passing rows | Meaning |
| --- | --- | --- |
| All eight | 0/16 | Full draft checklist |
| All except storage | 1/16 | Q-chunking |
| All except storage, human labor, wall time | 2/16 | Q-chunking, RoboFAC |
| Training and deployment only | 3/16 | FLASH, Q-chunking, RoboFAC |

These are sensitivity checks on existing codes, not new source-verified scores.
Every inference must also state applicability: e.g. human labor is currently
marked applicable in 8 rows and storage in 9, so the corresponding counts are
0/8 and 0/9, not evidence that all 16 methods consume either resource.

`summarize_audit.py` originally trusted the manually typed `fully_reconstructable`
column. The audit should compute that field and reject contradictions. A stronger
corpus requires independent annotation of versioned source passages and explicit
baseline/intervention boundaries before assigning missingness. Re-reading the
same table in this session is not inter-annotator reliability.

## Replacement and paper status

The [replacement design](experiments/EXPERIMENT_DESIGN.md) specifies:

1. A real public LIBERO-trained checkpoint and baseline reproduction gate.
2. A crossed refresh-interval / denoising-effort sweep with independent episode
   streams, held-out evaluation, adverse-outcome constraints, and uncertainty.
3. Independent trace-versus-driver resource reconciliation and missing-event
   fault injection, which directly tests the accounting contribution.
4. A comparison-level literature audit with sensitivity to accounting boundaries.

The server is sufficient to begin inference and profiling with the selected
checkpoint, subject to measured memory and compatibility checks. A GPU does not
remove the need for a trained policy, valid simulator interface, or a measured
baseline. The paper remains a research draft until those experiments are done.
