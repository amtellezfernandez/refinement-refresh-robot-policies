# Completed discovery port — September 13, 2026

Only completed discovery and frozen pre-confirmation predictions enter this
manuscript snapshot. Confirmation was running during the edit; no partial
confirmation outcomes were imported into the manuscript or used for selection.
The remote campaign and its frozen protocol were not changed by this port.

## Reconciliation

`experiments/tooling/port_discovery.py` independently checks the archived
480 episodes and 87,364 dispatched actions against the 40-block schedule, runtime
manifest, completion summary, keyed policy noise, producer/consumer action hashes,
terminal outcomes, and generation/denoising/vision/prefill/expert counts. Initial
states and observations match across all 12 conditions within each block.
It also verifies the discovery input hash in the frozen prediction, the calibrated
budget digest, and absence of overlap between discovery and confirmation tasks.
The generated JSON retains source SHA-256 digests.

Calibration and discovery logs contain 59 and 6,230 GPU-process samples,
respectively, with maximum gaps below 5.03 seconds. No other CUDA compute process
appears in those samples. Display processes and unsampled interference remain
possible; exclusive hardware ownership is not claimed.

## Scientific boundaries made explicit

- Calibration covers complete-generation **dense arithmetic**, including vision,
  prefill, and action expert. It excludes embedding lookups, normalization,
  activations, elementwise operations, and external preprocessing. It must not be
  described as exact total scalar computation.
- Grid results use 40 attempts per configuration and report all four tasks.
  Descriptive bootstrap intervals were specified after discovery and resample
  states within fixed tasks, preserving both seed repeats and all cells. They
  are not multiplicity-adjusted or powered noninferiority tests.
- Model CV log losses are 0.7048 versus 0.7067 for the best allocation and
  compute-only models. Their small difference is not evidence of strong
  predictive advantage. Every recommendation is an already measured grid point;
  the confirmation is task transfer, not an untested-configuration prediction.
- At quarter and half budgets the model-selected row is itself the empirical
  grid winner; the frozen code chooses the strongest distinct grid comparator.
  This detail is stated in the appendix, avoiding a claim that the comparator
  includes the recommendation itself.
- Confirmation coverage was fixed before discovery rather than selected using
  development-based precision. No noninferiority claim is made. Diagnostic
  outcomes, actual training-target coverage, second-policy transfer, and
  adaptation results remain unavailable.

## Manuscript changes

Abstract, introduction, limitations, and conclusion now reflect completed
discovery plus ongoing confirmation. A new main-text section and quality-versus-
compute figure introduce the calibration and frozen predictions. The appendix
contains per-task outcomes for all 12 cells, actual episode cost/time means,
the prediction table, selection details, and outstanding diagnostic limitations.
The duplicated compact pilot resource card was removed; the complete appendix
card remains. Earlier pilot results and evidence are retained separately.

Reproduction: run `python experiments/tooling/port_discovery.py`, then
`latexmk -pdf -outdir=build paper/main.tex`. This does not run GPU inference.

Editorial pass removes run-status language, dated snapshots, and scheduling
commentary from the manuscript. Results are framed as experimental measurements;
allocation predictions are explicitly distinguished from validated transfer
effects. Operational status remains in repository notes only.

Validation: 79 integrated offline tests pass with two dependency-related skips;
28 archived artifacts and 68 mutation trials reconcile. The independent discovery
audit passes on all 480 episodes and 87,364 actions. The PDF builds with no
undefined references or overfull boxes and has 18 pages (nine main text, two
references, seven appendix). Rendered pages were visually inspected, including
the new figure, results tables, and prediction table.
