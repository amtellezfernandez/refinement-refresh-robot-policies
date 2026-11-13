# Refinement versus Refresh

Compute allocation for closed-loop robot policies: when should inference compute
refine an action prediction, and when should it obtain a fresh prediction?

[Accepted camera-ready](output/pdf/camera-ready-submission-2026-11-02.pdf) ·
[Original submitted version](release/submitted.pdf) ·
[Poster](output/pdf/refinement-refresh-zurich-poster.pdf) ·
[REVA worked example](https://github.com/maatinstitute/REVA/tree/main/examples/refinement_refresh) ·
[ICLR roadmap](ROADMAP.md) ·
[Camera-ready source](paper/camera-ready-2026-11-02.tex)

This repository is the publication and retained-evidence record for the study.
The maintained evaluation infrastructure and compact reproduction of the
published held-out totals now live in the
[REVA worked example](https://github.com/maatinstitute/REVA/tree/main/examples/refinement_refresh).
Historical runners and evidence remain here to preserve provenance; they are not
maintained separately from REVA.

Next target: **ICLR 2027 main conference**. Priorities are targeted second-policy
evidence and the existing observation-age diagnostic. These are planned
extensions, not completed results. The accepted PDF uses the NeurIPS workshop
template; [submission notes](SUBMISSION.md) track version history and author tasks.

## Current evidence

One frozen SmolVLA checkpoint, evaluated on LIBERO:

- **Discovery:** 480 attempts across 12 configurations and four Spatial tasks.
- **Confirmation:** 1,120 unique attempts across seven configurations and eight
  Goal/Object tasks excluded from selection. All originally planned identities
  are complete; two repeated records are excluded.
- **Primary result:** at a two-action prefix, two denoising steps achieve
  139/160 successes versus 122/160 for five steps. The paired difference is
  +10.6 percentage points, with a descriptive, task-conditional 95% interval
  of [+4.4, +16.9].
- **Allocation question:** the five-action, two-denoising-step setting achieves
  138/160 successes at 16.1 TFLOPs per attempt, versus 41.1 for the two-action
  setting. Their quality difference remains unresolved; this is not an
  equivalence or non-inferiority result.
- **Selector limitation:** the fitted model and a simple cost-aware rule select
  the same settings at all three tested ceilings.

These are 1,600 attempts across the two main studies; the earlier 80-attempt
pilot is separate. Arithmetic is reference-calibrated, not a complete scalar
operation count or a latency measurement. The separate stale-input diagnostic
has 20/40 successes with five-tick input refresh versus 28/40 matched fresh
inputs, a descriptive paired difference of -20.0 points with interval
[-32.5, -7.5]. Its journals lack per-tick supplied-input hashes and observation
ages, so it does not isolate every implementation detail.

The accepted camera-ready retains the allocation question and reports all three frozen
comparisons, the amended descriptive comparisons, task-level counts, resource
costs, and collection provenance. Its flow is allocation and fixed cost,
controlled study, refinement results, then descriptive refresh results;
extended fitting and measurement diagnostics are in the appendices. See the
[revision audit](review/REVISION_AUDIT.md) for verification and limitations.

## Repository map

| Location | Purpose |
| --- | --- |
| `paper/` | Manuscript sources, bibliography, and generated publication assets |
| `release/` | Frozen submitted PDF, supplement, checksums, and dated archive |
| `experiments/tooling/` | Evaluation runners, analysis tools, offline tests |
| `experiments/runs/` | Recorded experiments and reproducible derived evidence |
| `experiments/remote/` | Historical remote-status snapshots, not a live dashboard |
| `review/` | Evidence checks and protocol amendments |
| `review/archive/` | Superseded editorial notes |
| `output/` | Supplementary deliverables |

Historical trace and instrument-validation materials remain under
`experiments/real_smolvla/`, `experiments/instrument_validation/`, and
`experiments/historical_temporal_reuse/`. They are not new deployment results.

## Citation and reuse

Use GitHub's **Cite this repository** menu or [`CITATION.cff`](CITATION.cff) to
cite the paper and this version of its artifacts. A copyable entry is:

```bibtex
@article{tellezfernandez2026refinement,
  author  = {Tellez Fernandez, Alba Maria},
  title   = {Refinement versus Refresh: Deployment-Time Compute Allocation for Chunked Robot Policies},
  year    = {2026},
  note    = {NeurIPS 2026 Workshop on Post-Training Adaptation of Robot Foundation Models},
  url     = {https://github.com/maatinstitute/REVA/tree/main/examples/refinement_refresh}
}
```

The paper provides the scientific case study and retained evidence.
[REVA](https://github.com/maatinstitute/REVA) provides the maintained,
source-available evaluation infrastructure and the paper's worked reproduction.
Cite both when both the study and software support your work. See
[`LICENSE.md`](LICENSE.md) for code, manuscript, figure, and third-party
licensing boundaries.

## Reproduce and build

Use Python 3.12+ with NumPy and matplotlib; the broader legacy tests also require
jsonschema. LaTeX builds use latexmk, pdfLaTeX, and BibTeX. These commands do not
launch model inference or remote experiments.

```bash
# Replay all completed-confirmation journal checks and compare generated evidence.
python3 experiments/tooling/analyze_confirmation.py --check

# Offline runner and analysis tests.
python3 -m unittest discover -s experiments/tooling -p 'test_*.py'

# Render publication tables and vector figures from the retained reports.
python3 experiments/tooling/render_paper_visuals.py
python3 experiments/tooling/render_paper_visuals.py --check

# Build the accepted camera-ready manuscript.
cd paper
latexmk -pdf -outdir=../build/camera-ready camera-ready-2026-11-02.tex
cd ..
```

For `analyze_confirmation.py`, omit `--check` only to regenerate the confirmation
report and four evidence tables.
The separate presentation renderer owns `paper/generated/`: eight tables, two
vector figures, and a source/output hash manifest. It never refits, resamples,
or overwrites evidence in `experiments/runs/`. Its `--check` mode regenerates
assets in a temporary directory and compares every file, including the PDFs.
Figures use 10-point text at final print width. The presentation tests check
every plotted estimate and interval against the retained reports, table totals,
and rejection of clipped or undersized figure text.
The [run index](experiments/runs/README.md) identifies the compressed evidence
snapshot and supporting studies. No checkpoint download or GPU is needed.

The broader `python3 verify_artifacts.py` currently fails the legacy
instrument-validation frozen runner-hash check. Its frozen hashes have not been
rewritten; this limitation is separate from the passing confirmation audit.

The original submitted PDF is immutable. The accepted camera-ready and its source
are maintained separately. See [version/build notes](SUBMISSION.md) for artifact
digests and [the roadmap](ROADMAP.md) for the next research steps.
