# Withdrawn toy experiment

The results in `results.json` and `results_1000.json` are preserved historical
outputs, **not valid evidence of foundation-model efficiency**. Both files contain
the same 1,000-episode-per-condition run; they are not independent replications.
The original script is intentionally unchanged so its audit remains reproducible.

The 2026-09-10 audit found:

- The network uses random weights; no pretrained or learned policy is loaded.
- Ten contraction layers reduce maximum absolute neural output to about
  `2.88e-11` on 256 probes; the default 80 layers produce exactly zero on those
  probes. A handwritten attractive/repulsive controller determines actions.
- Replacing neural output with zero preserves all five recorded outcome fields
  in 100/100 episodes at each of five refresh intervals (500 comparisons). This
  checks outcome identity, not complete trajectory identity.
- Only 2/100 episode-start RNG states match the interval-1 baseline at each other
  interval. Variable episode lengths consume different amounts of shared random
  noise and change subsequent initial conditions. A shared seed is insufficient.
- Reuse holds the last action; there is no trajectory prediction or reanchoring.
- `control_dt` increments a counter; it does not scale dynamics or enforce a
  wall-clock control deadline. The claimed 20 Hz is a nominal simulation label.
- Original collision rates increase from 0/1,000 to 72/1,000 at interval 8.
  Similar success rates would not establish equivalent safety even with valid
  pairing. Early failure also makes attempted episodes cheaper.
- GPU timings lack warm-up, randomized order, repeated timing blocks, CUDA event
  measurements, and recorded interference. Artificial padding is not VLA work.

Run the diagnostic from this directory in an environment with NumPy and PyTorch:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_experiment.py --output audit_results.json
```

`audit_results.json` records the original source SHA-256, environment versions,
numerical probes, ablations, and RNG-state fingerprints. It was produced on the
SSH server CPU using PyTorch 2.6.0+cu124 and NumPy 1.26.4. It does not remeasure
the original GPU timings. See [the replacement design](../EXPERIMENT_DESIGN.md).
