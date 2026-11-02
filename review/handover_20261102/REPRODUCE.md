# Reproduce the checks

## Package checksums (ZIP only)

From the `FINAL_HANDOVER` directory:

```bash
shasum -a 256 -c 09_SHA256SUMS.txt
```

On Linux, `sha256sum -c 09_SHA256SUMS.txt` is equivalent.

## Arithmetic recount

Needs Python 3 and no other packages or GPU. The archive is on `main`, at `experiments/runs/refinement_refresh_final_001/confirmation_completed.tar.gz`, 30 MB, and is not in the ZIP. From a clone of the repository:

```bash
shasum -a 256 experiments/runs/refinement_refresh_final_001/confirmation_completed.tar.gz
python3 review/handover_20261102/recount_confirmation.py experiments/runs/refinement_refresh_final_001/confirmation_completed.tar.gz
```

Expected: archive `36c1d5f2dcaef85664ec7a1bfe7409c849b3be17831566e783d24a00ea4e8df0`; `raw 1122 unique 1120`; (2,2) 139 of 160; (2,5) 122 of 160; `wins 24 losses 7`; `paired_difference_points 10.625`; `bootstrap_95_percent 4.375 16.875`; `primary_actions 50451 all_actions 171805`.

This checks arithmetic and record binding. It does not validate the instrument that wrote the journals.

## Schedules

```bash
python3 experiments/tooling/cross_policy/build_protocols.py --check
```

Expected: `checked 9 files`. This shows the files regenerate. It is not execution evidence.
