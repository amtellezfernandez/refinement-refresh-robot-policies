# MAC CPU NORMALIZATION CHECK ONLY

Overall preflight verdict: **PREFLIGHT_INCOMPLETE**. This is not evaluator parity, and not RTX or CUDA evidence. It does not validate denoising.

## Result

The pi0.5 policy loaded with the real `pi05_libero` weights on Mac mini CPU applies normalization identical to `norm_stats.json`.

| Check | Result |
|---|---|
| `norm_stats.json` SHA-256 | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`, the expected value |
| `Unnormalize` transforms found | 1 |
| Quantile mode | True |
| Eight fields equal to the file: `state` and `actions` x `mean`, `std`, `q01`, `q99` | All 8 true |

The comparison is exact, in float64, of the loaded policy's statistics against the file. Output: `mini_output.txt`.

## Denoising: INCOMPLETE, not validated

The job ended by itself at 10:22:41Z with exit 137 (killed); the cause is not established. Recorded before that: `action_out_proj` called 2 times at d=2 and 5 times at d=5. **d=10 did not run.** `result.json` still says `RUNNING`. These partial counts are not a validation.

## Provenance

| Item | Value |
|---|---|
| Host | Mac mini M4 Pro, macOS 26.2 arm64, `TFRT_CPU_0`; no GPU, RTX or CUDA |
| Checkpoint | Public `gs://openpi-assets/checkpoints/pi05_libero/`, 16 files, bucket sizes match. SHA-256 of all 16 match those recorded on 1 November in `experiments/runs/cross_policy_confirmation_v1/preflight_cpu_evidence/` (`ckpt_mini_sha256.txt`). Not included here |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Stack | Python 3.11.16, JAX 0.5.3, CPU |
| Environment deviation | `uv.lock` versions without the CUDA and NVIDIA packages and `triton`, which have no Mac wheels. `evdev` and `antlr4-python3-runtime` failed to install. Packages listed in the log were installed one at a time, without dependency resolution |
| Commands | `job_command.sh`, which contains the job as run; `check.py` |

## Resource-limit breach

Approved in advance: 3 threads, 24 GiB. Observed: about 27.4 GiB RSS and about 691% CPU; the thread flags did not hold. Warren extended the limits for this job after this was found. The job had ended before his later instruction to stop it, so no stop was needed.

## Limitations

- One Mac CPU environment, with a modified dependency set. Not the campaign host.
- Shows what the loaded policy holds. Action units in the simulator were not checked.
- No simulator, rollout, parity, first-action, reset, smoke, video, timing or state-hash work ran here.
- SmolVLA was not run.
- SmolVLA stays ARITHMETICALLY REPRODUCED — INSTRUMENT VALIDITY PENDING. The invalid pi0.5 campaign stays excluded.
- The frozen protocols, schedules, submitted PDF and 1 November records are unchanged.
- No checkpoint, secret or large file is in this folder.
