# Final refinement–refresh campaign — 2026-09-12

Remote root: `/mnt/NVME1/projectx-robopad-proposal2` on `projectx@zetroc.fr:16641`.
Result directory: `results/refinement_refresh_final_001`.
Supervisor log: `results/refinement_refresh_final_001.launch.log`.

The user authorized prioritizing this study over Proposal 1 and requested an
unattended final campaign while disconnected for 12 hours. Proposal 1 process
41559 received SIGTERM and was confirmed stopped; its existing output was retained.
The new launcher uses nohup and independent subprocess sessions. Twelve hours is
an absence duration, not an episode truncation deadline.

`protocol.json` fixes the design before calibration. The stages are conventional
full-generation dense-arithmetic FLOP calibration and full-interval solver checks,
480 discovery attempts, grouped model selection, a frozen prediction artifact,
up to 1120 confirmation attempts, paired analysis, and 40 stale-input attempts.
The operator inventory documents arithmetic counted and excluded. Display GPU
processes remain; unrelated CUDA compute processes trigger failure and retention
of partial journals. This is not a guarantee against every source of contention.

Development: Spatial tasks 5,6 (pilot tasks) and 7,8 (added before new outcomes),
states 10–14, seeds 2000/2001. Confirmation: Object and Goal tasks 5–8, states
20–29, seeds 3000/3001. These are held out from selection; historical nonexposure
is not certified. Primary budget is 0.5 native; 0.25 and 1.0 are secondary.
Selection may recommend an existing grid point; the artifact labels this task
transfer rather than prediction of an untested configuration. No superiority is
presumed, and no noninferiority margin or power claim is imposed.

The stale diagnostic refreshes both images and proprioception every five ticks,
while generating every tick and executing action position zero at d=5. It uses
the same development state/seed identities as the fresh h1,d5 and h5,d5 rows.

## Later-position source audit

Pinned `modeling_smolvla.py` computes unreduced MSE over the full predicted action
chunk (line 723). Its policy forward method (lines 285–332) masks `action_is_pad`
and reduces over valid time/action entries. It does not explicitly suppress later
valid positions. Inference generates the full chunk and the queue executes its
first h actions in order. This establishes source semantics only. The archived
checkpoint alone cannot establish actual per-position training coverage, mask
frequencies, or adequacy of the original training data. Those remain unresolved.

## Inspecting the run

`completed.json` means all scheduled stages finished. `failure.json` explains a
failed stage; keep partial JSONL files. Each evaluator writes `summary.json` only
on completion. `frozen_prediction.json` must precede confirmation collection.
Do not relaunch into an existing result directory or pool partial attempts into
a completed sample. Calibration timing/profiling costs are separate from rollout
costs; `*_gpu.jsonl` logs CUDA compute-process checks every five seconds.

No second-family policy or adapted-checkpoint sweep is included in this frozen
campaign. Those require their own validated runtime and computation calibration.
