# Resource disclosure coding guide (v1)

Read only the assigned sources and frozen comparison scope. Do not inspect other
readers' labels. Fill the eight resource cells for each of the 16 comparisons.

## Labels (nominal, not ordinal)

- Q: a quantitative amount with units, denominator and a boundary sufficient
  to recover this coordinate for the assigned comparison. Document any arithmetic.
- P: resource use is established, but disclosure is a proxy, incomplete amount,
  missing denominator or otherwise insufficiently bounded quantity.
- U: resource use is established, but no quantity is found after the documented
  source search. U requires evidence of use and a search record; it is not a
  synonym for an inaccessible paper.
- NA: the resource is outside the frozen comparison or demonstrably unused.
  Explain the applicability decision. Silence alone never establishes NA.

Leave `label` null if source access or applicability remains unresolved; explain
the blocker in `rationale`. Do not guess to reach 128 completed labels.

## Coordinates

- real_acquisition: physical robot interaction acquired for the scoped method;
  report transitions/trajectories with horizon or robot time and acquisition scope.
- human_labor: person-time spent on scoped demonstrations/interventions/labeling;
  intervention count alone is a proxy unless duration/person-time is recoverable.
- simulation: simulator-generated transitions; episodes require known lengths.
- wm_generation: learned-world-model-generated transitions; rollout counts require
  lengths and generation scope. Do not merge these with physics simulation.
- training_compute: scoped optimization compute with usable hardware/time or
  FLOPs. Device count or training steps alone usually lack a required denominator.
- deployment_compute: scoped model/planner calls with model/solver configuration
  and an episode/decision denominator, or measured compute with hardware and scope.
  Nominal simulator control Hz is not model-call frequency or inference latency.
- wall_time: elapsed time for the frozen stages, separating setup and execution;
  summed accelerator-hours are not automatically elapsed wall time.
- persistent_storage: persistent state footprint in bytes (or derivable element
  counts, dtype and dimensions), with lifetime and included components.

Do not sum unlike units or infer missing costs as zero. Shared/pretrained costs
are included only when the frozen boundary includes them. Distinguish reported
values from derived estimates; Q never means that two different papers are
automatically resource-matched.

## Required evidence for each resolved cell

Fill `source_locator` (document/version + page/section/table or stable code line),
`evidence` (a brief paraphrase or short excerpt), and `rationale` (why the label
follows under the frozen boundary). Fill `quantity`, `unit`, `denominator` for Q;
use strings so derivations and intervals remain explicit. Fill `search_log` for U
with inspected sections/supplement/code. Record contradictory disclosures rather
than selecting the convenient one. Do not use another audit's labels as evidence.
