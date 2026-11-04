# Fixture provenance

`fixtures.jsonl` holds 20 synthetic trajectories, recorded by `replay/make_fixtures.py`, on tasks outside every paper
task set: `libero_object` task 2 and `libero_goal` task 1. None of them is paper evidence.

- 16 trajectories use action sequences derived from LIBERO human demonstrations,
  `yifengzhu-hf/LIBERO-datasets` at revision `f13aa24a3da8c43c7225569f28c562979fa0e35a` (Apache-2.0):
  - `libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5` (sha256 `45960606…c5ac`);
  - `libero_goal/put_the_bowl_on_the_stove_demo.hdf5` (sha256 `ff77d3bf…e99`).
  Demos 0 to 7 of each file were played open-loop from each demo's own first state, after the wrapper's 10 settle steps.
  Of these, 15 succeeded and 1 failed (object demo 0).
- 4 trajectories are synthetic failures: all-zero actions, and seeded uniform random actions.

Success labels are what the LIBERO predicate returned on the recording host. They are not the original demonstration labels.
