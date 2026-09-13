#!/usr/bin/env python3
"""Monitor the existing continuation, audit coverage, then run the frozen diagnostic."""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from last_campaign import DEV, SEEDS, gpu_processes, plan, run_child, write
from run_paired import require


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--parent-pid', type=int, required=True)
    args = ap.parse_args()
    c = args.root/'results/refinement_refresh_final_001'
    phases = ['confirmation', 'primary_deadline', 'confirmation_continuation_partial',
              'confirmation_continuation_rest']
    final = c/phases[-1]
    with (c/'post_submission_gpu.jsonl').open('x', buffering=1) as monitor:
        while True:
            monitor.write(json.dumps(dict(time=time.time(), compute_pids=gpu_processes()))+'\n')
            if (final/'summary.json').exists():
                break
            for phase in phases[2:]:
                require(not (c/phase/'failure.json').exists(), f'{phase} failed')
            os.kill(args.parent_pid, 0)
            stat = Path(f'/proc/{args.parent_pid}/stat').read_text().split()
            require(stat[2] != 'Z', 'continuation terminated before completion')
            time.sleep(5)
    expected = {r['planned_run_id']: r for r in json.loads((c/'confirmation_plan.json').read_text())['schedule']}
    kept, repeats, sources, reset = {}, [], {}, {}
    for phase in phases:
        for line in (c/phase/'episodes.jsonl').read_text().splitlines():
            r = json.loads(line)
            key = r['planned_run_id']
            require(key in expected, 'unexpected identity')
            p = expected[key]
            require(all(r[k] == p[k] for k in ('episode_identity','h','denoising_steps')), 'identity mismatch')
            if key in kept:
                repeats.append(dict(planned_run_id=key, excluded_run_id=r['run_id'], phase=phase))
                continue
            g = (r['steps']+r['h']-1)//r['h']
            require(r['counts'] == dict(generations=g, denoise_steps=g*r['denoising_steps'],
                    vision_encoder_calls=2*g, prefill_calls=g, action_expert_calls=g*r['denoising_steps']),
                    'episode neural counts differ')
            identity = json.dumps(r['episode_identity'], sort_keys=True)
            pair = (r['initial_state']['sha256'], r['initial_observation_sha256'])
            require(identity not in reset or reset[identity] == pair, 'cross-phase reset mismatch')
            reset[identity] = pair
            kept[key], sources[key] = r, phase
    require(set(kept) == set(expected) and len(kept) == 1120, 'incomplete confirmation coverage')
    counts = Counter((r['h'],r['denoising_steps']) for r in kept.values())
    require(len(counts) == 7 and set(counts.values()) == {160}, 'unbalanced conditions')
    write(c/'confirmation_coverage_audit.json', dict(unique_attempts=len(kept),
          phases=phases, excluded_repeats=repeats, sources=sources,
          checks=['planned identities','condition coverage','initial-state/observation pairing','episode neural counts'],
          pending_checks=['full action-journal reconciliation','runtime provenance comparison','paired statistical analysis']))
    with (c/'confirmation_canonical_episodes.jsonl').open('x') as f:
        for key in expected:
            f.write(json.dumps(kept[key])+'\n')
    print('COVERAGE AUDIT PASSED: 1120 unique confirmation attempts', flush=True)
    # The stale-input intervention was specified in the original campaign protocol.
    scripts = args.root/'paper/proposal_paper2/experiments/tooling'
    while gpu_processes():
        time.sleep(5)
    write(c/'stale_plan.json', plan(DEV, range(10,15), SEEDS, [(1,5)], 'stale'))
    import sys
    run_child(SimpleNamespace(output=c), 'stale', [sys.executable, str(scripts/'run_paired.py'),
        '--checkpoint', str(args.root/'runtime/checkpoint'), '--output', str(c/'stale'),
        '--plan', str(c/'stale_plan.json'), '--warmup', '20', '--observation-refresh', '5'])
    print('STALE DIAGNOSTIC COMPLETE; full result analysis remains required', flush=True)


if __name__ == '__main__':
    main()
