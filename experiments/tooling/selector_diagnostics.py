"""Discovery-only diagnostics; never reads confirmation outcomes or changes plans."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from last_campaign import GRID, features, fit, predict


def analyze(root):
    frozen = json.loads((root / 'frozen_prediction.json').read_text())
    calibration = json.loads((root / 'calibration.json').read_text())
    raw = (root / 'discovery/episodes.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == frozen['discovery_sha256']
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 480
    y = np.array([r['success'] for r in records], float)
    groups = np.array([r['episode_identity']['task_id'] for r in records])
    hd = [(r['h'], r['denoising_steps']) for r in records]
    costs = calibration['generation_flops']
    def loss(p):
        return float(-np.mean(y*np.log(np.clip(p, 1e-9, 1)) +
                             (1-y)*np.log(np.clip(1-p, 1e-9, 1))))
    intercept = np.zeros_like(y)
    folds = []
    for task in sorted(set(groups)):
        train = groups != task
        intercept[~train] = y[train].mean()
        folds.append(dict(task=int(task), training_prevalence=float(y[train].mean())))
    scores = dict(constant_half=loss(np.full_like(y, .5)),
                  fold_fitted_intercept=loss(intercept))
    for model in frozen['models']:
        x = features(hd, costs, model['kind'])
        cv = np.zeros_like(y)
        for task in sorted(set(groups)):
            train = groups != task
            cv[~train] = predict(x[~train], fit(x[train], y[train], model['ridge']))
        score = loss(cv)
        assert abs(score-model['grouped_log_loss']) < 1e-10
        scores[f"{model['kind']}_ridge_{model['ridge']}"] = score
    compute = min((m for m in frozen['models'] if m['kind'] == 'compute'),
                  key=lambda m: (m['grouped_log_loss'], m['ridge']))
    candidates = [(r['h'], r['d']) for r in frozen['candidates']]
    cp = dict(zip(candidates, predict(features(candidates, costs, 'compute'),
                                     np.array(compute['coefficients']))))
    rate = lambda c: costs[str(c[1])]/c[0]
    observed = {c: np.mean([r['success'] for r in records
                           if (r['h'], r['denoising_steps']) == c]) for c in GRID}
    choices = []
    for row in frozen['comparisons']:
        eligible = [c for c in candidates if rate(c) <= row['ceiling']*(1+1e-12)]
        grid = [c for c in GRID if c in eligible]
        choices.append(dict(budget=row['fraction'], learned=row['recommendation'],
            empirical=min(grid, key=lambda c: (-observed[c], rate(c), c)),
            simple=min(eligible, key=lambda c: (c[1], c[0])),
            compute_only=min(eligible, key=lambda c: (-cp[c], rate(c), c))))
    timing = []
    for d, rows in calibration['timings'].items():
        assert len(rows) == 60
        blocks = []
        for block in range(3):
            subset = [r for r in rows if r['block'] == block]
            assert len(subset) == 20
            blocks.append(dict(block=block,
                host_ms=float(np.mean([r['host_s']*1000 for r in subset])),
                cuda_ms=float(np.mean([r['cuda_ms'] for r in subset]))))
        timing.append(dict(d=int(d), gflops=costs[d]/1e9, blocks=blocks,
            host_ms=float(np.mean([r['host_s']*1000 for r in rows])),
            cuda_ms=float(np.mean([r['cuda_ms'] for r in rows]))))
    return dict(scope='Post-start amendment, discovery/calibration only; no confirmation outcomes read',
        input_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ['frozen_prediction.json', 'calibration.json', 'discovery/episodes.jsonl']},
        scoring='Mean natural-log binary cross-entropy; whole-task validation folds',
        scores=scores, intercept_folds=folds, selectors=choices,
        compute_model=compute, timings=sorted(timing, key=lambda r: r['d']),
        timing_scope='Full-generation intervals only; no component-level timing or component-attributed FLOP counts recorded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
