#!/usr/bin/env python3
"""Diagnose the legacy experiment; does not repair or validate its claims."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

import run_experiment as legacy


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate(weights, interval, episodes, zero):
    rng = np.random.default_rng(20260910)
    original = legacy.frozen_model
    if zero:
        legacy.frozen_model = lambda obs, weights: np.zeros(2, dtype=np.float32)
    fingerprints, outcomes = [], []
    try:
        for _ in range(episodes):
            fingerprints.append(digest(rng.bit_generator.state))
            outcomes.append(legacy.one_episode(rng, weights, interval, 160, 0.05))
    finally:
        legacy.frozen_model = original
    return fingerprints, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cpu')
    probes = np.random.default_rng(17).uniform(-2, 2, size=(256, 8))
    magnitudes, ablations, states = [], [], {}
    for padding in (0, 10, 80):
        weights = legacy.make_model(device, padding)
        outputs = np.array([legacy.frozen_model(obs, weights) for obs in probes])
        magnitudes.append({'padding': padding, 'max_abs_model_output': float(abs(outputs).max()),
                           'mean_output_l2': float(np.linalg.norm(outputs, axis=1).mean()),
                           'padding_matrix_spectral_norm': float(torch.linalg.matrix_norm(weights[6], 2))})
    weights = legacy.make_model(device, 10)
    for interval in (1, 2, 4, 8, 16):
        fingerprints, outcomes = evaluate(weights, interval, args.episodes, False)
        zero_fingerprints, zero_outcomes = evaluate(weights, interval, args.episodes, True)
        states[interval] = fingerprints
        ablations.append({'interval': interval, 'episodes': args.episodes,
                          'identical_outcome_tuples_with_zero_model': sum(a == b for a, b in zip(outcomes, zero_outcomes)),
                          'identical_episode_rng_states_with_zero_model': sum(a == b for a, b in zip(fingerprints, zero_fingerprints))})
    pairing = [{'interval': k, 'rng_state_matches_vs_interval_1': sum(a == b for a, b in zip(states[1], v)),
                'episodes': args.episodes, 'first_four_rng_states': v[:4]} for k, v in states.items()]
    result = {'status': 'invalid_as_evidence_for_foundation_model_efficiency',
              'source_sha256': hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
              'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
              'device': str(device), 'model_output_probes': magnitudes, 'zero_model_ablation': ablations,
              'episode_pairing': pairing,
              'limits': 'CPU diagnostic on 100 episodes per condition; outcome identity is not trajectory identity. Original GPU timing is not remeasured.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
