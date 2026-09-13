#!/usr/bin/env python3
"""Controlled temporal-reuse experiment for the Proposal 2 resource claim.

This is intentionally a simulator-only instrumentation experiment, not a VLA
benchmark.  It holds the controller, task seeds, control frequency, horizon,
and model weights fixed while changing only the interval between frozen-model
invocations.  The output is a JSON ledger suitable for the paper appendix.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def make_model(device: torch.device, padding: int) -> tuple[torch.Tensor, ...]:
    """Create deterministic frozen weights and a compute-padding matrix."""
    g = torch.Generator(device="cpu").manual_seed(20260910)
    w1 = torch.randn((8, 64), generator=g, dtype=torch.float32).to(device) * 0.25
    b1 = torch.zeros(64, dtype=torch.float32, device=device)
    w2 = torch.randn((64, 64), generator=g, dtype=torch.float32).to(device) * 0.15
    b2 = torch.zeros(64, dtype=torch.float32, device=device)
    w3 = torch.randn((64, 2), generator=g, dtype=torch.float32).to(device) * 0.12
    b3 = torch.zeros(2, dtype=torch.float32, device=device)
    pad = torch.randn((64, 64), generator=g, dtype=torch.float32).to(device) * 0.01
    return w1, b1, w2, b2, w3, b3, pad, padding


def frozen_model(obs: np.ndarray, weights: tuple[torch.Tensor, ...]) -> np.ndarray:
    """A deterministic frozen model with fixed compute padding."""
    w1, b1, w2, b2, w3, b3, pad, padding = weights
    x = torch.as_tensor(obs, dtype=torch.float32, device=w1.device).unsqueeze(0)
    h = torch.tanh(x @ w1 + b1)
    for _ in range(padding):
        h = torch.tanh(h @ pad)
    h = torch.tanh(h @ w2 + b2)
    y = torch.tanh(h @ w3 + b3).squeeze(0)
    return y.detach().cpu().numpy()


def one_episode(rng: np.random.Generator, weights: tuple[torch.Tensor, ...], interval: int,
                horizon: int, control_dt: float) -> tuple[bool, bool, int, int, float]:
    pos = rng.uniform(-0.82, 0.82, size=2)
    goal = rng.uniform(-0.82, 0.82, size=2)
    if np.linalg.norm(goal - pos) < 0.5:
        goal = np.array([0.78, -0.78]) if pos[0] < 0 else np.array([-0.78, 0.78])
    obstacle = rng.uniform(-0.35, 0.35, size=2)
    phase = rng.uniform(0, 2 * math.pi)
    cached = np.zeros(2, dtype=np.float32)
    calls = 0
    collision = False
    sim_time = 0.0
    for step in range(horizon):
        # A moving obstacle makes stale actions materially different from fresh ones.
        obstacle = obstacle + np.array([0.012 * math.sin(phase + 0.11 * step),
                                         0.012 * math.cos(phase + 0.09 * step)])
        obstacle = np.clip(obstacle, -0.72, 0.72)
        if step % interval == 0:
            delta = goal - pos
            away = pos - obstacle
            d = max(float(np.linalg.norm(away)), 1e-3)
            repulsion = away / (d ** 3) * 0.035
            obs = np.concatenate([pos, goal, obstacle, np.array([d, math.sin(phase), math.cos(phase)])])
            # Eight features: position(2), goal(2), obstacle(2), distance, phase proxy.
            obs = np.asarray([pos[0], pos[1], goal[0], goal[1], obstacle[0], obstacle[1], d, math.sin(phase + step * 0.02)])
            raw = frozen_model(obs, weights)
            action = raw + 0.8 * delta + repulsion
            n = max(float(np.linalg.norm(action)), 1e-6)
            cached = (action / n * 0.075).astype(np.float32)
            calls += 1
        # Fixed-rate control; only the frozen-model invocation schedule changes.
        pos = pos + cached + rng.normal(0, 0.0035, size=2)
        sim_time += control_dt
        if np.linalg.norm(pos - obstacle) < 0.095:
            collision = True
            break
        if np.linalg.norm(pos - goal) < 0.085:
            return True, collision, step + 1, calls, sim_time
    return False, collision, horizon if not collision else step + 1, calls, sim_time


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = make_model(device, args.padding)
    if device.type == "cuda":
        torch.cuda.synchronize()
    rows = []
    for interval in args.intervals:
        rng = np.random.default_rng(args.seed)
        successes = collisions = steps = calls = 0
        start = time.perf_counter()
        for _ in range(args.episodes):
            ok, hit, n_steps, n_calls, _ = one_episode(rng, weights, interval, args.horizon, args.control_dt)
            successes += int(ok)
            collisions += int(hit)
            steps += n_steps
            calls += n_calls
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append({
            "reuse_interval_control_steps": interval,
            "control_hz": 1.0 / args.control_dt,
            "episodes": args.episodes,
            "success_rate": successes / args.episodes,
            "collision_rate": collisions / args.episodes,
            "mean_control_steps": steps / args.episodes,
            "mean_model_calls": calls / args.episodes,
            "model_calls_total": calls,
            "wall_seconds": elapsed,
            "wall_ms_per_episode": 1000.0 * elapsed / args.episodes,
            "device": str(device),
            "model_padding_matmuls": args.padding,
            "seed": args.seed,
        })
    return {
        "experiment": "controlled_historical_temporal_reuse",
        "claim_tested": "fixed control frequency does not determine foundation-model invocation cost",
        "disclaimer": "Synthetic 2D moving-obstacle simulator with a frozen deterministic surrogate model; not a VLA benchmark.",
        "common": {"horizon": args.horizon, "control_dt": args.control_dt, "episodes": args.episodes},
        "results": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=160)
    p.add_argument("--control-dt", type=float, default=0.05)
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--padding", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
