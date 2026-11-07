# Pinned stock sampler: source-only audit

Read on 2026-11-07, without importing OpenPI, JAX or any native stack:

- Repository: Physical-Intelligence/openpi
- Commit: `215abfb217dbac7d5f1273282331b9b1866c0479`
- File: `src/openpi/models/pi0.py`
- SHA-256 of retrieved UTF-8 file bytes:
  `24b32d7c6ed5e409459afa8ca5af96a4dae09c2dcdf48e828907e411f4f12342`
- [Immutable source](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)

`Pi0.sample_actions` begins at line 217 and defaults num_steps to 10. The source
sets dt to -1 / num_steps (line 228), initializes the loop time at 1 (line 278),
passes that time to embed_suffix before predicting the velocity, advances the
action and time in the same step (line 271), and continues while time is at least
-dt / 2 (line 276). It uses jax.lax.while_loop (line 278).

This source read independently supports the recurrence implemented by
attestation.expected_time_values: start at 1, record the current time before
each Euler update, and add the constant negative step. It closes the earlier
literal source-inspection gap; it does not close native counter qualification.
The stopping condition is part of the stock reference and must be preserved in
any faithful wrapper comparison, not replaced solely by an assumed loop count.

The source does not explicitly force the scalar loop carry to float32. The
verifier's struct-based float32 recurrence therefore remains a native dtype
assumption to check under the pinned JAX/XLA configuration, including any x64
setting and conversion applied by the served wrapper. Source inspection cannot
prove the dtype of compiled operands, callback execution, numerical outputs,
executed evaluation counts, or preservation under instrumentation.

Still outstanding: stock-versus-wrapper native parity on frozen diagnostic
observations at native settings, preprocessing/action mapping and units,
independent executed-count controls, the attestation fire drill and timing cost
of deterministic flags. No model, simulator, GPU, parity case, E0, confirmation
identity, scientific outcome or coverage simulation was executed for this audit.
No qualification item is promoted; no execution permission or release changes.
