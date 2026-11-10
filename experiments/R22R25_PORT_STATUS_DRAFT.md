# R22/R25 BenchStudio operator port — draft only

This branch is a filtered port from Warren's implementation branch onto current RoboPAD main, for code review only.

**DO NOT MERGE OR RUN.** Protocol v3 remains unreleased, both native qualifications remain UNVERIFIED, required native state generation and Protocol E fixtures remain unresolved, and no campaign release or authorization is present. This port adds no permissions and does not execute simulations or consume identities.

The code boundary is intentional:
- BenchStudio supplies generic trusted process execution and service lifetime through `benchstudio.processes.ProcessCommand`.
- RoboPAD owns the R22/R25 recipe, native stack adapters, scientific checks, qualification records, and analysis.
- The BenchStudio integration is a thin installed pipeline over the same RoboPAD recipe; it does not create a parallel runner.

Included: operator recipe, native runtime/tooling dependencies, synthetic regression tests, protocol-v3 draft documents, and qualification records that explicitly remain UNVERIFIED.

Excluded: historical/generated campaign schedules, reserved identity populations, archived diagnostic/preflight evidence, release/authorization records, and CI workflow activation. No commands or tests were run while preparing this port.

This draft is for review of the integration boundary and code only. Any future merge, native qualification, state generation, campaign release, and campaign execution require their existing independent review and authorization.
