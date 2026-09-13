# Allocation literature: source claims and reporting requirements

Checked 2026-09-12 for the Proposal 2 revision starting at `a51a4af`.
The four studies motivate measurement requirements. They do not validate our
ledger, and our archive does not validate or refute their scaling relationships.

## Pearce et al.: representation changes the counting unit

Citation: *Scaling Laws for Pre-training Agents and World Models*, ICML 2025,
PMLR 267:48542–48562. The [publisher record](https://proceedings.mlr.press/v267/pearce25a.html)
provides the published title, six authors, venue, year, and pagination.
The [author manuscript, Section 3.4](https://arxiv.org/html/2411.04434v2#S3.SS4)
defines data as transformer inputs, with multiple inputs per observation/action
pair in tokenized models and one in BC-CNN. Sections 3.2–3.4 distinguish frozen
tokenizers and trainable components. The study examines generative pretraining
loss; it does not establish a universal robotic return law.

Our application: record transitions, represented inputs, repeated exposure,
tokenizer/architecture identity, parameter inclusion, and the quality objective.
This is an interpretation of what the study's definitions require for a named
comparison, not an accusation of incorrect counting.

## Baniodeh et al.: training and deployment in a controlled model family

Citation: *Scaling Laws of Motion Forecasting and Planning -- Technical Report*,
[arXiv:2506.08228v2](https://arxiv.org/abs/2506.08228v2), revised 2025-09-08.
The record supplies the 17-author list and title. The
[full manuscript](https://arxiv.org/html/2506.08228v2) studies training allocation,
inference sampling, and closed-loop planning (Section 5.2). Its closed-loop
analysis specifies route-conditioning adaptation, scenario duration, planning
selection, and failure criteria.

Our application: name the model family, adaptation and inference boundaries,
evaluation regime, and quality measure. The paper is direct prior work on
embodied allocation; we claim neither the first such study nor the first use
of closed-loop metrics.

## Kwok et al.: candidate count, policy work, and action verification

Citation: *RoboMonkey: Scaling Test-Time Sampling and Verification for
Vision-Language-Action Models*, CoRL 2025, PMLR 305:3200–3217.
The [publisher record](https://proceedings.mlr.press/v305/kwok25a.html) supplies
the eight authors and final metadata. Its abstract distinguishes initial policy
samples, Gaussian proposals, and a learned action verifier. The
[author manuscript, Sections 4.4 and 5.4 and Appendix G](https://arxiv.org/html/2506.17811v1)
describes proposal generation, latency, and serving with caching and batching.

Our application: distinguish policy calls, proposal production, verifier calls,
verifier adaptation, and measured time. RoboMonkey already considers deployment
cost. Its action verifier scores actions; our validator checks measurement
records. Our existing SmolVLA ledger does not implement the RoboMonkey pipeline.

## Sardana et al.: inference demand belongs in the allocation boundary

Citation: *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model
Scaling Laws*, ICML 2024, PMLR 235:43445–43460.
The [publisher record](https://proceedings.mlr.press/v235/sardana24a.html) supplies
the four authors and final metadata. The analysis incorporates inference demand
when choosing training/model allocations for a quality target, considering
compute and monetary expenditure.

Our application: cite this precedent next to adaptation-plus-repeated-use
accounting. The identity is not claimed as novel. Robot labor, training GPU
time, and deployment elapsed time remain separate unless a compatible boundary
and explicit valuation are declared. The original synthetic table supplies no
cost crossover.

## Jiang et al.: deployment performance with accuracy assumed

Added 2026-09-12: *How Fast Can I Run My VLA? Demystifying VLA Inference Performance
with VLA-Perf*, [arXiv:2602.18397v1](https://arxiv.org/abs/2602.18397v1).
The [manuscript](https://arxiv.org/html/2602.18397v1), Section 3, assumes deployment
accuracy requirements are met. Sections 4.5, 4.7, and 4.9 analyze denoising,
predicted chunk size, hardware/network deployment, and asynchrony. These are direct
systems precedents, not omissions we claim to discover. Our proposed study measures
cost and closed-loop outcomes together; its executed prefix differs from predicted
chunk size. The existing [SmolVLA citation](https://arxiv.org/abs/2506.01844v1)
also explicitly introduces asynchronous inference and simulated/real-world tests.

## Decision target and evidence limits

The manuscript defines the best contract-specified scalar quality among a finite
declared set of configurations satisfying separate resource caps. This defines
the question; it does not fit a frontier or scaling exponent. Unknown applicable
resource values cannot certify feasibility. An empty feasible set supplies no
feasible choice. Measured quality needs its own uncertainty and threshold.

The archive illustrates why a generation-count and elapsed-time resource axis
can differ. The constructed-defect suite checks specified record mechanisms.
Neither resolves a quality-preserving allocation. The prospective study retains
the four h/d conditions and permits agreeing, disagreeing, or inconclusive
comparisons; it does not assume equal compute or require a rank reversal.

Publisher records establish bibliographic identity; the linked versioned author
manuscripts provide the detailed methodological locators above. Source reading
is not replication of any of the four studies.
