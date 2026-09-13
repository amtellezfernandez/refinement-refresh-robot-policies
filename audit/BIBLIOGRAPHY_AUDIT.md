# Proposal 2 bibliography audit

## Addition checked 2026-09-12: VLA inference systems

Added `vlaperf`, bringing the bibliography to 37 entries. The
[versioned primary record](https://arxiv.org/abs/2602.18397v1) verifies the title,
2026 preprint status, and four authors: Wenqi Jiang, Jason Clemons, Karu
Sankaralingam, and Christos Kozyrakis. See [source notes](ALLOCATION_LITERATURE.md)
for the scope of the manuscript's use. The existing SmolVLA citation is reused.

## Addition checked 2026-09-12: allocation studies

Four references were added to the existing 32-entry bibliography (36 entries
at that revision). [Detailed source and claim notes](ALLOCATION_LITERATURE.md) separate
published metadata from methodological reading of versioned author manuscripts.

| Key | Primary metadata source | Role in the manuscript |
|---|---|---|
| `pearcescaling` | [ICML 2025 / PMLR 267:48542–48562](https://proceedings.mlr.press/v267/pearce25a.html) | Representation-dependent data and parameter boundaries |
| `motionscaling` | [arXiv:2506.08228v2](https://arxiv.org/abs/2506.08228v2) | Direct embodied allocation and closed-loop planning precedent |
| `robomonkey` | [CoRL 2025 / PMLR 305:3200–3217](https://proceedings.mlr.press/v305/kwok25a.html) | Policy sampling, Gaussian proposals, action verification, and deployment cost |
| `beyondchinchilla` | [ICML 2024 / PMLR 235:43445–43460](https://proceedings.mlr.press/v235/sardana24a.html) | Prior inference-demand-aware allocation, cited beside the accounting identity |

No publication status is inferred from the preprint title: Baniodeh et al. is
cited as a versioned technical report. The other three use publisher metadata.
The methods inform our proposed reporting requirements; none is presented as
neglecting resource measurement, and no new scaling relationship is claimed.

## Original 2026-09-10 audit (32 entries at that revision)

Audit date: **2026-09-10**. Scope: the 28 entries and their citation uses in the incoming `main.tex`, plus four missing references added for the revised manuscript. This is a bibliographic and citation-scope audit, **not** an independent recoding of the 16-paper disclosure table. Metadata verification does not establish whether a paper reports a particular resource.

## Corrections made

- Corrected five erroneous given names across four entries: **Tian-Yu Xiang**, **Fardin Afdideh**, **Rongxiang Zeng**, **Yongqi Dong**, and **Harsh Raj**. The draft instead used Jiannan, Fazel, Y., Z., and S., respectively.
- Replaced the outdated RoboFAC author list and order with the seven authors on arXiv v4: Zewei Ye, Weifeng Lu, Minghao Ye, Tao Lin, Shuo Yang, Junchi Yan, and Bo Zhao. This is a version-dependent correction; old extraction notes must retain the source version they actually used.
- Expanded abbreviated `and others` author lists using the primary records. Conference entries use their conference author metadata where this differs from an earlier preprint.
- Corrected WoVR's title to the current v2 title ending in **“with RL.”**
- Repaired the Chinchilla entry's mixture of preprint title and published venue. The NeurIPS 2022 title is **“An Empirical Analysis of Compute-Optimal Large Language Model Training.”**
- Updated Snell et al. to the ICLR 2025 publication and its changed title ending in **“Scaling Parameters for Reasoning.”** Updated Tulu 3 to COLM 2025, SAM2Act to ICML 2025, and Q-chunking to NeurIPS 2025. Conference works use `@inproceedings` and `booktitle`.
- Pinned arXiv URLs and identifiers to the inspected version. Added a rendered `howpublished` field, because the manuscript's traditional BibTeX style need not display `eprint`/`archivePrefix`. A preprint is labeled as such; an author-reported acceptance is identified separately from a proceedings record.
- Retained the verified E2HiL RA-L volume, issue, pages, year, DOI, and journal author list; added its author-maintained project URL. The older arXiv author list is different and is not substituted for the journal list.
- Added **`smolvla`**, **`libero`**, **`modelcards`**, and **`energyreporting`**. These cover the experimental backbone/benchmark and direct precedents for standardized cards and resource accounting.

The original identifiers resolve to the intended works. No nonexistent reference was found among the 28 original entries. The original bibliography nevertheless contained substantive attribution and version errors.

## Primary-source register

“Preprint” below means the consulted primary record does not establish a peer-reviewed publication. It does not assert that publication elsewhere is impossible. Years of arXiv entries follow initial release; later revision dates are listed separately. Source metadata was checked on the audit date, not continuously monitored.

| Key | Record inspected and version/status | Disposition |
| --- | --- | --- |
| `robotpadcfp` | [RoboPAD call for papers](https://robotpad2026.github.io/call-for-papers.html), 2026 organizer webpage | Retained with access date. Non-archival workshop; page is a policy/scope source, not scientific evidence. |
| `persistentwm` | [arXiv 2603.25685v2](https://arxiv.org/abs/2603.25685v2), revised 2026-09-04 | Authors/title verified. Record reports ECCV 2026 acceptance; cite the versioned author manuscript without invented proceedings pages. |
| `hiwm` | [arXiv 2604.21741v2](https://arxiv.org/abs/2604.21741v2), revised 2026-05-05; preprint | Authors/title verified; version pinned. |
| `tauzero` | [arXiv 2608.16885v1](https://arxiv.org/abs/2608.16885v1), 2026-08-17; preprint | Expanded all 39 authors; version pinned. |
| `wovr` | [arXiv 2602.13977v2](https://arxiv.org/abs/2602.13977v2), revised 2026-06-27; preprint | Expanded 14 authors; corrected v2 title and pinned version. |
| `vlarl` | [arXiv 2505.18719v1](https://arxiv.org/abs/2505.18719v1), 2025-05-24; [author publication list](https://ziweiwangthu.github.io/) identifies IROS 2026 | Retained cited 2025 preprint; separately notes author-listed venue. No invented IROS DOI or proceedings pagination. |
| `e2hil` | [Author project and journal BibTeX](https://e2hil.github.io/), RA-L 11(7):8084–8091, 2026, DOI [10.1109/LRA.2026.3693577](https://doi.org/10.1109/LRA.2026.3693577) | Journal metadata verified from author primary source. Direct DOI page could not be read. The [arXiv record](https://arxiv.org/abs/2601.19969) has a different author list; distinguish it in extraction notes. |
| `flash` | [arXiv 2605.13778v1](https://arxiv.org/abs/2605.13778v1), 2026-05-13; preprint | Authors/title verified; version pinned. |
| `svvla` | [arXiv 2604.02965v1](https://arxiv.org/abs/2604.02965v1), 2026-04-03; explicitly under review in record | Authors/title verified; retained as preprint. |
| `step` | [arXiv 2602.08245v2](https://arxiv.org/abs/2602.08245v2), revised 2026-05-02; [ICML 2026 listing](https://icml.cc/Downloads/2026) | Acceptance supported by arXiv comment and official conference listing. Retained versioned author manuscript; no unverified PMLR volume/pages added. |
| `a2c2` | [arXiv 2509.23224v1](https://arxiv.org/abs/2509.23224v1), 2025-09-27; preprint | Authors/title verified; version pinned. |
| `wherecompute` | [arXiv 2607.13389v1](https://arxiv.org/abs/2607.13389v1), 2026-07-15; preprint | Authors/title verified; version pinned. |
| `embodimentgap` | [arXiv 2608.18433v1](https://arxiv.org/abs/2608.18433v1), 2026-08-19; journal reference specifies TMLR, August 2026 | Retained journal attribution; made author-manuscript version explicit. OpenReview final record was not independently inspected. |
| `instructgpt` | [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html), volume 35, DOI 10.52202/068431-2011 | Expanded 20-author list from proceedings, including Jeffrey Wu; converted to conference entry; added official URL/DOI. |
| `chinchilla` | [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html), volume 35, DOI 10.52202/068431-2176 | Corrected published title, expanded 22 authors, and used published entry. |
| `snell` | [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html) | Corrected title/year/venue/URL to published work. Earlier [arXiv v1](https://arxiv.org/abs/2408.03314v1) remains a distinct 2024 version. |
| `tulu3` | [COLM 2025 accepted-paper list](https://colmweb.org/2025/AcceptedPapers.html), linking [OpenReview](https://openreview.net/forum?id=i1uGbfHHpH) | Updated venue/year and all 23 conference-listed author names, including Xinxi Lyu and Christopher Wilhelm. [arXiv v5](https://arxiv.org/abs/2411.15124v5) uses some different name forms. OpenReview itself presented browser verification. |
| `xiangtaxonomy` | [arXiv 2506.20966v2](https://arxiv.org/abs/2506.20966v2), revised 2026-01-29; preprint | Corrected Tian-Yu Xiang; expanded 14 authors; pinned version. |
| `sixdim` | [arXiv 2608.06246v1](https://arxiv.org/abs/2608.06246v1), 2026-08-06; preprint | Corrected Fardin Afdideh; added Fernando Seoane and Farhad Abtahi. |
| `latentwm` | [arXiv 2603.09086v1](https://arxiv.org/abs/2603.09086v1), 2026-03-10; record says under review | Corrected both given names; retained as preprint, without claiming an IEEE journal publication. |
| `modelharness` | [arXiv 2607.28802v1](https://arxiv.org/abs/2607.28802v1), 2026-07-30; preprint | Corrected Harsh Raj and expanded seven authors. |
| `rssposttraining` | [RSS 2026 workshop/challenge website](https://posttraining-for-robotics.github.io/), July 13, 2026 | Title/year/event verified; retained organizer webpage with access date. |
| `sam2act` | [ICML 2025 / PMLR 267:15925–15942](https://proceedings.mlr.press/v267/fang25c.html); [arXiv v4](https://arxiv.org/abs/2501.18564v4), revised 2025-07-13 | Published citation replaces preprint-only entry. Distinguish SAM2Act from the SAM2Act+ memory variant when extracting evidence. |
| `sear` | [arXiv 2603.01891v2](https://arxiv.org/abs/2603.01891v2), revised 2026-08-13; preprint | Authors/title verified; version pinned. |
| `qchunking` | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html), volume 38, DOI 10.52202/085713-1856 | Published citation replaces preprint-only entry; uses conference-listed Zhiyuan (Paul) Zhou. Later [arXiv v4](https://arxiv.org/abs/2507.07969v4) is dated 2026-05-11 and should be named if used for extraction. |
| `corft` | [arXiv 2508.02219v1](https://arxiv.org/abs/2508.02219v1), 2025-08-04; preprint | Authors/title verified; version pinned. |
| `fpcvla` | [arXiv 2509.04018v2](https://arxiv.org/abs/2509.04018v2), revised 2025-12-03; preprint | Verified the displayed 11-author list (not the inconsistent prose count in arXiv's generated description); version pinned. |
| `robofac` | [arXiv 2505.12224v4](https://arxiv.org/abs/2505.12224v4), revised 2026-03-22; preprint | Corrected author list/order for v4; retained 2025 initial year with revision note. |
| `smolvla` (new) | [arXiv 2506.01844v1](https://arxiv.org/abs/2506.01844v1), 2025-06-02; preprint | Added full 14-author model reference. A model paper is not a substitute for the exact experimental checkpoint revision. |
| `libero` (new) | [NeurIPS 2023 Datasets and Benchmarks proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html), volume 36, DOI 10.52202/075280-1939 | Added seven-author benchmark citation. Software revision and task/init-state identifiers still belong in the run manifest. |
| `modelcards` (new) | [arXiv record with FAT* 2019 journal reference and ACM DOI](https://arxiv.org/abs/1810.03993), DOI [10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596) | Added published 2019 conference citation and all nine authors. |
| `energyreporting` (new) | [JMLR 21(248):1–43, 2020](https://jmlr.org/papers/v21/20-312.html) | Added six-author journal citation for standardized energy/carbon accounting. |

## Citation-scope findings for the manuscript redesign

1. **Card and accounting novelty needs narrower positioning.** Model Cards already establishes standardized model documentation, and Henderson et al. already offers resource measurement and reporting infrastructure. The useful proposed extension is a comparison-specific ledger for heterogeneous robot adaptation resources. Do not claim that standardized cards, executable accounting, or reporting cost alongside quality are themselves new.

2. **A short selected bibliography cannot prove universal absence.** The draft's claim that prior work does not provide any operational lifecycle ledger is stronger than this audit establishes. Describe the proposed scope and differences from named precedents, rather than asserting no existing system covers the problem.

3. **The 16-paper disclosure labels require source-specific evidence.** A valid bibliographic record supports identity, not a Q/P/U/NA label. Neither the abstracts nor this audit establish that every row was checked, that missing quantities are applicable, or that a comparison cannot be reconstructed. The changed WoVR/RoboFAC versions, E2HiL journal/preprint difference, and SAM2Act/SAM2Act+ distinction are especially important. Freeze extraction versions independently from the bibliography; do not relabel old rows as newly verified.

4. **Do not equate VLA-RL's test-time optimization with deployment deliberation.** The paper describes online RL adaptation, so optimization can entail learning and simulator use. Its abstract's phrase about test-time scaling is not by itself evidence of a per-decision search/inference budget. Any resource card must specify the particular experiment and denominator. This audit does not validate the draft's “48 GPU-hours” as a complete per-recipe cost.

5. **SEAR and Q-chunking are adjacent action-chunking RL precedents.** Their primary abstracts concern online/offline-to-online RL, not necessarily adaptation of a pretrained robot foundation model. A corpus described as 16 robot-foundation-model post-training systems overstates its homogeneity. Explicitly justify inclusion or describe it as a sample of relevant mechanisms.

6. **Model or Harness does not demonstrate independent human inter-annotator agreement.** Its abstract describes independent reasoning-agent judges compared against human category labels. A generic assertion that it tests annotation reproducibility is supportable, but it cannot substitute for independent human coding of this proposal's literature audit.

7. **Workshop websites support scope and community priorities.** The RSS page asks for shared problem definitions, evaluation protocols, reusable reports, and baselines. It does not establish adoption of this resource card or a dedicated resource-disclosure standard. RoboPAD's call explicitly welcomes early results, negative results, positions, and evaluation work; it supports a clearly labeled instrument/position submission.

8. **Efficiency precedents are contextual, not evidence for this pilot.** FLASH, SV-VLA, STEP, and A2C2 motivate the latency/reactivity trade-off; they do not establish that the proposal's specific chunk schedule preserves success or safety. SmolVLA and LIBERO citations establish model/benchmark provenance only. Exact checkpoint, software, timing boundary, seeds, paired initial states, and uncertainty must come from the experiment artifacts.

## Verification limits and remaining gates

- All 32 final entries have a primary identity/metadata source recorded above. This was not an exhaustive venue search for every preprint and did not independently reproduce any cited paper.
- Direct IEEE DOI access failed; E2HiL metadata is supported by the author-maintained journal BibTeX. OpenReview browser verification prevented direct inspection of the Tulu 3 forum and TMLR final record; COLM's official list and the author-deposited TMLR journal reference provide the stated evidence. These access limits must not be described as publisher-level checks.
- The bibliography now makes version choices explicit. Updating citation metadata **does not update or validate historical resource extractions**. A completed comparison-level audit needs an evidence location, source version, unit, phase, denominator, and derivation for every scored item.
- Structural checks passed: 32 unique entries, balanced braces, author/title/year/URL present for every entry, and no unresolved citation key in the manuscript snapshot checked (27 cited keys). `git diff --check` passed. A full LaTeX/BibTeX render remains the integration check; no standalone BibTeX executable was found on PATH during this subtask.

No manuscript or experiment file was edited by this bibliography subtask.
