# Independent second annotation: coordinator protocol

Status: deferred and **outside the current submission**, because no second human
annotator is available. Prepared, not conducted. No reliability result exists.
The old CSV is a preliminary single-annotator coding, not ground truth.

## Freeze the comparison before handing off

Use `blind_annotation.json`: 16 systems × 8 coordinates = 128 blank labels.
It contains identifiers only, never the existing labels or evidence judgments.
Paper identifiers are inherited from the pilot, not newly source-verified here.
Before annotation, a coordinator must verify each source and fill the exact
`source_version`, `comparison_id`, `claim_locator`, and `boundary` for every paper.
Boundary means the named method/configuration and comparator, evaluation setting,
and included acquisition, adaptation and deployment stages. Freeze identical
copies for both readers. A whole paper can contain several incompatible budgets;
do not let annotators silently select different experiments.

Distribute ONLY the frozen JSON, source documents, and `../ANNOTATOR_GUIDE.md`.
Do not give the second reader this repository, its manuscript, the CSV, existing
summaries, or adjudication notes. Readers must declare prior exposure. Someone
who already saw the labels cannot retrospectively become a blind annotator.
Independent model runs are not independent human annotation.

## Collect independent evidence

Both readers complete the same scope using the same guide. The original reader
must also supply cell-level evidence under this guide; do not invent source
locators to upgrade the historical CSV. Record annotator ID, human/model-assisted
status, exposure declaration and completion time. Unavailable sources remain
unresolved (`label: null`), never automatically U or NA. No kappa is produced
until every cell is resolved and supported.

Preserve signed-off originals. Run:

```bash
python agreement.py annotator_a.json annotator_b.json
python -m unittest -v test_agreement
```

The report gives unweighted nominal Cohen's kappa, raw agreement, four-class
confusion matrices, marginal counts, and disagreements overall and per dimension.
NA is a fourth class, not missing data. Also report the both-applicable subset
with its changed denominator; never silently drop NA disagreements. Undefined
kappa (zero denominator) is null, not 1. Per-dimension n=16 is small; pooled
128 cells are clustered within papers and heterogeneous across dimensions.
No independent-cell significance test or reliability threshold is asserted.
Formula reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

## Adjudicate only after measuring agreement

For each disagreement, preserve A/B labels and evidence, then separately record
the adjudicated label, source locator, rationale, adjudicator and date. Unresolved
disagreements stay unresolved. Never calculate reliability on the adjudicated
table, or overwrite independent annotations with consensus. A high kappa does
not establish factual accuracy, source completeness or field-wide generality.

Do not update the abstract with agreement numbers until genuine independent
annotations are returned. The present draft remains preliminary coding.
