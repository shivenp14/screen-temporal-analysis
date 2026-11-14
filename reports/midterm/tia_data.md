# Data preparation and temporal analysis

This contribution uses the **accepted intermediate** SCREEN table. It does not reconstruct or claim to reproduce the final 1,492-article corpus described by the dataset authors.

## Reproduce

Install the repository requirements in Python 3.12 and fetch the pinned source as described in the README. Run from the project root:

```sh
python -m src.data.preprocess
python -m src.analysis.temporal
python -m unittest discover -s tests -v
```

Preprocessing writes full article records under ignored `data/processed/screen/`. Its public outputs are `results/data/preprocessing.json`, `split_manifest.csv`, and `excluded_records.csv`. The command refreshes those generated public summaries, allowing reproduction in a fresh clone that already contains the committed summaries. It refuses to overwrite a nonempty processed-data folder; for a second run use a new `--output` path and optionally a new `--public` path. Temporal summaries and SVG figures are written to `results/temporal/`.

## Data decisions

- Match accepted CSV records to raw JSONL using the source's global Snapchat-then-YouTube index, full identifier, and normalized full text. A mismatch stops the run.
- Normalize whitespace, preserving punctuation, capitalization, and negation. Do not remove stop words or learn a vocabulary during preprocessing.
- Keep the four-class source stance and separately derive the proposal's three-class sentiment by mapping Mixed to Neutral.
- Exclude missing text or invalid labels. Mark unparseable dates as missing rather than guessing them from filenames. Date failures remain eligible for classification but are excluded from temporal summaries.
- Within each platform, collapse exact normalized text duplicates. If copies disagree on the original stance, exclude that entire exact-text/platform group for later adjudication. Choose the earliest valid date for retained duplicates and preserve all original IDs locally.
- Retain the same text across platforms because the unit of analysis is article/platform framing. Keep those copies in one split. Counts therefore describe article/platform records, not globally unique articles.
- Group near duplicates using word-five-shingle Jaccard similarity >= 0.85 and connected components. The lower cutoff captures lightly edited syndicated copies such as the Arkansas reports while still requiring strong overlap. This heuristic reduces overlap but cannot guarantee that every paraphrase or related story is detected. Grouping may join different platform labels; it does not overwrite their targets.
- Build the duplicate graph jointly with Collin's 36 review texts. Reserve the entire component of every review item, including related accepted articles. Review records are never used in the training/validation/test benchmark.
- Use seeded, shuffled five-fold stratified group assignment: one fold test, one validation, and three training. The intended proportions are approximately 20/20/60; group constraints can make actual sizes differ. Require all three sentiment classes in each benchmark partition.
- Leave topics missing (`null`). Missing topic annotations are not negative labels.

## Reading the results

The executed run retains **758 article/platform records** after removing 174 within-platform copies. Those records contain 614 distinct normalized texts in 437 duplicate components. All retained dates parse and fall within 2024. There were no conflicting-label exact duplicates within a platform in this run.

The benchmark uses 329 training, 110 validation, and 110 test records. A further 209 records are reserved because their duplicate component intersects the review packet. This reservation is larger than the 36 selected review items because it includes related articles, not just the selected identifiers. After the duplicate cutoff was tightened, the retained set has 405 duplicate components and 1,422 near-duplicate pairs when the review texts are included. The shared test split was regenerated from these groups.

YouTube has 417 retained records: 285 Negative, 62 Neutral (including Mixed), and 70 Positive. Snapchat has 341: 257 Negative, 84 Neutral, and no Positive. These are intermediate-stage source-label counts, not predictions or the final source-paper corpus.

In this subset, Snapchat's Negative share is 96.7% in Q1 (177/183) and 41.0% in Q4 (25/61). YouTube's corresponding shares are 82.9% (150/181) and 55.0% (66/120), with a non-monotonic pattern between them. These descriptive changes need sensitivity checks: the source selection and earliest-date deduplication policy can affect the shape, and a quarterly plot hides monthly variation.

`preprocessing.json` records counts, exclusions, date range, platform/label distributions, split support, and dataset/manifest hashes. The monthly and quarterly CSVs include both numerator and denominator for every class, including zero counts. Bins with fewer than ten records are marked `small_bin`; no significance tests or causal claims are made.

Figures describe the retained source labels across all partitions, including review-reserved records. They do not describe model predictions or held-out performance. The Neutral line includes Mixed under the proposal's mapping. Missing platform/time bins appear as gaps rather than zeros. Sparse bins and intermediate-stage selection mean these plots should be treated as initial exploration, not population trends.

The source's four annotation CSVs remain separate: recovered relevance rows are not treated as stance labels, and unresolved stance votes are not promoted to gold. Excluded records remain identifiable for later reconciliation.

## Review handoff

Tia should review the date policy, duplicate threshold, exclusions, and plotted denominators. Any change to those decisions requires regenerating the split manifest and rerunning downstream models. The source and split hashes should accompany all model results. Near-duplicate grouping and exact-label conflicts are the main decisions to inspect before treating this as the shared final dataset.

Implementation reference: scikit-learn's [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). The dataset provenance is the pinned [SCREEN release](https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset/tree/246d3d365d5898f6fdf5240264215e49b082d743).
