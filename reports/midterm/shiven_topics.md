# Topic inventory, exploration, and annotation plan

The released CSVs have **no topic columns**. Their labels are accepted/provisional stance votes or relevance decisions. The source Stage 2 notebook defines a 12-topic vocabulary and describes generating topic outputs, but those outputs are not in the pinned release. A supervised multi-label baseline and topic F1 scores would therefore require labels we do not currently have.

This contribution takes the fallback specified in the task: preliminary topic exploration and a concrete annotation plan. The `src/topics/baseline.py` entry point runs that fallback and explicitly records unavailable supervised metrics as `null`.

## Reproduce

Install the requirements in Python 3.12 and obtain the pinned source clone at `data/raw/screen-source` as described in `annotations/README.md`. Run from the project root with fresh output directories:

```sh
python -m src.data.preprocess --source data/raw/screen-source --output data/processed/screen-reproduction --public results/data-reproduction
python -m src.topics.baseline --source data/raw/screen-source --dataset data/processed/screen-reproduction/articles.jsonl --preprocessing results/data-reproduction/preprocessing.json --output results/topics-reproduction
python -m unittest discover -s tests -v
```

These commands preserve the committed results. For another run, choose new empty directories for all three output paths and update `--dataset` and `--preprocessing` to match. The new topic outputs are written under `results/topics-reproduction/`.

Source CSV inventory hashes normalize CRLF line endings to LF before SHA-256, matching the release audit convention. Generated dataset and split-manifest hashes remain hashes of their exact artifact bytes.

The topic command verifies the pinned source and dataset hash, audits all four released CSV schemas, and uses only training records. It collapses cross-platform exact-text copies for exploration, leaving **282 distinct training texts** from 339 article/platform training records. Validation, test, and review-reserved texts are not used. The committed reference outputs are the inventory/configuration/components in `results/topics/exploration.json`, keyword candidates in `results/topics/keyword_candidates.jsonl`, and a component-count plot. Reproduction writes those same filenames into the fresh topic output directory.

## Vocabulary and label status

Use the exact 12 topic names in `src/llm/schema.py` and the definitions in `annotations/guidelines.md`: Anxiety; Depression; Suicide/Self-Harm; Bullying/Cyberbullying; Body Image/Eating Disorders; Addiction/Overuse; Sleep Disruption; Policy/Lawsuits/Bans; School/Education; Therapy/Access to Care; Stigma/Awareness; Parental Controls/Safety Features.

Each reviewed article may receive zero to three primary topics. A reviewed empty array differs from missing review. The proposal mentions substance use as an example, but it is not a released topic label or part of the source notebook's vocabulary; add it only through a documented schema revision and consistent annotation. The source notebook unions topic outputs from several models, which can exceed its individual three-topic cap. Imported outputs would need provenance and schema checks before comparison with our reviewed labels.

## Initial exploration

TF-IDF uses English stop words, unigrams/bigrams, minimum document frequency 2, maximum document frequency 0.95, and a 10,000-feature cap. Nonnegative matrix factorization (NMF) uses eight components, nndsvda initialization, seed 584, a 1,000-iteration limit, and tolerance 0.0001. The fit completed in 37 iterations. Eight components is a fixed exploratory setting, not a tuned number of mental-health topics. Reconstruction error is retained as a fitting diagnostic, not a classification score.

| Component | Selected terms | Texts with this strongest component |
| --- | --- | ---: |
| 0 | social, media, children, platforms | 104 |
| 1 | boards, school, district, Toronto | 25 |
| 2 | city, New York, lawsuit, companies | 50 |
| 3 | constantly, teenagers, Pew, report | 6 |
| 4 | Annie, high school, advice | 5 |
| 5 | EU, commission, DSA, recommender | 8 |
| 6 | Arkansas, lawsuit, claims, YouTube | 23 |
| 7 | children, quality, Nandy, content | 61 |

These are interpretations of component terms, not topic annotations. The components mix broad language, named events, and recurring syndicated stories. For example, school-board and city-lawsuit components suggest useful policy/education review candidates, while the Pew and advice-column components show that NMF can isolate a particular story family instead of a mental-health concept. The generic component and imperfectly coherent component 7 are further reasons not to map components automatically into the 12 labels.

![Exploratory component distribution](../../results/topics/nmf_components.svg)

A separate regex pass finds texts containing search cues for each proposed topic. The largest candidate counts are Policy/Lawsuits/Bans (169/282), School/Education (152/282), and Addiction/Overuse (139/282). The smallest include Therapy/Access to Care (24/282), Sleep Disruption (25/282), and Parental Controls/Safety Features (29/282). Counts overlap because a text can match multiple cues. They are **not topic prevalence**: a school mention can be incidental, a risk may be negated, and a relevant concept may use different wording. Candidate files must not be used as reference labels or shown to blinded diagnostic reviewers.

## Annotation and supervised-baseline plan

1. Obtain the missing final topic-bearing release if possible. Check how labels were generated, whether final topics exceed three, and whether the exact article IDs/texts match. Do not treat a `MH_Relevant` value as a topic label.
2. Complete Collin's independent 36-item diagnostic review and adjudication. Keep it reserved from training. Use the agreement tool to report topic agreement and set Jaccard before resolving differences; preserve both reviewers' evidence.
3. Select a separate annotation batch from the training partition. Use the exploratory candidates to improve coverage of less frequent topics, while retaining some randomly selected training texts. Two reviewers should label the full text under the same frozen guidelines. Keep sampling provenance so enriched topic counts are not described as corpus prevalence.
4. Encode reviewed topics as a binary matrix using the fixed schema order. Drop unreviewed rows from supervised training/evaluation; keep reviewed empty arrays as all-zero targets. Check positive and negative support for every label in each partition. Labels with insufficient support need explicit treatment rather than a misleading F1.
5. Once labels exist, fit TF-IDF on reviewed training texts and a one-vs-rest logistic regression classifier. Select any thresholds on validation labels only. Evaluate micro-F1, macro-F1, per-topic precision/recall/F1 and support, and document zero-division conventions. Do not tune thresholds on the diagnostic subset or test labels.
6. If the deadline arrives before usable labels, submit this inventory and exploration with the limitation stated plainly. Do not manufacture a supervised result from keywords or NMF components.

Shiven should review the vocabulary, the representative training IDs in the JSON, and whether component themes justify any schema additions. The current outputs satisfy the no-label exploration path; they do not complete a supervised topic experiment.

References: the pinned [SCREEN release and notebooks](https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset/tree/246d3d365d5898f6fdf5240264215e49b082d743), and scikit-learn [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html). NMF is used as a lightweight exploratory method here; the proposal's BERTopic and SetFit comparisons remain future work.
