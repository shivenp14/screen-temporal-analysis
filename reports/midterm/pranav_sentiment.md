# Sentiment baseline and initial results

The baseline is TF-IDF plus class-balanced logistic regression, using Tia's shared grouped split and the proposal's Mixed-to-Neutral mapping. Results measure agreement with the **accepted intermediate source labels**, not independent human-gold performance or the original paper's final benchmark.

## Reproduce

After the data PR has been applied and its preprocessing command has run:

```sh
python -m src.sentiment.baseline
python -m unittest discover -s tests -v
```

The command verifies the processed dataset and split-manifest hashes, checks duplicate-group separation, trains on the 339 training records only, and evaluates the fixed model on 113 validation and 113 test records. The 193 review-reserved records are excluded. It writes metrics, per-record predictions, and confusion matrices to `results/sentiment/`; the serialized pipeline stays in ignored `results/models/`. Use new `--output` and `--model` paths for another run.

## Configuration

The TF-IDF vocabulary uses unigrams/bigrams, a 20,000-feature cap, minimum document frequency 2, and sublinear term frequency. No stop-word list is used, preserving negation. Vocabulary and IDF are learned from training texts only. Logistic regression uses C=1, balanced class weights, the lbfgs solver, a 2,000-iteration limit, and seed 584. The executed fit converged in 16 iterations. These are fixed baseline settings; no validation or test search was performed, and the model was not refit on validation records.

Training labels are 257 Negative, 48 Neutral, and 34 Positive. Balanced class weights reduce the dominance of Negative during fitting. A training-majority classifier is included as the reference baseline, rather than comparing only against a raw corpus percentage. Probability outputs are uncalibrated and should not be described as reliable confidence estimates.

## Results

| Split / model | n | Accuracy | Macro precision | Macro recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation / logistic regression | 113 | 0.9204 | 0.8702 | 0.8369 | 0.8480 |
| Validation / majority | 113 | 0.7611 | 0.2537 | 0.3333 | 0.2881 |
| Test / logistic regression | 113 | 0.8142 | 0.7131 | 0.7001 | 0.7060 |
| Test / majority | 113 | 0.7611 | 0.2537 | 0.3333 | 0.2881 |

| Test class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Positive | 12 | 0.8182 | 0.7500 | 0.7826 |
| Negative | 86 | 0.8837 | 0.8837 | 0.8837 |
| Neutral, including Mixed | 15 | 0.4375 | 0.4667 | 0.4516 |

The test accuracy improves by 5.31 percentage points over majority prediction, while macro-F1 improves by 0.4179. The gap between validation and test scores shows why one favorable split should not be treated as a stable performance estimate. Test support is small, especially for minority classes; these results do not establish statistical significance.

![Test confusion matrix](../../results/sentiment/test_confusion.svg)

The largest error directions are Negative -> Neutral (9) and Neutral -> Negative (7), accounting for 16 of 21 test errors. Of the 15 three-class Neutral test references, 14 were originally Mixed and only one was originally Neutral. This is a concrete reason to retain the original four-class labels: the hardest category here mostly represents mixed framing, not simply descriptive reporting.

As an illustrative case, the article identified by `snapchat|104|Meta, Snap, TikTok join forces to combat suicide and self-harm content` has a Mixed source label and is predicted Negative. It describes both harmful content and a coordinated safety response. A plausible explanation is that harm-related terms outweigh benefit/mitigation framing in this lexical baseline; this is an interpretation, not a verified attribution of the model's decision. The example was inspected after scoring and was not used to tune the baseline.

YouTube test accuracy is 0.7879 (n=66); Snapchat is 0.8511 (n=47). Per-platform macro scores in the JSON include all three classes with zero for undefined scores. Snapchat has no positive reference items, so its macro-F1 cannot be read as a comparison on equal class coverage. Always report support with these scores.

## Limits and next work

The source labels are intermediate consensus outputs, not new human annotations. Grouping blocks exact duplicates and high-similarity shingle variants, but remaining related stories could still make this easier than generalizing to a new event or future period. The benchmark is not a chronological holdout. The diagnostic human-review subset and a future time-based split are useful next checks. Keep this test set fixed and avoid repeatedly adjusting settings after seeing its errors.

Pranav should review the label mapping, class weighting, confusion matrix, and error examples. Transformer comparisons remain outside this baseline contribution. Exact input hashes, package versions, run time, settings, and full class metrics are stored in `results/sentiment/metrics.json`.

Implementation references: scikit-learn [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
