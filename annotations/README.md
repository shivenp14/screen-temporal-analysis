# Review packet

Run from the project root with Python 3.12. These commands use only the standard library, so no model or data-science dependencies are needed.

```sh
git clone https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset.git data/raw/screen-source
git -C data/raw/screen-source checkout 246d3d365d5898f6fdf5240264215e49b082d743
python -m src.llm.prepare_review --source data/raw/screen-source --output data/processed/review
python -m src.llm.classify prepare --articles data/processed/review/articles.jsonl --output results/llm_requests/zero-shot.jsonl
python -m unittest discover -s tests -v
```

If the source folder already exists, use it rather than cloning again. Packet generation refuses to overwrite a nonempty output folder. Keep completed reviews separate and use a fresh `review_v2` folder when regenerating. The source commit and Dataset working tree are checked before sampling.

The committed manifest contains identifiers and text hashes, not full article text. The local packet contains 36 articles, two blank reviewer sheets, a private source-label lookup, and the audit. Sampling uses seed 584 and a stable hash ordering, four items per available platform/stance group plus four unresolved stance items per platform. Exact whitespace-normalized duplicate texts are excluded from the packet. It does not detect near duplicates. The missing Snapchat/Positive group is left missing rather than filled with guessed labels.

See [guidelines.md](guidelines.md) before reviewing. Full text and working sheets stay in ignored local folders. Copy just the public manifest and aggregate audit into the repository when updating them, after checking their contents. Dates are recovered using Stage 1's identifier construction: filter texts shorter than 20 characters, concatenate Snapchat then YouTube, and use the global row index. The complete identifier and normalized text must both match before a date is attached. An unmatched selected item stops generation.

## Agreement evaluation

After both independent sheets are complete, calculate agreement before adjudication:

```sh
python -m src.llm.agreement --first data/processed/review/reviewer_1.csv --second data/processed/review/reviewer_2.csv --output results/metrics/review-agreement.json
```

Blank sheets fail validation rather than producing scores. The script requires two named reviewers, matching identifiers, evidence/confidence for relevant items, and valid labels. It reports relevance agreement, four-class and proposal-mapped stance agreement/kappa, topic agreement/Jaccard, exclusions, and items needing adjudication. Undefined kappa is `null`, not zero. This checks file structure; the team must ensure reviews were actually independent.

## LLM pilot

The `prepare` command writes one request per article. Each request records the complete messages, mode, text hash, prompt hash, truncation flag, character limit, and example identifiers. The default includes the first 12,000 characters, not tokens. Review truncation counts before execution; repeat with a larger limit or a documented full-text strategy if conclusions depend on the ending. The requests include only article text and platform, not source labels or review answers.

There is no provider client or automatic paid call. Run the saved requests with the team's chosen model and record its exact model identifier, run date, generation settings, and raw responses. A live run has **not** been performed. No credentials were configured in the working environment. Request generation and synthetic evaluator tests are preparation, not classification results.

For a few-shot comparison, supply `--examples path/to/examples.jsonl`. Every example must contain `doc_id`, `platform`, `text`, `split: "train"`, and `answer` with `label`, `topics`, and `reason`. Use only independently reviewed training items, ideally covering all four classes. The tool rejects matching evaluation identifiers or normalized text; Tia must also check near-duplicate groups. Do not select examples or tune prompts based on the 36 review outcomes. Compare zero-shot and few-shot on exactly the same held-out items and model settings.

After adjudication, export relevant reviewed records to JSONL:

```json
{"doc_id":"example-id","review_status":"adjudicated","stance":"Negative","topics":["Sleep Disruption"]}
```

This is a schema example, not a completed review. Omit `topics` when topic review is incomplete; use `[]` only for a reviewed absence. Export predictions separately:

```json
{"doc_id":"example-id","answer":{"label":"Negative","topics":["Sleep Disruption"],"reason":"The article emphasizes disrupted sleep."}}
```

For failed calls, use `{"doc_id":"example-id","error":"timeout"}`. Invalid responses must stay invalid; never replace them with Neutral. Keep the original response log with the run. Filter the saved requests to the adjudicated relevant evaluation IDs before scoring and report how many review candidates were excluded and why.

```sh
python -m src.llm.classify score --gold data/processed/review/gold.jsonl --predictions data/processed/review/predictions.jsonl --requests data/processed/review/evaluation-requests.jsonl --model MODEL_ID --run-id RUN_DATE_AND_MODE --output results/metrics/llm-pilot.json
```

Scoring reports coverage, missing/invalid/error counts, accuracy, macro precision/recall/F1, per-class support, and a confusion matrix with an ERROR column. `proposal_three_class` contains the **primary** results after mapping Mixed to Neutral in both reference and predictions, as specified in the proposal. Top-level stance scores retain all four classes as a diagnostic. Failed items remain in both denominators. Macro scores include all classes for the task (three or four), with zero for undefined scores. Topic scores use only records with reviewed topics and report that denominator; failures act as empty predictions, so coverage must always accompany topic F1. Topic macro-F1 includes all 12 topics, including unsupported ones. These conventions should be stated when comparing scores.

No source intermediate labels are treated as human-reviewed gold. The scorer requires an explicit adjudicated status, but it cannot verify who performed a review; preserve the original reviewer sheets as evidence.
