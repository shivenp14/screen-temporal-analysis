# Midterm handoff

Due **November 13, 2026, 11:59 p.m. Eastern**. This is a proposed handoff checklist, not a claim that teammates have agreed to new deadlines or completed work.

## What is ready

- Annotation guidelines: `annotations/guidelines.md`.
- Reproducible 36-item manifest: `annotations/review_manifest.csv`.
- Aggregate audit with source hashes: `results/metrics/release_audit.json`.
- Local blind packets and blank sheets: regenerate using `annotations/README.md`.
- LLM prompt export and evaluator: `src/llm/classify.py`.
- Report content organized to match the supplied guideline: `reports/midterm/report.md`.

## Decisions already grounded in the proposal

- Main sentiment task is three classes; map Mixed to Neutral. Keep `stance_4class` and derived `sentiment_3class` separately.
- Keep the original temporal/platform research questions.
- LLM and transformer comparisons are planned until actual runs exist.
- Topic examples in the proposal are not proof that those labels are released. Start from the documented source vocabulary; version any extensions.

## Handoffs before submission

| Owner | Needed artifact | Acceptance check |
| --- | --- | --- |
| Tia | Reconciled data and split manifest | Stable IDs; source stage; text; platform; original date and parsed date; duplicate group; train/validation/test assignment; counts of excluded rows. Split duplicate groups, not rows. |
| Pranav | Sentiment run and confusion matrix | Same split manifest; explicit Mixed-to-Neutral mapping; vectorizer fit only on training; seed/settings; accuracy, macro precision/recall/F1 and class support. |
| Shiven | Topic-label inventory and baseline or exploration | Confirm where labels came from. Relevance is not a topic. Distinguish unreviewed from reviewed empty sets. Report micro/macro/per-topic F1 only with valid held-out labels. |
| Collin + available reviewer | Two independent annotation sheets | No source labels or predictions shown; evidence/confidence recorded; adjudicator records decisions; reserve matching texts from training. |
| Collin | Final report assembly | Include only verified team results. Keep unresolved experiments in future work. Apply course template and verify at least four rendered pages. |

## Suggested shared record contract

Use `doc_id`, `platform`, `title`, `text`, `publication_date_raw`, `publication_date`, `source_file`, `source_commit`, `text_sha256`, `duplicate_group`, `split`, `relevance`, `stance_4class`, `sentiment_3class`, `topics`, and `label_provenance`. Date failures should remain explicit, not replaced by the filename's year. Topics should be a JSON array or missing, not a comma-split free-text string. Models must not use label/provenance columns as input features.

Publish a split-manifest hash with every result, along with the code revision, configuration, model/version, run date, sample counts, and exclusions. For temporal figures, include platform/bin denominators and missing-date counts. Keep the diagnostic packet separate from tuning data. Freeze the label schema before comparing models.

## Findings to act on

1. The final 1,492-row table is not among the released CSVs. Do not claim a reconstructed final corpus without explaining reconciliation.
2. The 220 recovered rows overlap the relevance-review file completely. They carry relevance decisions only.
3. The accepted table has 932 rows but 614 distinct whitespace-normalized texts. Check conflicting labels within duplicate groups before deciding which records to keep.
4. Accepted Snapchat/Positive has zero rows; do not silently discard this class from aggregate metrics or infer a platform-wide absence.
5. The source assigns IDs after concatenating Snapchat then YouTube. YouTube's index is not its local JSONL line number. Packet generation reproduces this rule and checks text before attaching dates.
6. Topic predictions are described in the source notebook, but the released CSV schemas have no topic column. The notebook unions model topic lists; that can exceed its prompt's three-topic cap.

## Final assembly checks

- [ ] Add Tia's actual data/split/temporal findings, or state exactly what remains pending.
- [ ] Add Pranav's actual baseline results, with run evidence.
- [ ] Add Shiven's actual topic findings, with label provenance.
- [ ] Complete human review if feasible; otherwise retain the pending status.
- [ ] Run an LLM pilot if credentials and reviewed references become available; otherwise retain the preparation-only status.
- [ ] Obtain the course template, apply its format, and verify at least four pages.
- [ ] Check references, captions, denominators, and consistent label names.
- [ ] Review the PR and merge when the team is ready. This contribution does not merge itself.
