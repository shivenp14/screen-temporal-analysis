# Annotation guidelines, v1.1

## What we are labeling

Read the full article and label its framing of **YouTube or Snapchat's impact on youth mental health**. This is stance toward an impact, not the mood of the article, a diagnosis, or our opinion of the platform. Use the platform named in the packet. If an article discusses both, only assign evidence that applies to the named platform. Do not assume a claim about another platform applies to this one.

SCREEN uses four stance labels. Keep all four in the annotation files. Our project proposal explicitly maps Mixed to Neutral for the main three-class sentiment analysis: Positive -> Positive, Negative -> Negative, Neutral -> Neutral, and Mixed -> Neutral. Apply that mapping as a separate derived field after review, never by overwriting the original label. Report three-class results as the primary comparison and four-class stance results as a diagnostic. This mapping combines balanced/no-leaning coverage with substantive mixed framing, so three-class Neutral must be interpreted accordingly.

## Relevance first

Use `yes`, `no`, or `uncertain`. A relevant article substantively connects the named platform to mental health or well-being in young people. A platform being mentioned as the place a video was posted is not enough. Policy stories qualify when they connect platform use or design to mental-health risks or benefits. General adult mental-health coverage without a youth connection should be flagged for discussion.

For `no` or `uncertain`, leave stance and topics blank and explain why. Do not turn missing context into Neutral. Relevance disagreements go to adjudication before stance evaluation.

## Stance

| Label | Use when | Watch out for |
| --- | --- | --- |
| Positive | Benefits, support, resources, or helpful features dominate the impact discussion. | A company announcing a safety tool does not itself show a benefit. |
| Negative | Harms or risks dominate, including overuse, harassment, or harmful exposure. | Negative words elsewhere in a story are not enough. |
| Neutral | The impact is described without a clear leaning. | Neutral is not a substitute for unreadable text or uncertainty. |
| Mixed | Both benefits and harms receive substantial weight. | One passing counterargument does not make a strongly one-sided story Mixed. |

Use the full article rather than only the headline. Attribute quoted claims to the framing the article presents; do not count quotes as votes. A denial in an otherwise harm-focused article does not automatically make it Mixed. If the evidence remains unclear, use low confidence and flag it for adjudication instead of inventing a rule.

## Topics

Choose **zero to three primary topics**, using the exact names below. A topic needs explicit evidence related to mental health, not just a keyword. If more than three qualify, choose those receiving the most substantive attention and note the others for adjudication. The cap follows the source notebook's per-annotator prompt. Its later union of model topic lists can exceed three, so those outputs would need reconciliation before comparison.

A primary topic must be central to the article's explanation of an impact, response, or intervention. Look for a substantive passage or repeated discussion with supporting detail. A name, risk list, related-story link, or boilerplate mention is insufficient. If more than three topics qualify, prioritize the article's main argument and supporting detail; flag close ties for adjudication.

Apply topics to the named platform. General social-media claims can qualify when the article clearly includes that platform in their scope. Do not transfer claims explicitly limited to another platform. Note ambiguous attribution for adjudication.

Keep the source's 12 topics for this phase. Substance use remains outside this vocabulary; record relevant evidence in notes for a later schema review. A reviewed empty set is valid when no listed primary topic applies.

| Topic | Include | Exclude |
| --- | --- | --- |
| Anxiety | Anxiety or persistent worry tied to platform use. | A generic concern about a product. |
| Depression | Depressive symptoms or depression explicitly discussed. | Sad wording alone. |
| Suicide/Self-Harm | Suicide or self-harm risk, exposure, prevention, or support. | Inferring risk from depression alone. |
| Bullying/Cyberbullying | Harassment or bullying with a well-being connection. | Ordinary disagreement. |
| Body Image/Eating Disorders | Appearance pressure, body dissatisfaction, or eating disorders. | Appearance mentioned without an impact connection. |
| Addiction/Overuse | Compulsive use, dependency, or harmful excessive use. | High usage statistics without harm framing. |
| Sleep Disruption | Platform-related sleep loss or disrupted sleep. | Nighttime use with no sleep claim. |
| Policy/Lawsuits/Bans | Regulation, litigation, or restrictions linked to mental health. | Unrelated business disputes. |
| School/Education | School policies, learning, educational content, or public-health education with an explicit youth well-being connection. | Student status, generic advice, or a passing mention of education. |
| Therapy/Access to Care | Treatment, counseling, or access to mental-health support. | Generic advice without care or treatment. |
| Stigma/Awareness | Mental-health awareness, disclosure, stigma, or normalization. | Any mental-health mention by itself. |
| Parental Controls/Safety Features | Parental controls or platform protections tied to well-being. | A feature announcement with no relevant connection. |

An empty JSON list `[]` means reviewed and no listed topic applies. A blank cell means not reviewed. These must stay different. Use JSON arrays in the CSV topics cell, for example `["Anxiety", "Sleep Disruption"]`.

## Practice examples

These are invented examples for understanding the rules, not dataset observations or experiment results.

* A story says late-night Snapchat use disrupts teenagers' sleep and increases worry, with no substantive benefit discussed: Negative; Sleep Disruption and Anxiety.
* A story describes young people finding counseling resources through YouTube and focuses on improved access: Positive; Therapy/Access to Care.
* A story gives substantial attention to supportive communities and compulsive use: Mixed; Addiction/Overuse, plus another topic only if explicitly supported.
* A descriptive account of a mental-health-related school phone policy without a clear harm or benefit argument: Neutral; School/Education and Policy/Lawsuits/Bans.

## Review process

1. Collin prepares the packet with `python -m src.llm.prepare_review`. Share `articles.jsonl` and one blank reviewer sheet per reviewer. Keep `reference_private.csv`, sampling metadata, and model predictions out of the review packet.
2. Two team members independently label the same 36 articles. Suggested pairing: Collin and Tia, subject to availability. Start with six practice items, discuss the guideline, then independently re-review them under the frozen version before scoring agreement. Do not count practice discussion as independent agreement.
3. Each reviewer fills in their own name, relevance, stance, topics, a short evidence excerpt or paragraph reference, confidence (`high`, `medium`, `low`), and any notes. Read the full article; skip distressing content if needed and note the skip.
4. Compare relevance first. On jointly relevant articles, calculate raw stance agreement and Cohen's kappa before resolving disagreements. Report the denominator, excluded/skipped items, and class counts. For topics, report per-topic binary agreement and set Jaccard; define two empty sets as Jaccard 1. Do not report agreement statistics until two independent reviews exist.
5. A third team member resolves disagreements using both evidence notes and the full article. Record the final decision, adjudicator, date, and guideline version; preserve both original reviews. Mark only resolved items `adjudicated`.
6. Reserve the reviewed articles and matching normalized text hashes from training and prompt examples. Tia should extend this exclusion to near-duplicate articles. The packet is a diagnostic subset, not a representative estimate of platform prevalence or the final benchmark.

No reviews have been completed in this contribution. The subset is organized and ready to distribute.

## Provenance

v1.1 clarifies topic prominence, educational coverage, and platform attribution. The label names and three-topic cap are unchanged.

Vocabulary and four-class stance definition: [SCREEN repository](https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset/tree/246d3d365d5898f6fdf5240264215e49b082d743), README and Stage 2 notebook. The youth-specific relevance clarification and review procedure above are our proposed project protocol, not a claim about the source annotators' decisions.
