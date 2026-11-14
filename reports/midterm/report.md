# Understanding Temporal Sentiments and Topics in News Coverage of Video-Based Social Media and Youth Mental Health

Collin Vo, Pranav Goyal, Shiven Pandya, and Tia Hussain
CS 584 Natural Language Processing | Midterm working draft | November 13, 2026

**Submission status:** This draft contains verified release-audit findings and Collin's annotation and experiment preparation. It is not a completed team-results report. The course template still needs to be supplied, and the team needs to add its completed model results and figures before submission at 11:59 p.m. Eastern on November 13. Pending work is identified below rather than presented as completed.

## 1. Introduction

### Abstract

We study how news coverage of YouTube, Snapchat, and youth mental health changes over time. The project combines article-level sentiment classification, multi-label topic analysis, and comparisons across time periods and platforms. Our planned comparisons include TF-IDF logistic regression, transformer models, and zero-shot/few-shot classification. At midterm, the coordination work has established a reproducible audit of the released SCREEN files, annotation guidelines, a 36-article review packet, and an offline workflow for preparing and evaluating an LLM pilot. The audit identified intermediate annotation files rather than the final corpus described by the source, substantial repeated article text, and a missing positive Snapchat stratum in the accepted intermediate records. These findings affect how we construct splits, interpret class imbalance, and evaluate model predictions. Human review and live model evaluation remain pending in this contribution.

### Motivation and background

The project proposal asks three related questions: whether news coverage becomes more positive, negative, or neutral; which mental-health topics emerge or decline; and how topic prevalence relates to sentiment. Comparing YouTube and Snapchat adds a platform dimension to these questions. Our analysis concerns news coverage, not direct measurement of young people's mental health or evidence that one platform causes a particular outcome.

SCREEN provides the project's starting point [1]. Its source repository describes a corpus of 1,492 articles with four stance categories: Positive, Negative, Neutral, and Mixed. Our proposal [2] specifies a three-class analysis by merging Mixed into Neutral. We preserve that decision while retaining the four original labels for annotation and diagnostic reporting. This makes the relationship between the source task and our task explicit: the proposal's Neutral category includes both descriptive coverage and substantive mixed framing.

The proposal also identifies BERTopic for exploratory topic discovery, SetFit for few-shot classification, and change-point methods for temporal analysis. These remain planned extensions rather than completed experiments. The immediate priority is a reliable data and evaluation contract so later comparisons use the same records, label definitions, and split boundaries.

### Progress and obstacles

The repository already contained the project scaffold, dependency list, and pinned dataset acquisition instructions. This contribution adds an audit that can be rerun from the source commit, a shared stance/topic vocabulary, independent-review instructions, deterministic subset selection, prompt export, and prediction scoring. No completed sentiment baseline, topic baseline, or temporal figure was present in the inspected main branch, and no open PRs were returned at inspection. That describes repository-visible progress, not work teammates may have completed elsewhere.

The main obstacles are the absence of the final accepted corpus in the release, missing released topic columns, and repeated text under different record identifiers. We address these by preserving provenance, separating relevance from stance labels, reserving a review subset, and handing the data-reconciliation decisions to the relevant task owners. We do not fill missing labels with guessed values or copy source-paper model scores into our results.

## 2. Problem formulation

For article i, let x_i contain the title and body, p_i identify the platform, and t_i be the publication date. The original stance target s_i belongs to {Positive, Negative, Neutral, Mixed}. Our primary sentiment target is y_i = g(s_i), where g(Mixed) = Neutral and all other labels are unchanged. The sentiment model predicts y_i from x_i. We also retain predictions and review labels in the four-class space to examine whether an apparent Neutral prediction hides disagreement about mixed framing.

The topic task uses a binary vector z_i of length 12. Each component indicates whether a named mental-health-related topic is a primary topic of the article. The annotation protocol selects at most three topics per article; an explicitly reviewed empty set is valid. Missing topic review is not an all-zero vector. Our initial vocabulary follows the source notebook and includes Anxiety, Depression, Suicide/Self-Harm, Bullying/Cyberbullying, Body Image/Eating Disorders, Addiction/Overuse, Sleep Disruption, Policy/Lawsuits/Bans, School/Education, Therapy/Access to Care, Stigma/Awareness, and Parental Controls/Safety Features.

The proposal's illustrative topics also mention substance use. It is not in the source notebook's 12-topic vocabulary. We do not treat it as a released label. Shiven should decide whether to add it in a versioned extension after examining examples; any change needs corresponding guidelines and consistent annotation before evaluation.

For platform p and time bin b, sentiment prevalence is the number of eligible articles with predicted class c divided by the total eligible articles in that platform/bin. Topic prevalence uses the same denominator for each topic, so topic percentages may sum above 100%. The final analysis should publish bin counts, missing-date exclusions, and the duplicate policy alongside proportions. Small bins can produce unstable apparent changes; quarterly or yearly aggregation may be more defensible than monthly plots when support is sparse.

## 3. Methods and implementation

### Annotation and review

The guidelines define relevance before stance. Articles must connect the named platform to youth mental health or well-being; merely hosting a video is insufficient. Reviewers judge the full article's framing rather than its emotional wording or their own beliefs. Positive and Negative require substantive benefit or harm framing, Neutral indicates no clear leaning, and Mixed requires meaningful weight on both sides. Reviewers record evidence and confidence so borderline decisions can be discussed consistently.

The packet samples four articles from each available platform/stance group in the accepted intermediate table, then four unresolved stance articles per platform. A stable hash ordering with seed 584 makes the selection reproducible. The resulting 36 items contain 20 YouTube and 16 Snapchat records. Exact whitespace-normalized duplicate texts are excluded from the packet. This is a diagnostic sample designed to expose disagreement and minority classes; it is not a random estimate of corpus prevalence.

Two reviewers should independently annotate the same packet without source labels or model predictions. After comparing relevance decisions, we will report stance agreement and Cohen's kappa on jointly relevant items, with sample sizes and excluded cases. For topics, we plan per-topic binary agreement and set Jaccard similarity. A third reviewer should adjudicate disagreements, preserving both original sheets and the final rationale. No independent reviews or agreement values are reported yet.

### LLM experiment preparation

The prompt uses the same four stance definitions and 12-topic vocabulary as the review guidelines. It requests structured output containing a stance label, up to three topics, and a short reason. Article content is explicitly treated as data. The exporter records the exact messages, text and prompt hashes, truncation flag, character limit, and demonstration identifiers. The default limit is the first 12,000 characters; this is a character budget, not a token budget, and truncation must be examined before interpreting results.

A few-shot run will use reviewed training examples with the same model and settings as the zero-shot run. The exporter rejects demonstrations that share an identifier or normalized text with an evaluation item and requires a training-split marker. Near-duplicate checks remain part of Tia's split preparation. Model execution is separate from prompt preparation, allowing the team to choose a provider and preserve the original responses without embedding credentials in the repository.

The evaluator rejects unadjudicated reference labels and unexpected or duplicate prediction identifiers. It records missing, invalid, and failed responses rather than replacing them with Neutral. The primary three-class scores apply the proposal's mapping to both references and predictions. Four-class scores remain available as a diagnostic. Topic metrics are computed only where topic review is complete, and their denominator is reported separately.

### Integration with the planned baselines

Pranav's TF-IDF logistic regression baseline will provide a reproducible first comparison for sentiment, with accuracy, macro precision/recall/F1, class support, and a confusion matrix. Shiven's one-vs-rest logistic regression topic baseline depends on usable topic labels; without them, the immediate deliverable is exploratory topic analysis and an annotation plan. These are the assigned methods, not claims of trained models in this contribution.

All models should use the same versioned dataset and split manifest. Repeated or near-duplicate text must stay in one partition, including copies under different platform identifiers. The reviewed diagnostic subset must be excluded from training and few-shot examples. If a platform-specific analysis retains the same text for both platforms, the shared text group must still be respected during splitting. Transformer and temporal comparisons should wait until these basic evaluation conditions are in place.

## 4. Datasets, experiments, and findings

### Released data inspected

The audit uses SCREEN commit `246d3d365d5898f6fdf5240264215e49b082d743`. File hashes and counts are stored in `results/metrics/release_audit.json`. The raw JSONL files contain 676 Snapchat rows and 1,326 YouTube rows, totaling 2,002 raw records. These raw records are not the same as the source's reported final corpus.

| Released CSV | Rows | What the labels support |
| --- | ---: | --- |
| High Confidence Relevant | 932 | Accepted intermediate four-class stance labels |
| Stance Ambiguous | 342 | Two provisional stance votes; no final label |
| Recovered Relevant Articles Doccano | 220 | Relevance decisions, not stance or topic labels |
| Relevance Ambiguous | 463 | Relevance-review candidates |

All 220 recovered relevance identifiers also occur in the 463-row relevance-review file. The other pairwise identifier intersections are zero. The accepted, stance-ambiguous, and recovered sets therefore contain 1,494 distinct identifiers, but this arithmetic does not reconstruct the source's final 1,492 accepted articles. The final filtering decisions and topic-bearing output are not released in these CSVs. Relevance label `MH_Relevant` must not be interpreted as a stance class.

### Label distribution and duplicates

| Platform | Negative | Mixed | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Snapchat | 306 | 87 | 14 | 0 | 407 |
| YouTube | 354 | 93 | 7 | 71 | 525 |
| Total | 660 | 180 | 21 | 71 | 932 |

These are row-level counts from the accepted intermediate table, before deduplication, not the published final-corpus distribution. Under the proposal's mapping, the totals become 660 Negative, 201 Neutral, and 71 Positive. Negative rows represent 70.8% of this table. An always-Negative prediction would therefore achieve 70.8% accuracy on these same rows; this is a descriptive majority-class reference, not a held-out baseline result. Macro-F1 and class support are needed to avoid mistaking majority prediction for balanced performance.

The accepted table contains no repeated identifiers but has 318 repeated normalized texts beyond the first occurrence, leaving 614 distinct texts. The other CSVs contain 30, 55, and 59 excess repeated texts in the recovered, relevance-ambiguous, and stance-ambiguous files, respectively. These counts normalize whitespace only; they do not measure near duplicates or establish that duplicate rows should always be removed. They do show why identifier uniqueness alone is insufficient to prevent leakage.

The absence of accepted positive Snapchat rows is a limitation of this intermediate subset, not evidence that positive Snapchat coverage does not exist. It prevents balanced sampling of all eight platform/stance combinations and may make class-stratified platform evaluation infeasible. The source's final-corpus summary lists positive Snapchat articles, reinforcing the need to distinguish the release stages.

### Work executed and limits

The audit and packet-generation workflow ran successfully on the pinned source. Every selected record's identifier and normalized full text matched the raw source before its date was attached. The workflow produced 36 blind article records, two blank reviewer sheets, a private reference lookup, and 36 zero-shot request records. One of the 36 requests exceeds the 12,000-character limit and is flagged as truncated. Eight unit tests passed, covering label mapping, failure accounting, topic validation, example leakage checks, and agreement edge cases. A second packet-generation run reproduced the committed manifest and audit exactly.

No live zero-shot or few-shot run was executed because no model API credentials were configured. Prompt files and synthetic tests are not model performance results. Human review is also pending, so no agreement score or human-gold model score is available. The baseline and temporal sections require the respective team members' actual results. We will not substitute the original dataset authors' benchmark numbers for our own experiments.

### Next experiments

First, reconcile the released stages or obtain the final table, freeze the duplicate policy, and construct shared splits. Second, complete independent review and adjudication of the diagnostic packet. Third, compare the sentiment baseline with a zero-shot run and a controlled few-shot run on the same eligible items, reporting coverage and errors. Topic modeling can proceed with verified labels or remain explicitly exploratory. Finally, compare temporal summaries under monthly, quarterly, and yearly aggregation, checking whether visible changes survive changes in bin size and classifier choice. Any effect of the Mixed-to-Neutral mapping should be shown using the retained four-class diagnostics.

## 5. Project management

| Member | Responsibility | Repository-visible status / next handoff |
| --- | --- | --- |
| Collin Vo | Coordination, annotation, agreement evaluation, LLM experiments, report assembly | Guidelines, release audit, review packet, prompt/scoring workflow, and this draft prepared. Human review and model execution pending. |
| Tia Hussain | Data preparation, splits, temporal analysis, figures | Supply reconciled data, duplicate groups, date validation, split manifest, and temporal figures. Implementation/results were not present in inspected main. |
| Pranav Goyal | Sentiment baseline, class imbalance, evaluation | Supply runnable TF-IDF baseline, configuration, held-out metrics, and confusion matrix. Implementation/results were not present in inspected main. |
| Shiven Pandya | Topic labels, multi-label baseline or exploration | Confirm label availability and vocabulary; supply model results or a documented exploration/annotation plan. Implementation/results were not present in inspected main. |

The proposal scheduled exploration and baselines in October, the midterm and model comparisons in November, and a final presentation in December. The midterm deadline is November 13, 2026, at 11:59 p.m. Eastern. Before submission, the team should integrate the available contributions, check that every score links to a run and split, transfer this draft into the provided course template, verify that the rendered report is at least four pages, and remove internal submission notes. The supplied guideline does not itself include the template, so template formatting remains unverified.

After the midterm, the next milestones are independent annotation and agreement analysis, a first live LLM comparison, verified topic targets or a revised topic-exploration plan, and temporal robustness checks. Exact follow-up dates should be agreed by the team. Shared code review and report editing should preserve each contributor's ownership and avoid reporting planned methods as completed experiments.

## 6. Key references

1. Singh, J., Jotaniya, H., Xie, T., Bae, S. W., and Wang, P. (2026). *SCREEN: A News Media Stance Dataset for Understanding Youth Mental Health Narratives Around Video-Based Social Media.* Source repository and Stage 1/2 notebooks, [pinned release](https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset/tree/246d3d365d5898f6fdf5240264215e49b082d743). Bibliographic details follow the source repository; findings here come from the released files.
2. Vo, C., Goyal, P., Pandya, S., and Hussain, T. *CS 584 Project Proposal.* Team document, supplied November 13, 2026. The proposed three-class mapping, methods, and original timeline are retained here.
3. *CS 584 Project Midterm Report Guidelines.* Course handout supplied with this task. Requires updated progress, formulation, methods, experiments, management, references, and at least four pages in the course template.
4. Additional literature retained as planned background in the proposal: Grootendorst (2022), *BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure*, arXiv:2203.05794; Tunstall et al. (2022), *Efficient Few-Shot Learning Without Prompts*, arXiv:2209.11055; Truong, Oudre, and Vayatis (2020), *Selective Review of Offline Change Point Detection Methods*, Signal Processing 167, 107299. No new findings from these papers are claimed in this draft.
