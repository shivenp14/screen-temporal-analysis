# SCREEN Temporal Analysis

A CS 584 Natural Language Processing project at Stevens Institute of Technology exploring how news coverage of video-based social media and youth mental health evolves over time.

## Overview

Using the [SCREEN dataset](https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset), containing 1,492 news articles about YouTube, Snapchat, and youth mental health, this project focuses on two main tasks:

- **Temporal Sentiment Analysis:** Classify articles as positive, negative, or neutral and analyze how sentiment changes over time.
- **Temporal Multi-Label Topic Analysis:** Identify mental-health-related topics within articles and examine how their prevalence and relationships with sentiment evolve.

## Planned Methods

- TF-IDF with Logistic Regression
- RoBERTa and DeBERTa
- Zero-shot and few-shot classification
- SetFit and BERTopic
- Temporal trend analysis and visualization

## Python setup

Use Python 3.12 to keep the team's environments consistent. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Each team member creates their own `.venv`; the shared `requirements.txt` records project dependencies. Add dependencies to that file when introducing them, and rerun the install command after pulling updates. Run `deactivate` when finished.

## Team

- Collin Vo
- Pranav Goyal
- Shiven Pandya
- Tia Hussain

## Dataset

Singh et al. (2026). *SCREEN: A News Media Stance Dataset for Understanding Youth Mental Health Narratives Around Video-Based Social Media.*

[SCREEN Dataset Repository](https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset)

The raw files are stored locally in `data/raw/` and are excluded from Git. To reproduce the current copy, run from the repository root:

```sh
git clone https://github.com/LEAF-Lab-Stevens/SCREEN-Dataset.git /tmp/screen-dataset-source
git -C /tmp/screen-dataset-source checkout 246d3d365d5898f6fdf5240264215e49b082d743
mkdir -p data/raw
cp /tmp/screen-dataset-source/Dataset/* data/raw/
cp /tmp/screen-dataset-source/LICENSE data/raw/
```

The source repository supplies two original news JSONL files and four CSV files from intermediate relevance and stance annotation stages. The files are not a single ready-made 1,492-row labeled table; inspect the annotation pipeline and reconcile records before building the analysis dataset. The source repository provides its data under the MIT license; its license is included alongside the local copy.
