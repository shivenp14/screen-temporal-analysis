"""Audit released topic fields and explore training texts when gold topics are absent."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analysis.temporal import save_svg
from src.llm.prepare_review import SOURCE_COMMIT
from src.llm.schema import TOPICS

# Search cues for discovery, not rules that assign topic annotations.
CUES = {
    "Anxiety": r"\banxiety\b|\banxious\b",
    "Depression": r"\bdepression\b|\bdepressive\b",
    "Suicide/Self-Harm": r"\bsuicid\w*\b|\bself[- ]harm\b",
    "Bullying/Cyberbullying": r"\b(?:cyber)?bully\w*\b|\bharassment\b",
    "Body Image/Eating Disorders": r"\bbody image\b|\beating disorder\w*\b|\banorexia\b|\bbulimia\b",
    "Addiction/Overuse": r"\baddict\w*\b|\boveruse\b|\bcompulsive\b",
    "Sleep Disruption": r"\bsleep\w*\b|\binsomnia\b",
    "Policy/Lawsuits/Bans": r"\blawsuit\w*\b|\blitigation\b|\bban(?:s|ned|ning)?\b|\bregulat\w*\b",
    "School/Education": r"\bschool\w*\b|\beducation\b|\bclassroom\w*\b",
    "Therapy/Access to Care": r"\btherap(?:y|ist|ists)\b|\bcounsel(?:ing|ling)\b|\bmental health care\b",
    "Stigma/Awareness": r"\bstigma\b|\bawareness\b",
    "Parental Controls/Safety Features": r"\bparental controls?\b|\bsafety features?\b|\bparental supervision\b",
}


def source_sha256(path):
    """Hash source bytes with CRLF normalized to LF, as in the release audit."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def topic_inventory(source):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", "Dataset"], text=True).strip():
        raise ValueError("Use the clean pinned source")
    inventory = {}
    for path in sorted((source / "Dataset").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            rows = list(reader)
        topic_fields = [field for field in fields if "topic" in field.lower()]
        inventory[path.name] = {"rows": len(rows), "columns": fields, "topic_columns": topic_fields,
                                "sha256": source_sha256(path)}
    if any(item["topic_columns"] for item in inventory.values()):
        raise ValueError("Topic fields found: inspect their provenance before using the no-label fallback")
    return inventory


def training_texts(rows):
    unique = {}
    for row in rows:
        if row["split"] == "train":
            unique.setdefault(row["text_sha256"], row)
    if not unique:
        raise ValueError("No training texts")
    return [unique[key] for key in sorted(unique)]


def keyword_cues(text):
    return [topic for topic, pattern in CUES.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def run(source, dataset, preprocessing, output, components=8):
    inventory = topic_inventory(source)
    provenance = json.loads(preprocessing.read_text(encoding="utf-8"))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != provenance["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r.get("topics") is not None for r in rows):
        raise ValueError("Processed topic labels are present; inspect them before fallback exploration")
    train = training_texts(rows)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=.95, stop_words="english")
    matrix = vectorizer.fit_transform([r["text"] for r in train])
    if not 2 <= components <= min(matrix.shape):
        raise ValueError("Component count must fit the training matrix and be at least two")
    model = NMF(n_components=components, init="nndsvda", random_state=584, max_iter=1000, tol=1e-4)
    weights = model.fit_transform(matrix)
    words = vectorizer.get_feature_names_out()
    assigned = np.argmax(weights, axis=1)
    has_weight = np.sum(weights, axis=1) > 0
    clusters = []
    for index, component in enumerate(model.components_):
        top = np.argsort(component)[::-1][:12]
        representatives = np.argsort(weights[:, index])[::-1][:3]
        clusters.append({"component": index, "top_terms": words[top].tolist(),
                         "top_term_weights": component[top].tolist(),
                         "strongest_component_count": int(np.sum((assigned == index) & has_weight)),
                         "representative_training_ids": [train[i]["doc_id"] for i in representatives if weights[i, index] > 0]})
    cues = Counter()
    candidate_rows = []
    for row in train:
        found = keyword_cues(row["text"])
        cues.update(found)
        candidate_rows.append({"doc_id": row["doc_id"], "keyword_candidates_not_labels": found})
    result = {"status": "exploratory_only_no_topic_gold", "source_commit": SOURCE_COMMIT,
              "run_utc": datetime.now(timezone.utc).isoformat(), "released_csv_inventory": inventory,
              "dataset_sha256": provenance["dataset_sha256"], "split_manifest_sha256": provenance["split_manifest_sha256"],
              "training_article_platform_rows": sum(r["split"] == "train" for r in rows),
              "unique_training_texts": len(train), "held_out_texts_used": 0,
              "vocabulary": list(TOPICS), "config": {"n_components": components, "seed": 584,
                  "max_features": 10000, "ngram_range": [1, 2], "min_df": 2, "max_df": .95,
                  "stop_words": "english", "init": "nndsvda", "max_iter": 1000, "tol": 1e-4},
              "versions": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "scipy")},
              "iterations": model.n_iter_, "reconstruction_error_not_f1": float(model.reconstruction_err_),
              "zero_weight_texts": int(np.sum(~has_weight)), "components": clusters,
              "keyword_candidate_counts_not_prevalence": {topic: cues[topic] for topic in TOPICS},
              "micro_f1": None, "macro_f1": None, "per_topic_f1": None,
              "metrics_unavailable_reason": "Released CSVs have no topic columns; no adjudicated topic gold is available."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "exploration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "keyword_candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidate_rows), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    labels = [f"{r['component']}: " + ", ".join(r["top_terms"][:3]) for r in clusters]
    bars = ax.barh(labels, [r["strongest_component_count"] for r in clusters], color="#447799")
    ax.bar_label(bars); ax.invert_yaxis(); ax.set_xlabel("Unique training texts with this strongest component")
    ax.set_title("NMF exploration of training texts\nComponents are not annotated mental-health topics")
    save_svg(fig, output / "nmf_components.svg"); plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/screen-source"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/screen/articles.jsonl"))
    parser.add_argument("--preprocessing", type=Path, default=Path("results/data/preprocessing.json"))
    parser.add_argument("--output", type=Path, default=Path("results/topics"))
    parser.add_argument("--components", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty results folder")
    result = run(args.source, args.dataset, args.preprocessing, args.output, args.components)
    print(json.dumps({"unique_training_texts": result["unique_training_texts"], "iterations": result["iterations"],
                      "components": result["components"], "keyword_counts": result["keyword_candidate_counts_not_prevalence"]}, indent=2))


if __name__ == "__main__":
    main()
