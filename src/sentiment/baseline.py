"""Train a fixed TF-IDF/logistic regression baseline on the shared grouped split."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from src.analysis.temporal import save_svg

LABELS = ["Positive", "Negative", "Neutral"]
CONFIG = {"max_features": 20000, "ngram_range": [1, 2], "min_df": 2,
          "sublinear_tf": True, "stop_words": None, "C": 1.0, "class_weight": "balanced",
          "max_iter": 2000, "solver": "lbfgs", "seed": 584}


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=CONFIG["max_features"], ngram_range=tuple(CONFIG["ngram_range"]),
                                  min_df=CONFIG["min_df"], sublinear_tf=True)),
        ("classifier", LogisticRegression(C=CONFIG["C"], class_weight=CONFIG["class_weight"],
                                           max_iter=CONFIG["max_iter"], solver=CONFIG["solver"], random_state=CONFIG["seed"]))])


def validate_records(rows):
    if not rows or len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must have unique identifiers")
    assignments = defaultdict(set)
    hashes = defaultdict(set)
    for row in rows:
        if not row["text"].strip() or row["sentiment_3class"] not in LABELS:
            raise ValueError("Missing text or unsupported sentiment label")
        expected = "Neutral" if row["stance_4class"] == "Mixed" else row["stance_4class"]
        if row["sentiment_3class"] != expected:
            raise ValueError("Three-class label disagrees with the proposal mapping")
        if row["split"] not in ("train", "validation", "test", "review_reserved"):
            raise ValueError("Unknown split")
        assignments[row["duplicate_group"]].add(row["split"])
        hashes[row["text_sha256"]].add(row["split"])
    if any(len(s) != 1 for s in list(assignments.values()) + list(hashes.values())):
        raise ValueError("Duplicate text/group crosses partitions")
    for split in ("train", "validation", "test"):
        if not any(r["split"] == split for r in rows):
            raise ValueError(f"Empty {split} split")


def metrics(y, predicted):
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    return {"n": len(y), "accuracy": accuracy_score(y, predicted),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {label: report[label] for label in LABELS},
            "confusion_matrix": confusion_matrix(y, predicted, labels=LABELS).tolist(), "label_order": LABELS}


def load_verified(dataset, manifest, preprocessing):
    expected = json.loads(preprocessing.read_text(encoding="utf-8"))
    for key, path in (("dataset_sha256", dataset), ("split_manifest_sha256", manifest)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]:
            raise ValueError(f"Hash mismatch: {path}")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_records(rows)
    with manifest.open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    by_id = {r["doc_id"]: r for r in manifest_rows}
    if len(by_id) != len(manifest_rows) or set(by_id) != {r["doc_id"] for r in rows}:
        raise ValueError("Dataset and manifest identifiers differ")
    for row in rows:
        for field in ("split", "text_sha256", "duplicate_group", "sentiment_3class"):
            if row[field] != by_id[row["doc_id"]][field]:
                raise ValueError(f"Manifest field differs: {field}")
    return rows, expected


def run(dataset, manifest, preprocessing, output, model_path):
    rows, provenance = load_verified(dataset, manifest, preprocessing)
    train = [r for r in rows if r["split"] == "train"]
    model = make_model()
    model.fit([r["text"] for r in train], [r["sentiment_3class"] for r in train])
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit([[0]] * len(train), [r["sentiment_3class"] for r in train])
    result = {"scope": "agreement with intermediate source labels, not independent human-gold accuracy",
              "run_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
              "dataset_sha256": provenance["dataset_sha256"], "split_manifest_sha256": provenance["split_manifest_sha256"],
              "source_commit": provenance["source_commit"],
              "versions": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "scipy", "joblib")},
              "training_n": len(train), "training_labels": dict(Counter(r["sentiment_3class"] for r in train)),
              "vocabulary_size": len(model["tfidf"].vocabulary_),
              "iterations": model["classifier"].n_iter_.tolist(), "evaluations": {}}
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for split in ("validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        y = [r["sentiment_3class"] for r in subset]
        predicted = model.predict([r["text"] for r in subset])
        probabilities = model.predict_proba([r["text"] for r in subset])
        result["evaluations"][split] = {"logistic_regression": metrics(y, predicted),
                                        "majority": metrics(y, majority.predict([[0]] * len(y))),
                                        "by_platform": {}}
        for platform in ("youtube", "snapchat"):
            indices = [i for i, row in enumerate(subset) if row["platform"] == platform]
            if indices:
                result["evaluations"][split]["by_platform"][platform] = metrics([y[i] for i in indices], [predicted[i] for i in indices])
        for row, pred, probability in zip(subset, predicted, probabilities):
            prediction_rows.append({"doc_id": row["doc_id"], "platform": row["platform"], "split": split,
                                    "reference": row["sentiment_3class"], "prediction": str(pred),
                                    "probabilities_uncalibrated": dict(zip(model.classes_, map(float, probability)))})
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ConfusionMatrixDisplay.from_predictions(y, predicted, labels=LABELS, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"TF-IDF + logistic regression: {split} (n={len(y)})\nGrouped split; intermediate source labels")
        save_svg(fig, output / f"{split}_confusion.svg"); plt.close(fig)
    errors = [r for r in prediction_rows if r["split"] == "test" and r["reference"] != r["prediction"]]
    result["test_error_counts"] = dict(Counter(f"{r['reference']} -> {r['prediction']}" for r in errors))
    result["illustrative_test_error_ids"] = [r["doc_id"] for r in errors[:5]]
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prediction_rows), encoding="utf-8")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/screen/articles.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("results/data/split_manifest.csv"))
    parser.add_argument("--preprocessing", type=Path, default=Path("results/data/preprocessing.json"))
    parser.add_argument("--output", type=Path, default=Path("results/sentiment"))
    parser.add_argument("--model", type=Path, default=Path("results/models/tfidf_logreg.joblib"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty results folder")
    if args.model.exists():
        parser.error("Choose a new model path")
    result = run(args.dataset, args.manifest, args.preprocessing, args.output, args.model)
    print(json.dumps(result["evaluations"], indent=2))


if __name__ == "__main__":
    main()
