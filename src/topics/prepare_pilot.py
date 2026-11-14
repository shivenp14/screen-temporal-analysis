"""Prepare blinded local topic-review sheets from unique training groups."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.topics.baseline import keyword_cues

SEED = 584
REVIEW_FIELDS = ("doc_id", "platform", "relevance", "stance", "topics", "evidence", "confidence", "notes")


def _rank(seed, purpose, row):
    value = f"{seed}|{purpose}|{row['doc_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_packet(rows, seed=SEED):
    """Select 40 pilot and 6 practice items from distinct training groups."""
    groups = {}
    for row in sorted(rows, key=lambda item: item["doc_id"]):
        if row.get("split") == "train":
            groups.setdefault(row["duplicate_group"], row)
    candidates = list(groups.values())
    if len(candidates) < 46:
        raise ValueError("Need at least 46 distinct training groups for the pilot and practice items")

    general = sorted(candidates, key=lambda row: _rank(seed, "general", row))[:20]
    selected_ids = {row["doc_id"] for row in general}
    topic_rows = {}
    for row in candidates:
        for topic in keyword_cues(row["text"]):
            topic_rows.setdefault(topic, []).append(row)
    for topic, items in topic_rows.items():
        topic_rows[topic] = sorted(items, key=lambda row: _rank(seed, f"topic:{topic}", row))

    rare_topics = sorted(topic_rows, key=lambda topic: (len(topic_rows[topic]), topic))
    enriched = []
    while len(enriched) < 20:
        added = False
        for topic in rare_topics:
            row = next((item for item in topic_rows[topic] if item["doc_id"] not in selected_ids), None)
            if row is not None:
                enriched.append(row)
                selected_ids.add(row["doc_id"])
                added = True
                if len(enriched) == 20:
                    break
        if not added:
            break
    if len(enriched) < 20:
        raise ValueError("Not enough distinct keyword-candidate groups to complete the enriched sample")

    training = sorted(general + enriched, key=lambda row: row["doc_id"])
    remaining = [row for row in candidates if row["doc_id"] not in selected_ids]
    practice = sorted(remaining, key=lambda row: _rank(seed, "practice", row))[:6]
    if len({row["duplicate_group"] for row in training + practice}) != 46:
        raise AssertionError("Selected items must come from distinct duplicate groups")
    return training, practice


def _write_articles(path, rows):
    path.write_text("".join(json.dumps({key: row[key] for key in ("doc_id", "platform", "text")},
                                       ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _write_sheet(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({"doc_id": row["doc_id"], "platform": row["platform"]} for row in rows)


def prepare(dataset, output, seed=SEED):
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    training, practice = select_packet(rows, seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_articles(output / "training_articles.jsonl", training)
    _write_articles(output / "practice_articles.jsonl", practice)
    for kind, selected in (("training", training), ("practice", practice)):
        for reviewer in (1, 2):
            _write_sheet(output / f"{kind}_reviewer_{reviewer}.csv", selected)
    metadata = {"seed": seed, "training_article_platform_rows": sum(row["split"] == "train" for row in rows),
                "training_unique_groups_available": len({row["duplicate_group"] for row in rows if row["split"] == "train"}),
                "pilot_items": len(training), "practice_items": len(practice),
                "source_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "selection": "20 hash-ranked training groups plus 20 round-robin keyword-candidate groups; six separate hash-ranked practice groups",
                "blind_review_fields": list(REVIEW_FIELDS)}
    (output / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/screen/articles.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/review_topics_pilot"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty local packet folder")
    print(json.dumps(prepare(args.dataset, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
