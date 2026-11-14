"""Build a provisional analysis table from accepted intermediate SCREEN records."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.llm.prepare_review import SOURCE_COMMIT, digest, write_csv
from src.llm.schema import STANCE_LABELS


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_date(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(timezone.utc)
        return stamp.date().isoformat()
    except (ValueError, TypeError, AttributeError):
        return None


def raw_records(source):
    records = {}
    for platform in ("snapchat", "youtube"):
        path = source / "Dataset" / f"news-{platform}-noagegroup-mental-2024.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            body = row.get("body") or row.get("text") or ""
            if isinstance(body, list):
                body = " ".join(body)
            text = f"{row.get('title', '')}. {body}".strip()
            if len(text) < 20:
                continue
            doc_id = f"{platform}|{len(records)}|{row.get('title', '')}"
            records[doc_id] = {"text": text, "date_raw": row.get("publication_date")}
    return records


def duplicate_groups(texts, threshold=0.85):
    """Connected components of >=threshold token-five-shingle Jaccard pairs.

    Exact hashes are input keys. Short documents use word sets. Quadratic pair
    checks are appropriate for this small corpus, not a large collection.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1]")
    keys = sorted(texts)
    parents = {key: key for key in keys}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    shingles = {}
    for key in keys:
        words = re.findall(r"\w+", texts[key].lower())
        shingles[key] = {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))} or set(words)
    pairs = 0
    for i, key in enumerate(keys):
        a = shingles[key]
        for other in keys[:i]:
            b = shingles[other]
            if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            intersection = len(a & b)
            if intersection / (len(a) + len(b) - intersection) >= threshold:
                left, right = find(key), find(other)
                parents[max(left, right)] = min(left, right)
                pairs += 1
    return {key: find(key) for key in keys}, pairs


def assign_splits(rows, reserved_groups, seed=584):
    eligible = [row for row in rows if row["duplicate_group"] not in reserved_groups]
    if not eligible:
        raise ValueError("No eligible training records")
    y = np.array([r["sentiment_3class"] for r in eligible])
    groups = np.array([r["duplicate_group"] for r in eligible])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(folds.split(np.zeros(len(y)), y, groups)):
        for index in indices:
            eligible[index]["split"] = "test" if fold == 0 else "validation" if fold == 1 else "train"
    for row in rows:
        if row["duplicate_group"] in reserved_groups:
            row["split"] = "review_reserved"
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row["duplicate_group"]].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise AssertionError("Duplicate group crosses partitions")
    for split in ("train", "validation", "test"):
        if {r["sentiment_3class"] for r in rows if r["split"] == split} != {"Positive", "Negative", "Neutral"}:
            raise ValueError(f"{split} does not contain all three classes")


def prepare(source, review_manifest, output, public, seed=584):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", "Dataset"], text=True).strip():
        raise ValueError("Use the clean pinned SCREEN source")
    raw = raw_records(source)
    accepted = read_csv(source / "Dataset" / "High Confidence Relevant.csv")
    buckets, excluded = defaultdict(list), []
    for row in accepted:
        text = " ".join(row["full_text"].split())
        if not text or row["final_label"] not in STANCE_LABELS:
            excluded.append({"doc_id": row["doc_id"], "reason": "missing_text_or_label"})
            continue
        original = raw.get(row["doc_id"])
        if original is None or digest(original["text"]) != digest(text):
            raise ValueError(f"Source join mismatch: {row['doc_id']}")
        row = {**row, "text": text, "text_sha256": digest(text),
               "publication_date_raw": original["date_raw"], "publication_date": parse_date(original["date_raw"])}
        buckets[(row["platform"], row["text_sha256"])].append(row)
    records = []
    for _, copies in sorted(buckets.items()):
        if len({r["final_label"] for r in copies}) != 1:
            excluded.extend({"doc_id": r["doc_id"], "reason": "conflicting_exact_duplicate_stance"} for r in copies)
            continue
        # Earliest valid date represents an exact-text/platform article group.
        copies.sort(key=lambda r: (r["publication_date"] or "9999", r["doc_id"]))
        row = copies[0]
        excluded.extend({"doc_id": r["doc_id"], "reason": "duplicate_same_platform_text"} for r in copies[1:])
        records.append({"doc_id": row["doc_id"], "platform": row["platform"], "title": row["title"],
                        "text": row["text"], "text_sha256": row["text_sha256"],
                        "publication_date_raw": row["publication_date_raw"], "publication_date": row["publication_date"],
                        "source_ids": [r["doc_id"] for r in copies], "source_commit": commit,
                        "source_file": "High Confidence Relevant.csv", "label_provenance": "intermediate_accepted",
                        "stance_4class": row["final_label"],
                        "sentiment_3class": "Neutral" if row["final_label"] == "Mixed" else row["final_label"],
                        "topics": None})
    texts = {r["text_sha256"]: r["text"] for r in records}
    review_hashes = set()
    for review in read_csv(review_manifest):
        original = raw[review["doc_id"]]
        if digest(original["text"]) != review["text_sha256"]:
            raise ValueError("Review manifest text hash does not match the source")
        review_hashes.add(review["text_sha256"])
        texts[review["text_sha256"]] = original["text"]
    groups, pairs = duplicate_groups(texts)
    reserved_groups = {groups[h] for h in review_hashes}
    for row in records:
        row["duplicate_group"] = groups[row["text_sha256"]]
    assign_splits(records, reserved_groups, seed)
    records.sort(key=lambda r: r["doc_id"])
    output.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    dataset = output / "articles.jsonl"
    dataset.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    fields = ["doc_id", "platform", "text_sha256", "duplicate_group", "split", "stance_4class", "sentiment_3class", "publication_date"]
    manifest = public / "split_manifest.csv"
    write_csv(manifest, [{k: r[k] for k in fields} for r in records], fields)
    write_csv(public / "excluded_records.csv", excluded, ["doc_id", "reason"])
    dates = [r["publication_date"] for r in records if r["publication_date"]]
    stats = {"dataset_scope": "accepted intermediate records only; not the final 1492-row corpus", "source_commit": commit,
             "seed": seed, "input_rows": len(accepted), "retained_rows": len(records),
             "exclusions": dict(Counter(r["reason"] for r in excluded)),
             "missing_dates": sum(r["publication_date"] is None for r in records),
             "date_min": min(dates) if dates else None, "date_max": max(dates) if dates else None,
             "unique_texts": len({r["text_sha256"] for r in records}),
             "duplicate_groups": len({r["duplicate_group"] for r in records}), "near_duplicate_pairs_including_review": pairs,
             "similarity": "token-five-shingle Jaccard >= 0.85; connected components",
             "splits": {s: {"rows": sum(r["split"] == s for r in records),
                              "labels": dict(Counter(r["sentiment_3class"] for r in records if r["split"] == s))}
                        for s in ("train", "validation", "test", "review_reserved")},
             "platform_labels": {p: dict(Counter(r["sentiment_3class"] for r in records if r["platform"] == p)) for p in ("youtube", "snapchat")},
             "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
             "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "review_manifest_sha256": hashlib.sha256(review_manifest.read_bytes()).hexdigest()}
    (public / "preprocessing.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/screen-source"))
    parser.add_argument("--review-manifest", type=Path, default=Path("annotations/review_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/screen"))
    parser.add_argument("--public", type=Path, default=Path("results/data"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Choose an empty processed-data folder: {args.output}")
    print(json.dumps(prepare(args.source, args.review_manifest, args.output, args.public), indent=2))


if __name__ == "__main__":
    main()
