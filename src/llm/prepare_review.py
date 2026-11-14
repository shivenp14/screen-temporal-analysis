"""Audit the pinned release and create a blind, local annotation packet."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_COMMIT = "246d3d365d5898f6fdf5240264215e49b082d743"
FILES = ("High Confidence Relevant.csv", "Recovered Relevant Articles Doccano.csv",
         "Relevance Ambiguous.csv", "Stance Ambiguous.csv")


def digest(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def source_sha256(path):
    """Hash source bytes with CRLF line endings normalized to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(source, output, seed=584):
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Check out source commit {SOURCE_COMMIT}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                "--", "Dataset"], text=True).strip():
        raise ValueError("Source Dataset has local changes")
    tables = {}
    audit = {"source_commit": commit, "seed": seed, "files": {}, "overlaps": {}}
    for name in FILES:
        path = source / "Dataset" / name
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        tables[name] = rows
        ids = [r["doc_id"] for r in rows]
        texts = [r.get("full_text", r.get("text", "")) for r in rows]
        audit["files"][name] = {
            "sha256": source_sha256(path),
            "rows": len(rows), "columns": list(rows[0]),
            "duplicate_ids": len(ids) - len(set(ids)),
            "duplicate_normalized_texts": len(texts) - len({digest(t) for t in texts}),
            "empty_texts": sum(not t.strip() for t in texts),
            "platforms": dict(Counter(r["platform"] for r in rows)),
            "labels": dict(Counter(r.get("final_label", r.get("label", "unavailable")) for r in rows)),
        }
        if len(ids) != len(set(ids)) or any(not t.strip() for t in texts):
            raise ValueError(f"Duplicate identifiers or blank text in {name}")
    for i, a in enumerate(FILES):
        for b in FILES[i + 1:]:
            audit["overlaps"][f"{a} / {b}"] = len(
                {r["doc_id"] for r in tables[a]} & {r["doc_id"] for r in tables[b]})
    raw = {}
    for platform in ("snapchat", "youtube"):
        path = source / "Dataset" / f"news-{platform}-noagegroup-mental-2024.jsonl"
        articles = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        audit["files"][path.name] = {"rows": len(articles), "sha256": source_sha256(path)}
        for article in articles:
            body = article.get("body") or article.get("text") or ""
            if isinstance(body, list):
                body = " ".join(body)
            full_text = f"{article.get('title', '')}. {body}".strip()
            if len(full_text) < 20:
                continue
            # Stage 1 concatenates Snapchat then YouTube before assigning ids.
            index = len(raw)
            raw[f"{platform}|{index}|{article['title']}"] = {**article, "full_text": full_text}
    strata = defaultdict(list)
    for name in (FILES[0], FILES[3]):
        for row in tables[name]:
            key = (row["platform"], row.get("final_label", "unresolved"))
            strata[key].append((name, row))
    selected, seen = [], set()
    for key, candidates in sorted(strata.items()):
        count = 0
        for name, row in sorted(candidates, key=lambda item: digest(f"{seed}|{item[1]['doc_id']}")):
            text_hash = digest(row["full_text"])
            if text_hash in seen:
                continue
            seen.add(text_hash)
            selected.append((name, row))
            count += 1
            if count == 4:
                break
    # Shuffle the display order independently of label strata; reviewers see no labels.
    selected.sort(key=lambda item: digest(f"display|{seed}|{item[1]['doc_id']}"))
    output.mkdir(parents=True, exist_ok=True)
    manifest, packet, reference = [], [], []
    for name, row in selected:
        article = raw.get(row["doc_id"])
        if article is None:
            raise ValueError(f"No exact raw record match for {row['doc_id']}")
        if digest(article["full_text"]) != digest(row["full_text"]):
            raise ValueError(f"Raw and annotation text differ for {row['doc_id']}")
        text_hash = digest(row["full_text"])
        manifest.append({"doc_id": row["doc_id"], "platform": row["platform"],
                         "source_file": name, "text_sha256": text_hash})
        packet.append({"doc_id": row["doc_id"], "platform": row["platform"],
                       "title": row["title"], "publication_date": article.get("publication_date", ""),
                       "text": row["full_text"]})
        reference.append({"doc_id": row["doc_id"], "source_label": row.get("final_label", ""),
                          "source_file": name})
    write_csv(output / "manifest.csv", manifest, list(manifest[0]))
    write_csv(output / "reference_private.csv", reference, list(reference[0]))
    (output / "articles.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packet), encoding="utf-8")
    fields = ["doc_id", "reviewer", "relevance", "stance", "topics", "evidence", "confidence", "notes"]
    for reviewer in ("reviewer_1", "reviewer_2"):
        write_csv(output / f"{reviewer}.csv", [{"doc_id": r["doc_id"]} for r in packet], fields)
    audit["review_subset"] = {"size": len(packet), "platforms": dict(Counter(r["platform"] for r in packet)),
                              "strata_available": {" / ".join(k): len(v) for k, v in sorted(strata.items())},
                              "status": "pending independent human review"}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/review"))
    parser.add_argument("--seed", type=int, default=584)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; do not overwrite completed reviews")
    audit = prepare(args.source, args.output, args.seed)
    print(json.dumps(audit["review_subset"], indent=2))


if __name__ == "__main__":
    main()
