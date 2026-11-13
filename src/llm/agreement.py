"""Measure independent review agreement before adjudication; never fills labels."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from src.llm.schema import TOPICS, validate_prediction


def read_reviews(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        if not row["doc_id"] or row["doc_id"] in result:
            raise ValueError("Missing or duplicate review identifier")
        if not row["reviewer"].strip() or row["relevance"] not in ("yes", "no", "uncertain"):
            raise ValueError("Incomplete reviewer or relevance decision")
        if row["relevance"] == "yes":
            if not row["evidence"].strip() or row["confidence"] not in ("high", "medium", "low"):
                raise ValueError("Relevant reviews need evidence and confidence")
            row["topics"] = json.loads(row["topics"])
            validate_prediction({"label": row["stance"], "topics": row["topics"], "reason": row["evidence"]})
        elif row["stance"].strip() or row["topics"].strip():
            raise ValueError("Nonrelevant/uncertain items should not have stance or topics")
        result[row["doc_id"]] = row
    if not result:
        raise ValueError("Empty review sheet")
    if len({r["reviewer"] for r in rows}) != 1:
        raise ValueError("Use one reviewer per sheet")
    return result


def categorical(pairs):
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    a, b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = sum(x == y for x, y in pairs) / n
    expected = sum(a[label] * b[label] for label in a.keys() | b.keys()) / n ** 2
    return {"n": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "reviewer_1_counts": dict(a), "reviewer_2_counts": dict(b)}


def agreement(first, second):
    if set(first) != set(second):
        raise ValueError("Review sheets must contain the same identifiers")
    if {r["reviewer"] for r in first.values()} & {r["reviewer"] for r in second.values()}:
        raise ValueError("Two different reviewers are required")
    paired = [(first[k], second[k]) for k in sorted(first)]
    relevant = [(a, b) for a, b in paired if a["relevance"] == b["relevance"] == "yes"]
    topics = [(set(a["topics"]), set(b["topics"])) for a, b in relevant]
    def mapped(value):
        return "Neutral" if value == "Mixed" else value
    return {
        "relevance": categorical([(a["relevance"], b["relevance"]) for a, b in paired]),
        "jointly_relevant": len(relevant), "excluded_from_stance": len(paired) - len(relevant),
        "stance_four_class": categorical([(a["stance"], b["stance"]) for a, b in relevant]),
        "proposal_three_class": categorical([(mapped(a["stance"]), mapped(b["stance"])) for a, b in relevant]),
        "topics": {"n": len(topics),
                   "mean_jaccard": sum(len(a & b) / len(a | b) if a | b else 1.0 for a, b in topics) / len(topics) if topics else None,
                   "per_topic": {topic: categorical([(topic in a, topic in b) for a, b in topics]) for topic in TOPICS}},
        "needs_adjudication": [a["doc_id"] for a, b in paired
                               if a["relevance"] != b["relevance"] or a["relevance"] == "uncertain"
                               or (a["relevance"] == "yes" and
                                   (a["stance"] != b["stance"] or set(a["topics"]) != set(b["topics"])))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    result = agreement(read_reviews(args.first), read_reviews(args.second))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
