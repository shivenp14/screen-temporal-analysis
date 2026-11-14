import unittest
import csv
import tempfile
from pathlib import Path

from src.llm.classify import build_requests, score
from src.llm.schema import validate_prediction
from src.llm.agreement import agreement, categorical, read_reviews


class PilotTests(unittest.TestCase):
    def test_prompt_hides_source_labels_and_records_truncation(self):
        requests = build_requests([{"doc_id": "1", "platform": "youtube", "text": "abcdef",
                                    "source_label": "SECRET"}], max_chars=3)
        self.assertTrue(requests[0]["truncated"])
        self.assertNotIn("SECRET", str(requests))
        self.assertIn("abc", requests[0]["messages"][-1]["content"])

    def test_few_shot_rejects_duplicate_text_and_nontraining_examples(self):
        article = {"doc_id": "1", "platform": "youtube", "text": "a b"}
        example = {"doc_id": "2", "platform": "youtube", "text": "a  b", "split": "train",
                   "answer": {"label": "Neutral", "topics": [], "reason": "descriptive"}}
        with self.assertRaises(ValueError):
            build_requests([article], [example])
        example.update(text="different", split="test")
        with self.assertRaises(ValueError):
            build_requests([article], [example])

    def test_errors_are_not_neutral_or_dropped(self):
        gold = [{"doc_id": str(i), "stance": label, "review_status": "adjudicated"}
                for i, label in enumerate(("Positive", "Negative", "Neutral", "Mixed"))]
        predictions = [
            {"doc_id": "0", "answer": {"label": "Positive", "topics": [], "reason": "benefits"}},
            {"doc_id": "1", "error": "timeout"},
            {"doc_id": "2", "answer": {"label": "invalid", "topics": [], "reason": "bad"}}]
        result = score(gold, predictions)
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["macro_f1"], 0.25)
        self.assertEqual(result["coverage"], 0.25)
        self.assertEqual(result["errors"], {"model_error": 1, "invalid_response": 1, "missing": 1})
        self.assertEqual(result["confusion_matrix"]["Neutral"]["ERROR"], 1)

    def test_rejects_unreviewed_gold_and_unexpected_predictions(self):
        with self.assertRaises(ValueError):
            score([{"doc_id": "1", "stance": "Negative"}], [])
        with self.assertRaises(ValueError):
            score([{"doc_id": "1", "stance": "Negative", "review_status": "adjudicated"}], [{"doc_id": "other"}])

    def test_topic_scores_use_reviewed_subset(self):
        gold = [{"doc_id": "1", "stance": "Negative", "review_status": "adjudicated", "topics": ["Anxiety"]},
                {"doc_id": "2", "stance": "Negative", "review_status": "adjudicated"}]
        predictions = [{"doc_id": "1", "answer": {"label": "Negative", "topics": ["Anxiety", "Depression"], "reason": "test"}}]
        result = score(gold, predictions)
        self.assertEqual(result["topics"]["n_with_gold"], 1)
        self.assertAlmostEqual(result["topics"]["micro_f1"], 2 / 3)

    def test_invalid_topics(self):
        for topics in (["Unknown"], ["Anxiety", "Anxiety"], "Anxiety", [{}]):
            with self.assertRaises(ValueError):
                validate_prediction({"label": "Negative", "topics": topics, "reason": "test"})

    def test_proposal_mapping_preserves_four_class_diagnostic(self):
        gold = [{"doc_id": "1", "stance": "Mixed", "review_status": "adjudicated"}]
        predictions = [{"doc_id": "1", "answer": {"label": "Neutral", "topics": [], "reason": "balanced"}}]
        result = score(gold, predictions)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["proposal_three_class"]["accuracy"], 1)
        self.assertEqual(result["proposal_three_class"]["per_class"]["Neutral"]["support"], 1)

    def test_agreement_reports_undefined_kappa_and_mapping(self):
        self.assertIsNone(categorical([("Neutral", "Neutral")])["kappa"])
        a = {"1": {"doc_id": "1", "reviewer": "one", "relevance": "yes", "stance": "Mixed", "topics": []}}
        b = {"1": {"doc_id": "1", "reviewer": "two", "relevance": "yes", "stance": "Neutral", "topics": []}}
        result = agreement(a, b)
        self.assertEqual(result["stance_four_class"]["agreement"], 0)
        self.assertEqual(result["proposal_three_class"]["agreement"], 1)
        self.assertEqual(result["topics"]["mean_jaccard"], 1)
        self.assertEqual(result["needs_adjudication"], ["1"])
        with self.assertRaises(ValueError):
            agreement(a, a)

    def test_skipped_reviews_are_accepted_and_excluded_from_agreement(self):
        fields = ["doc_id", "reviewer", "relevance", "stance", "topics", "evidence", "confidence", "notes"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewer.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"doc_id": "skip", "reviewer": "one", "relevance": "skipped",
                                 "notes": "Stopped reading distressing content"})
            reviews = read_reviews(path)
        a = {"skip": reviews["skip"],
             "scored": {"doc_id": "scored", "reviewer": "one", "relevance": "yes",
                        "stance": "Mixed", "topics": []},
             "uncertain": {"doc_id": "uncertain", "reviewer": "one", "relevance": "uncertain",
                           "stance": "", "topics": ""}}
        b = {"skip": {"doc_id": "skip", "reviewer": "two", "relevance": "yes",
                      "stance": "Negative", "topics": ["Anxiety"]},
             "scored": {"doc_id": "scored", "reviewer": "two", "relevance": "yes",
                        "stance": "Neutral", "topics": []},
             "uncertain": {"doc_id": "uncertain", "reviewer": "two", "relevance": "uncertain",
                           "stance": "", "topics": ""}}
        result = agreement(a, b)
        self.assertEqual(result["relevance"]["n"], 2)
        self.assertEqual(result["stance_four_class"]["n"], 1)
        self.assertEqual(result["skipped_items"]["n_skipped_by_either"], 1)
        self.assertEqual(result["skipped_items"]["n_skipped_by_both"], 0)
        self.assertEqual(result["skipped_items"]["doc_ids"], ["skip"])
        self.assertNotIn("skip", result["needs_adjudication"])
        self.assertIn("uncertain", result["needs_adjudication"])


if __name__ == "__main__":
    unittest.main()


def test_source_hash_normalizes_crlf(tmp_path):
    from src.llm.prepare_review import source_sha256

    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    lf.write_bytes(b"a,b\nquoted,\"line 1\nline 2\"\n")
    crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))
    assert source_sha256(lf) == source_sha256(crlf)
    crlf.write_bytes(b"different\r\n")
    assert source_sha256(lf) != source_sha256(crlf)
